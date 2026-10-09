import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { setImmediate } from "node:timers/promises";
import test from "node:test";
import ts from "typescript";

// Use the existing TypeScript compiler; no browser or additional test dependency.
const source = await readFile(new URL("../src/services/auth-client.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
}).outputText;
const { AuthClient, AuthError } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);

const user = { id: "1", email: "test@example.com", display_name: "测试用户" };
const tenant = { id: "10", code: "TEST", name: "测试租户", role: "ADMIN" };
const session = (token = "initial", currentTenant = tenant) => ({
  access_token: token, token_type: "Bearer", expires_in: 900, user, tenant: currentTenant,
});
const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { "Content-Type": "application/json" },
});
const expired = () => json({ detail: { code: "ACCESS_TOKEN_EXPIRED", message: "过期" } }, 401);
function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}
async function signedClient(handler, options = {}) {
  const calls = [];
  const client = new AuthClient({
    ...options,
    fetcher: async (url, init) => {
      calls.push({ url, init });
      return url.endsWith("/select-tenant") ? json(session()) : handler(url, init);
    },
  });
  await client.selectTenant("selection", "10");
  return { client, calls };
}

test("email login grants only a selection ticket; tenant confirmation grants identity", async () => {
  const calls = [];
  const client = new AuthClient({ fetcher: async (url, init) => {
    calls.push({ url, init });
    return url.endsWith("/login")
      ? json({ selection_token: "selection", user, tenants: [tenant] }) : json(session());
  } });
  const selection = await client.login(user.email, "password");
  assert.equal(client.snapshot.identity, null);
  assert.equal(selection.tenants[0].id, "10");
  await client.selectTenant(selection.selection_token, tenant.id);
  assert.equal(client.snapshot.identity.tenant.id, "10");
  assert.deepEqual(JSON.parse(calls[0].init.body), { email: user.email, password: "password" });
  assert.deepEqual(JSON.parse(calls[1].init.body), { selection_token: "selection", tenant_id: "10" });
  assert.ok(calls.every(({ init }) => init.credentials === "include"));
  assert.ok(calls.every(({ init }) => init.headers.get("X-Requested-With") === "XMLHttpRequest"));
});

test("concurrent expired fetches share one refresh and each retry with the new token", async () => {
  const refreshStarted = deferred();
  const releaseRefresh = deferred();
  const { client, calls } = await signedClient(async (url, init) => {
    if (url.endsWith("/refresh")) {
      refreshStarted.resolve();
      await releaseRefresh.promise;
      return json(session("fresh"));
    }
    return init.headers.get("Authorization") === "Bearer initial" ? expired() : json({ ok: true });
  });
  const key = client.snapshot.sessionKey;
  const requests = [client.fetch("/first"), client.fetch("/second"), client.fetch("/third")];
  await refreshStarted.promise;
  await setImmediate();
  assert.equal(calls.filter(({ url }) => url.endsWith("/refresh")).length, 1);
  releaseRefresh.resolve();
  assert.ok((await Promise.all(requests)).every((response) => response.ok));
  assert.equal(client.snapshot.sessionKey, key);
  assert.equal(calls.filter(({ url }) => !url.startsWith("/api/auth")).length, 6);
  assert.ok(calls.slice(-3).every(({ init }) => init.headers.get("Authorization") === "Bearer fresh"));
});

test("late expired response uses the token already refreshed by another request", async () => {
  const slowResponse = deferred();
  const slowStarted = deferred();
  const { client, calls } = await signedClient(async (url, init) => {
    if (url.endsWith("/refresh")) return json(session("fresh"));
    if (init.headers.get("Authorization") === "Bearer fresh") return json({ ok: true });
    if (url === "/slow") {
      slowStarted.resolve();
      await slowResponse.promise;
    }
    return expired();
  });
  const slow = client.fetch("/slow");
  await slowStarted.promise;
  await client.fetch("/fast");
  slowResponse.resolve();
  assert.equal((await slow).status, 200);
  assert.equal(calls.filter(({ url }) => url.endsWith("/refresh")).length, 1);
});

test("a second expiry is never retried a third time and clears the session", async () => {
  const { client, calls } = await signedClient(async (url) => url.endsWith("/refresh")
    ? json(session("fresh")) : expired());
  await assert.rejects(client.fetch("/document"), (error) => error instanceof AuthError && error.status === 401);
  assert.equal(calls.filter(({ url }) => url === "/document").length, 2);
  assert.equal(client.snapshot.identity, null);
});

test("403 does not refresh or discard identity", async () => {
  const { client, calls } = await signedClient(async () => json({ detail: "无权访问" }, 403));
  assert.equal((await client.fetch("/document")).status, 403);
  assert.ok(client.snapshot.identity);
  assert.equal(calls.filter(({ url }) => url.endsWith("/refresh")).length, 0);
});

test("revoked 401 invalidates identity without refreshing", async () => {
  const { client, calls } = await signedClient(async () => json({ detail: { code: "SESSION_REVOKED" } }, 401));
  await assert.rejects(client.fetch("/document"), (error) => error.code === "SESSION_REVOKED");
  assert.equal(client.snapshot.identity, null);
  assert.equal(calls.filter(({ url }) => url.endsWith("/refresh")).length, 0);
});

test("refresh dependency 503 retains identity and the next request can retry refresh", async () => {
  let refreshCount = 0;
  const { client } = await signedClient(async (url, init) => {
    if (url.endsWith("/refresh")) {
      refreshCount++;
      return refreshCount === 1 ? json({ detail: { message: "Redis 暂不可用" } }, 503) : json(session("fresh"));
    }
    return init.headers.get("Authorization") === "Bearer initial" ? expired() : json({ ok: true });
  });
  await assert.rejects(client.fetch("/document"), (error) => error.status === 503);
  assert.ok(client.snapshot.identity);
  assert.equal((await client.fetch("/document")).status, 200);
  assert.equal(refreshCount, 2);
});

test("invalid refresh cookie clears identity and cannot continue a protected request", async () => {
  const { client } = await signedClient(async (url) => url.endsWith("/refresh")
    ? json({ detail: { code: "REFRESH_TOKEN_INVALID" } }, 401) : expired());
  await assert.rejects(client.fetch("/document"), (error) => error.code === "REFRESH_TOKEN_INVALID");
  assert.equal(client.snapshot.identity, null);
});

test("failed logout retains the session; successful retry clears it", async () => {
  let attempts = 0;
  const { client, calls } = await signedClient(async () => ++attempts === 1
    ? json({ detail: { message: "Redis 暂不可用" } }, 503) : json({ ok: true }));
  await assert.rejects(client.logoutAll(), (error) => error.status === 503);
  assert.ok(client.snapshot.identity);
  await client.logoutAll();
  assert.equal(client.snapshot.identity, null);
  assert.equal(calls.at(-1).init.headers.get("Authorization"), "Bearer initial");
  assert.equal(calls.at(-1).init.credentials, "include");
});

test("XHR-style upload and fetch share the same refresh; upload retries only once", async () => {
  const refreshStarted = deferred();
  const release = deferred();
  const { client, calls } = await signedClient(async (url, init) => {
    if (url.endsWith("/refresh")) {
      refreshStarted.resolve();
      await release.promise;
      return json(session("fresh"));
    }
    return init.headers.get("Authorization") === "Bearer initial" ? expired() : json({ ok: true });
  });
  const tokens = [];
  const upload = client.authorizedRequest(async (token) => {
    tokens.push(token);
    return token === "initial" ? { status: 401, body: { detail: { code: "ACCESS_TOKEN_EXPIRED" } } }
      : { status: 202, body: { status: "STORED" } };
  }, async (response) => response.body);
  const fetch = client.fetch("/document");
  await refreshStarted.promise;
  await setImmediate();
  release.resolve();
  assert.equal((await upload).status, 202);
  assert.equal((await fetch).status, 200);
  assert.deepEqual(tokens, ["initial", "fresh"]);
  assert.equal(calls.filter(({ url }) => url.endsWith("/refresh")).length, 1);
});

test("cancel during refresh prevents the original upload from retrying", async () => {
  const refreshStarted = deferred();
  const release = deferred();
  const { client } = await signedClient(async () => {
    refreshStarted.resolve();
    await release.promise;
    return json(session("fresh"));
  });
  const controller = new AbortController();
  let attempts = 0;
  const upload = client.authorizedRequest(async () => {
    attempts++;
    return { status: 401, body: { detail: { code: "ACCESS_TOKEN_EXPIRED" } } };
  }, async (response) => response.body, controller.signal);
  await refreshStarted.promise;
  controller.abort();
  release.resolve();
  await assert.rejects(upload, (error) => error.name === "AbortError");
  assert.equal(attempts, 1);
});

test("an identity change during refresh prevents old tenant requests from retrying", async () => {
  const { client, calls } = await signedClient(async (url) => url.endsWith("/refresh")
    ? json(session("fresh", { ...tenant, id: "20", name: "另一租户" })) : expired());
  const key = client.snapshot.sessionKey;
  await assert.rejects(client.fetch("/old-tenant-document"), (error) => error.name === "AbortError");
  assert.equal(client.snapshot.identity.tenant.id, "20");
  assert.notEqual(client.snapshot.sessionKey, key);
  assert.equal(calls.filter(({ url }) => url === "/old-tenant-document").length, 1);
});

test("session invalidation cancels pending requests and late refresh cannot revive it", async () => {
  const refreshStarted = deferred();
  const release = deferred();
  const { client } = await signedClient(async (url) => {
    if (url.endsWith("/refresh")) {
      refreshStarted.resolve();
      await release.promise;
      return json(session("fresh"));
    }
    return expired();
  });
  const request = client.fetch("/document");
  await refreshStarted.promise;
  client.clearSession("已退出");
  release.resolve();
  await assert.rejects(request, (error) => error.name === "AbortError");
  assert.equal(client.snapshot.identity, null);
});

test("cookie mutation lock serializes two tabs refreshing and logging out", async () => {
  let tail = Promise.resolve();
  const cookieLock = (operation) => {
    const result = tail.then(operation);
    tail = result.catch(() => {});
    return result;
  };
  let cookie = "initial";
  let count = 0;
  const seen = [];
  const fetcher = async (url) => {
    seen.push({ url, cookie });
    await setImmediate();
    if (url.endsWith("/logout")) { cookie = null; return json({ ok: true }); }
    if (!cookie) return json({ detail: { code: "REFRESH_TOKEN_INVALID" } }, 401);
    cookie = `rotated-${++count}`;
    return json(session(cookie));
  };
  const first = new AuthClient({ fetcher, cookieLock });
  const second = new AuthClient({ fetcher, cookieLock });
  await Promise.all([first.refresh(), second.refresh()]);
  assert.deepEqual(seen.map((item) => item.cookie), ["initial", "rotated-1"]);
  await Promise.all([first.refresh(), second.logoutAll()]);
  assert.equal(cookie, null);
  await assert.rejects(first.refresh(), (error) => error.status === 401);
  assert.equal(first.snapshot.identity, null);
});

test("ACL invalidation cancels late resource responses while keeping the current identity usable", async () => {
  const started = deferred();
  const release = deferred();
  const { client } = await signedClient(async (url) => {
    if (url === "/old-document") { started.resolve(); await release.promise; }
    return json({ ok: true });
  });
  const key = client.snapshot.sessionKey;
  const pending = client.fetch("/old-document");
  await started.promise;
  client.cancelRequests();
  release.resolve();
  await assert.rejects(pending, (error) => error.name === "AbortError");
  assert.ok(client.snapshot.identity);
  assert.equal(client.snapshot.sessionKey, key);
  assert.equal((await client.fetch("/current-document")).status, 200);
});
