export interface AuthUser {
  id: string;
  email: string;
  display_name: string | null;
}

export interface AuthTenant {
  id: string;
  code: string;
  name: string;
  role: string;
}

export interface TenantSelection {
  selection_token: string;
  user: AuthUser;
  tenants: AuthTenant[];
}

export interface SessionResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: AuthUser;
  tenant: AuthTenant;
}

export interface AuthSnapshot {
  identity: { user: AuthUser; tenant: AuthTenant } | null;
  // Only a new identity/session remounts the workspace. Normal refresh keeps it alive.
  sessionKey: number;
  notice: string;
}

export class AuthError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(message: string, status: number, code = "") {
    super(message);
    this.name = "AuthError";
    this.status = status;
    this.code = code;
  }
}

export function responseCode(body: unknown): string {
  if (!body || typeof body !== "object" || !("detail" in body)) return "";
  const detail = body.detail;
  return detail && typeof detail === "object" && "code" in detail &&
    typeof detail.code === "string" ? detail.code : "";
}

export function responseMessage(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object" || !("detail" in body)) return fallback;
  const detail = body.detail;
  if (typeof detail === "string") return detail;
  return detail && typeof detail === "object" && "message" in detail &&
    typeof detail.message === "string" ? detail.message : fallback;
}

interface StatusResponse { status: number }
type CookieLock = <T>(operation: () => Promise<T>) => Promise<T>;

/** All tokens stay in memory; the browser manages the HttpOnly refresh cookie. */
export class AuthClient {
  private accessToken: string | null = null;
  private identity: AuthSnapshot["identity"] = null;
  private sessionKey = 0;
  private notice = "";
  private epoch = 0;
  private sessionController = new AbortController();
  private refreshPromise: Promise<void> | null = null;
  private listeners = new Set<(snapshot: AuthSnapshot) => void>();
  private readonly endpoint: string;
  private readonly fetcher: typeof fetch;
  private readonly cookieLock: CookieLock;

  constructor(options: {
    endpoint?: string;
    fetcher?: typeof fetch;
    cookieLock?: CookieLock;
  } = {}) {
    this.endpoint = (options.endpoint || "/api/auth").replace(/\/$/, "");
    this.fetcher = options.fetcher || ((...args) => fetch(...args));
    this.cookieLock = options.cookieLock || ((operation) => operation());
  }

  get snapshot(): AuthSnapshot {
    return { identity: this.identity, sessionKey: this.sessionKey, notice: this.notice };
  }

  subscribe(listener: (snapshot: AuthSnapshot) => void): () => void {
    this.listeners.add(listener);
    listener(this.snapshot);
    return () => this.listeners.delete(listener);
  }

  private emit() {
    for (const listener of this.listeners) listener(this.snapshot);
  }

  /** Invalidate in-flight resource reads after ACL changes, retaining the login. */
  cancelRequests() {
    this.sessionController.abort();
    this.sessionController = new AbortController();
  }

  clearSession(notice = "") {
    this.epoch++;
    this.cancelRequests();
    this.accessToken = null;
    this.identity = null;
    this.sessionKey++;
    this.notice = notice;
    this.emit();
  }

  private setSession(session: SessionResponse, newSession = false) {
    if (!session.access_token || !session.user?.id || !session.tenant?.id)
      throw new AuthError("登录响应不完整，请重试", 502);
    const changed = newSession || this.identity?.user.id !== session.user.id ||
      this.identity?.tenant.id !== session.tenant.id;
    if (changed) {
      this.sessionController.abort();
      this.sessionController = new AbortController();
      this.sessionKey++;
    }
    this.accessToken = session.access_token;
    this.identity = { user: session.user, tenant: session.tenant };
    this.notice = "";
    this.emit();
  }

  private async authRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
    const headers = new Headers(options.headers);
    headers.set("X-Requested-With", "XMLHttpRequest");
    if (options.body) headers.set("Content-Type", "application/json");
    const response = await this.fetcher(`${this.endpoint}${path}`, {
      ...options, method: options.method || "POST", credentials: "include", headers,
      signal: options.signal || AbortSignal.timeout(15_000),
    });
    const body = await response.json().catch(() => null);
    if (!response.ok)
      throw new AuthError(
        responseMessage(body, response.status === 503
          ? "登录服务暂不可用，请稍后重试" : "登录请求失败，请重试"),
        response.status, responseCode(body),
      );
    return body as T;
  }

  login(email: string, password: string): Promise<TenantSelection> {
    return this.authRequest("/login", { body: JSON.stringify({ email, password }) });
  }

  async selectTenant(selectionToken: string, tenantId: string): Promise<void> {
    const epoch = this.epoch;
    const session = await this.cookieLock(() => this.authRequest<SessionResponse>(
      "/select-tenant", { body: JSON.stringify({ selection_token: selectionToken, tenant_id: tenantId }) },
    ));
    if (epoch !== this.epoch) throw new DOMException("登录状态已变更", "AbortError");
    this.epoch++;
    this.setSession(session, true);
  }

  refresh(): Promise<void> {
    if (this.refreshPromise) return this.refreshPromise;
    const epoch = this.epoch;
    const operation = this.cookieLock(async () => {
      if (epoch !== this.epoch) throw new DOMException("登录状态已变更", "AbortError");
      try {
        const session = await this.authRequest<SessionResponse>("/refresh");
        if (epoch !== this.epoch) throw new DOMException("登录状态已变更", "AbortError");
        this.setSession(session);
      } catch (error) {
        if (epoch === this.epoch && error instanceof AuthError && error.status === 401)
          this.clearSession("登录已失效，请重新登录");
        throw error;
      }
    });
    const shared = operation.finally(() => {
      if (this.refreshPromise === shared) this.refreshPromise = null;
    });
    this.refreshPromise = shared;
    return shared;
  }

  async logoutAll(): Promise<void> {
    // Cookie mutations share a cross-tab lock, so an earlier refresh finishes first.
    await this.cookieLock(async () => {
      const headers = new Headers();
      if (this.accessToken) headers.set("Authorization", `Bearer ${this.accessToken}`);
      await this.authRequest("/logout", { headers });
      this.clearSession("已退出所有设备");
    });
  }

  /** Shared retry policy for fetch and XHR uploads, including late concurrent 401s. */
  async authorizedRequest<T extends StatusResponse>(
    send: (token: string, signal: AbortSignal) => Promise<T>,
    readError: (response: T) => Promise<unknown>,
    callerSignal?: AbortSignal | null,
  ): Promise<T> {
    const sessionSignal = this.sessionController.signal;
    const signal = callerSignal ? AbortSignal.any([callerSignal, sessionSignal]) : sessionSignal;
    signal.throwIfAborted();
    let token = this.accessToken;
    if (!token) throw new AuthError("请先登录", 401, "LOGIN_REQUIRED");
    for (let attempt = 0; attempt < 2; attempt++) {
      signal.throwIfAborted();
      const response = await send(token, signal);
      signal.throwIfAborted();
      if (response.status !== 401) return response;
      const body = await readError(response);
      signal.throwIfAborted();
      if (responseCode(body) !== "ACCESS_TOKEN_EXPIRED" || attempt === 1) {
        this.clearSession("登录已失效，请重新登录");
        throw new AuthError(responseMessage(body, "登录已失效，请重新登录"), 401, responseCode(body));
      }
      // Another request may already have refreshed while this response was in flight.
      if (this.accessToken === token) await this.refresh();
      signal.throwIfAborted();
      token = this.accessToken;
      if (!token) throw new AuthError("登录已失效，请重新登录", 401, "LOGIN_REQUIRED");
    }
    throw new AuthError("登录已失效，请重新登录", 401);
  }

  fetch(url: string, options: RequestInit = {}): Promise<Response> {
    return this.authorizedRequest(
      (token, signal) => {
        const headers = new Headers(options.headers);
        headers.set("Authorization", `Bearer ${token}`);
        if (!["GET", "HEAD"].includes((options.method || "GET").toUpperCase()))
          headers.set("X-Requested-With", "XMLHttpRequest");
        return this.fetcher(url, { ...options, headers, credentials: "include", signal });
      },
      (response) => response.clone().json().catch(() => null), options.signal,
    );
  }
}
