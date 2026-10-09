import { shallowRef } from "vue";
import { AuthClient, type AuthSnapshot } from "./auth-client";

async function lockCookies<T>(operation: () => Promise<T>): Promise<T> {
  if (typeof navigator !== "undefined" && navigator.locks)
    return await navigator.locks.request("zhiqing-auth-cookie", operation);
  return operation();
}

export const authClient = new AuthClient({
  endpoint: import.meta.env.VITE_AUTH_URL?.trim() || "/api/auth",
  cookieLock: lockCookies,
});

export const authState = shallowRef<AuthSnapshot>(authClient.snapshot);
authClient.subscribe((snapshot) => { authState.value = snapshot; });

// Only sign-out is broadcast. Access tokens never enter shared browser storage.
const authChannel = typeof BroadcastChannel !== "undefined"
  ? new BroadcastChannel("zhiqing-auth") : null;
if (authChannel) authChannel.onmessage = (event) => {
  if (event.data === "signed-out") authClient.clearSession("账号已在其他页面退出，请重新登录");
};

export async function logoutAllDevices() {
  await authClient.logoutAll();
  authChannel?.postMessage("signed-out");
}

export const authenticatedFetch = (url: string, options?: RequestInit) => authClient.fetch(url, options);
