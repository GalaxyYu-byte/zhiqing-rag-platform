import { authenticatedFetch } from "./auth";
import { responseCode, responseMessage } from "./auth-client";

export type Permission = "READ" | "WRITE" | "ADMIN";
export interface PermissionTarget { kind: "knowledge-base" | "document"; id: string; name: string }
export interface PermissionGrant {
  id: string;
  user_id: string | null;
  department_id: string | null;
  permission: Permission;
  subject_name: string;
}
export interface PermissionView {
  knowledge_base?: { id: string; name: string; acl_version: number };
  document?: { id: string; title: string; department_id: string | null; confidentiality: number };
  grants: PermissionGrant[];
  subjects: {
    users: { id: string; email: string; display_name: string | null; department_id: string | null; clearance: number }[];
    departments: { id: string; name: string }[];
  };
}
export class PermissionError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(message: string, status: number, code: string) {
    super(message);
    this.name = "PermissionError";
    this.status = status;
    this.code = code;
  }
}

function endpoint(target: PermissionTarget) {
  const base = target.kind === "knowledge-base"
    ? import.meta.env.VITE_KNOWLEDGE_BASES_URL?.trim() || "/api/knowledge-bases"
    : import.meta.env.VITE_DOCUMENTS_URL?.trim() || "/api/documents";
  return `${base.replace(/\/$/, "")}/${encodeURIComponent(target.id)}`;
}

async function request(url: string, signal: AbortSignal, method = "GET", body?: unknown): Promise<PermissionView> {
  const response = await authenticatedFetch(url, {
    method, signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)]),
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json().catch(() => null);
  signal.throwIfAborted();
  if (!response.ok) throw new PermissionError(
    responseMessage(data, `权限操作失败（${response.status}），请重试`), response.status, responseCode(data),
  );
  if (!data || !Array.isArray(data.grants) || !data.subjects)
    throw new Error("权限响应不完整，请重新加载");
  return data as PermissionView;
}

export const fetchPermissions = (target: PermissionTarget, signal: AbortSignal) =>
  request(`${endpoint(target)}/permissions`, signal);

export function setPermission(
  target: PermissionTarget,
  subject: { user_id: string; department_id?: never } | { department_id: string; user_id?: never },
  permission: Permission, signal: AbortSignal,
) {
  return request(`${endpoint(target)}/permissions`, signal, "POST", { ...subject, permission });
}

export const revokePermission = (target: PermissionTarget, grantId: string, signal: AbortSignal) =>
  request(`${endpoint(target)}/permissions/${encodeURIComponent(grantId)}`, signal, "DELETE");

export const setDocumentClassification = (documentId: string, confidentiality: number, signal: AbortSignal) =>
  request(`${endpoint({ kind: "document", id: documentId, name: "" })}/classification`, signal, "POST", { confidentiality });

export const classificationLabels: Record<number, string> = { 0: "内部公开", 1: "部门内部", 2: "机密" };
