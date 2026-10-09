<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { Building2, LoaderCircle, Plus, ShieldCheck, Trash2, UserRound, X } from "lucide-vue-next";
import {
  classificationLabels, fetchPermissions, PermissionError, revokePermission, setDocumentClassification,
  setPermission, type Permission, type PermissionGrant, type PermissionTarget, type PermissionView,
} from "../services/permissions";

const props = defineProps<{ target: PermissionTarget; clearance: number }>();
const emit = defineEmits<{ close: []; changed: []; inaccessible: [] }>();
const dialog = ref<HTMLDialogElement>();
const view = ref<PermissionView | null>(null);
const loading = ref(true);
const busy = ref(false);
const error = ref("");
const success = ref("");
const subjectKind = ref<"user" | "department">("user");
const subjectId = ref("");
const permission = ref<Permission>("READ");
const grantDrafts = ref<Record<string, Permission>>({});
const confidentiality = ref(1);
const controller = new AbortController();
const permissions = computed<Permission[]>(() => props.target.kind === "knowledge-base"
  ? ["READ", "WRITE", "ADMIN"] : ["READ", "WRITE"]);
const labels: Record<Permission, string> = { READ: "读取", WRITE: "读取与编辑", ADMIN: "管理员" };
const classifications = computed(() => [0, 1, 2].filter((level) => level <= props.clearance));
const departmentName = computed(() => view.value?.subjects.departments.find(
  (department) => department.id === view.value?.document?.department_id,
)?.name || "未归属部门");
watch(subjectKind, () => { subjectId.value = ""; });

function applyView(data: PermissionView) {
  view.value = data;
  grantDrafts.value = Object.fromEntries(data.grants.map((grant) => [grant.id, grant.permission]));
  confidentiality.value = data.document?.confidentiality ?? 1;
}

function showError(reason: unknown) {
  if (controller.signal.aborted) return;
  error.value = reason instanceof Error ? reason.message : "权限操作失败，请重试";
  if (reason instanceof PermissionError && [403, 404].includes(reason.status)) {
    view.value = null;
    emit("inaccessible");
  }
}

async function load() {
  loading.value = true;
  error.value = "";
  try { applyView(await fetchPermissions(props.target, controller.signal)); }
  catch (reason) { showError(reason); }
  finally { if (!controller.signal.aborted) loading.value = false; }
}

async function mutate(operation: () => Promise<PermissionView>, text: string) {
  if (busy.value) return;
  busy.value = true;
  error.value = "";
  success.value = "";
  try {
    const data = await operation();
    if (controller.signal.aborted) return;
    applyView(data);
    success.value = text;
    emit("changed");
  } catch (reason) { showError(reason); }
  finally { if (!controller.signal.aborted) busy.value = false; }
}

async function addGrant() {
  if (!subjectId.value) return;
  const subject = subjectKind.value === "user" ? { user_id: subjectId.value } : { department_id: subjectId.value };
  await mutate(() => setPermission(props.target, subject, permission.value, controller.signal), "授权已保存");
}

async function updateGrant(grant: PermissionGrant) {
  const subject = grant.user_id ? { user_id: grant.user_id } : { department_id: grant.department_id! };
  await mutate(() => setPermission(props.target, subject, grantDrafts.value[grant.id], controller.signal), "授权已更新");
}

function close() { if (!busy.value) emit("close"); }
onMounted(async () => { await nextTick(); if (!controller.signal.aborted) { dialog.value?.showModal(); await load(); } });
onBeforeUnmount(() => { controller.abort(); dialog.value?.close(); });
</script>

<template>
  <dialog ref="dialog" class="permission-dialog" aria-labelledby="permission-title" @cancel.prevent="close" @click="$event.target === dialog && close()">
    <div class="modal-content">
      <div class="modal-heading"><div><h2 id="permission-title">{{ target.kind === 'knowledge-base' ? '知识库授权' : '文档权限' }}</h2><p class="permission-target-name">{{ target.name }}</p></div><button class="icon-button" aria-label="关闭权限管理" :disabled="busy" @click="close"><X :size="20" /></button></div>
      <div v-if="loading" class="permission-loading" role="status"><LoaderCircle :size="24" class="spin" />正在加载授权…</div>
      <p v-if="error" class="permission-error" role="alert">{{ error }}</p>
      <div v-if="!loading && !view" class="permission-reload"><button class="button secondary" @click="load">重新加载</button></div>
      <template v-if="view && !loading">
        <section v-if="view.document" class="permission-classification">
          <div><strong><ShieldCheck :size="16" />文档密级</strong><small>所属部门：{{ departmentName }}</small></div>
          <form class="classification-form" @submit.prevent="mutate(() => setDocumentClassification(target.id, confidentiality, controller.signal), '文档密级已更新')"><select v-model.number="confidentiality" aria-label="文档密级" :disabled="busy"><option v-for="level in classifications" :key="level" :value="level">{{ classificationLabels[level] }}</option></select><button class="button secondary" :disabled="busy || confidentiality === view.document.confidentiality">保存密级</button></form>
          <p>{{ confidentiality === 0 ? '内部公开文档仍需知识库访问权限。' : view.document.department_id ? '默认仅所属部门成员可见，跨部门需在下方明确共享。' : '该文档未归属部门，非公开文档仅本人及明确共享对象可见。' }}</p>
        </section>
        <section class="permission-add"><h3>{{ target.kind === 'knowledge-base' ? '添加知识库授权' : '添加共享对象' }}</h3><form @submit.prevent="addGrant"><label>对象类型<select v-model="subjectKind" :disabled="busy"><option value="user">用户</option><option value="department">部门</option></select></label><label class="permission-subject">{{ subjectKind === 'user' ? '选择用户' : '选择部门' }}<select v-model="subjectId" required :disabled="busy"><option value="">请选择…</option><template v-if="subjectKind === 'user'"><option v-for="user in view.subjects.users" :key="user.id" :value="user.id">{{ user.display_name || user.email }}（{{ user.email }}）</option></template><template v-else><option v-for="department in view.subjects.departments" :key="department.id" :value="department.id">{{ department.name }}</option></template></select></label><label>权限<select v-model="permission" :disabled="busy"><option v-for="value in permissions" :key="value" :value="value">{{ labels[value] }}</option></select></label><button class="button primary" :disabled="busy || !subjectId"><LoaderCircle v-if="busy" :size="15" class="spin" /><Plus v-else :size="15" />添加</button></form></section>
        <p class="permission-scope-note"><ShieldCheck :size="14" />{{ target.kind === 'document' ? '共享对象还需知识库访问权限与对应密级。' : '知识库授权决定访问范围；部门内部和机密文档还需满足部门共享与密级要求。管理员可管理该知识库授权。' }}</p>
        <section class="permission-grants"><div class="permission-grants-heading"><h3>{{ target.kind === 'knowledge-base' ? '已有授权' : '明确共享' }}</h3><span>{{ view.grants.length }} 项</span></div><p v-if="!view.grants.length" class="permission-empty">{{ target.kind === 'knowledge-base' ? '暂无明确授权。' : '暂无明确共享，按文档默认可见范围访问。' }}</p><ul v-else><li v-for="grant in view.grants" :key="grant.id"><span class="permission-subject-icon"><UserRound v-if="grant.user_id" :size="17" /><Building2 v-else :size="17" /></span><div class="permission-grant-name"><strong>{{ grant.subject_name }}</strong><small>{{ grant.user_id ? '用户' : '部门' }}</small></div><select v-model="grantDrafts[grant.id]" :aria-label="`${grant.subject_name} 的权限`" :disabled="busy"><option v-for="value in permissions" :key="value" :value="value">{{ labels[value] }}</option></select><button class="text-button" :disabled="busy || grantDrafts[grant.id] === grant.permission" @click="updateGrant(grant)">保存</button><button class="icon-button delete-button" :disabled="busy" :aria-label="`撤销 ${grant.subject_name} 的授权`" @click="mutate(() => revokePermission(target, grant.id, controller.signal), '授权已撤销')"><Trash2 :size="16" /></button></li></ul></section>
        <p v-if="success" class="permission-success" role="status">{{ success }}</p>
        <p v-if="target.kind === 'knowledge-base'" class="permission-footnote">至少保留一位可用管理员。调整本人权限可能结束当前管理权限。</p>
      </template>
    </div>
  </dialog>
</template>
