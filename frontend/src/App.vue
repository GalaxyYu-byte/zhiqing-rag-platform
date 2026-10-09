<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { ArrowLeft, ArrowRight, Building2, Check, Eye, EyeOff, Layers3, LoaderCircle, LockKeyhole, Mail, ShieldCheck } from "lucide-vue-next";
import DocumentWorkspace from "./DocumentWorkspace.vue";
import { authClient, authState, logoutAllDevices } from "./services/auth";
import { AuthError, type TenantSelection } from "./services/auth-client";
import { isDemoMode } from "./services/upload";

const restoring = ref(!isDemoMode);
const restoreError = ref("");
const email = ref("");
const password = ref("");
const passwordVisible = ref(false);
const submitting = ref(false);
const formError = ref("");
const selection = ref<TenantSelection | null>(null);
const tenantId = ref("");
const loggingOut = ref(false);
const logoutError = ref("");
const identity = computed(() => authState.value.identity);
const tenantRoles: Record<string, string> = {
  OWNER: "所有者", ADMIN: "管理员", TENANT_ADMIN: "管理员", MEMBER: "成员", USER: "成员", VIEWER: "只读成员",
};

function message(error: unknown, fallback: string) {
  if (error instanceof Error && error.name === "TimeoutError") return "服务响应超时，请重试";
  if (error instanceof TypeError) return "无法连接服务，请检查网络后重试";
  return error instanceof Error ? error.message : fallback;
}

async function restoreSession() {
  restoring.value = true;
  restoreError.value = "";
  try {
    await authClient.refresh();
  } catch (error) {
    if (error instanceof AuthError && error.status === 401) {
      // A first visit without a refresh cookie is a normal signed-out state.
      authClient.clearSession();
    } else restoreError.value = message(error, "登录服务暂不可用，请重试");
  } finally {
    restoring.value = false;
  }
}

function returnToLogin() {
  selection.value = null;
  tenantId.value = "";
  formError.value = "";
  password.value = "";
}

async function login() {
  if (submitting.value) return;
  submitting.value = true;
  formError.value = "";
  try {
    selection.value = await authClient.login(email.value.trim(), password.value);
    password.value = "";
    passwordVisible.value = false;
    tenantId.value = selection.value.tenants.length === 1 ? selection.value.tenants[0].id : "";
  } catch (error) {
    formError.value = message(error, "登录失败，请重试");
  } finally {
    submitting.value = false;
  }
}

async function enterTenant() {
  if (submitting.value || !selection.value || !tenantId.value) return;
  submitting.value = true;
  formError.value = "";
  try {
    await authClient.selectTenant(selection.value.selection_token, tenantId.value);
    selection.value = null;
    tenantId.value = "";
    logoutError.value = "";
  } catch (error) {
    if (error instanceof AuthError && error.status === 401) {
      returnToLogin();
      formError.value = "登录确认已过期或失效，请重新输入邮箱和密码";
    } else formError.value = message(error, "工作空间加载失败，请重试");
  } finally {
    submitting.value = false;
  }
}

async function logout() {
  if (loggingOut.value) return;
  loggingOut.value = true;
  logoutError.value = "";
  try {
    await logoutAllDevices();
    returnToLogin();
  } catch (error) {
    logoutError.value = message(error, "退出失败，请重试");
  } finally {
    loggingOut.value = false;
  }
}

onMounted(() => { if (!isDemoMode) void restoreSession(); });
</script>

<template>
  <DocumentWorkspace v-if="isDemoMode || identity" :key="authState.sessionKey" :identity="identity" :logging-out="loggingOut" :logout-error="logoutError" @logout="logout" />
  <div v-else class="auth-page">
    <aside class="auth-intro" aria-label="知擎平台介绍">
      <a class="brand auth-brand" href="/" aria-label="知擎首页"><span class="brand-icon"><Layers3 :size="24" /></span><span>知擎<span class="brand-en">ZHIQING</span></span></a>
      <div class="auth-intro-content"><span class="auth-eyebrow">团队知识 · 触手可及</span><h1>让知识相连，<br />让工作更从容。</h1><p>将团队文档汇聚到工作空间，<br />安全地管理、检索和分享每一份知识。</p><div class="auth-intro-visual" aria-hidden="true"><span><Layers3 :size="42" /></span><i></i><i></i><i></i></div></div>
      <span class="auth-intro-footer"><ShieldCheck :size="15" />团队知识的安全工作空间</span>
    </aside>
    <main class="auth-main">
      <section class="auth-card" aria-labelledby="auth-title">
        <div v-if="restoring" class="auth-restoring" role="status"><LoaderCircle :size="30" class="spin" /><h2 id="auth-title">正在恢复登录</h2><p>请稍候，即将进入工作空间。</p></div>
        <div v-else-if="restoreError" class="auth-restoring"><ShieldCheck :size="32" /><h2 id="auth-title">暂时无法恢复登录</h2><p class="auth-error" role="alert">{{ restoreError }}</p><button class="button primary auth-submit" @click="restoreSession">重新连接 <ArrowRight :size="17" /></button></div>
        <template v-else-if="selection">
          <span class="auth-step">登录确认 · 选择租户</span><h2 id="auth-title">选择工作空间</h2><p class="auth-description">{{ selection.user.display_name || selection.user.email }}，请选择本次要进入的租户。</p>
          <form @submit.prevent="enterTenant">
            <fieldset class="auth-tenants" :disabled="submitting"><legend class="sr-only">可访问的租户</legend><label v-for="tenant in selection.tenants" :key="tenant.id" class="auth-tenant" :class="{ selected: tenantId === tenant.id }"><input v-model="tenantId" type="radio" name="tenant" :value="tenant.id" required /><span class="auth-tenant-icon"><Building2 :size="23" /></span><span class="auth-tenant-details"><strong>{{ tenant.name }}</strong><small>{{ tenantRoles[tenant.role.toUpperCase()] || '团队成员' }} · {{ tenant.code }}</small></span><Check v-if="tenantId === tenant.id" :size="19" /></label></fieldset>
            <p v-if="!selection.tenants.length" class="auth-error" role="alert">当前账号尚未加入可用租户，请联系管理员。</p>
            <p v-if="formError" class="auth-error" role="alert">{{ formError }}</p>
            <button class="button primary auth-submit" type="submit" :disabled="submitting || !tenantId"><LoaderCircle v-if="submitting" :size="17" class="spin" />{{ submitting ? '正在进入…' : '进入工作空间' }}<ArrowRight v-if="!submitting" :size="17" /></button>
          </form>
          <button class="auth-back" :disabled="submitting" @click="returnToLogin"><ArrowLeft :size="15" />使用其他账号</button>
        </template>
        <template v-else>
          <span class="auth-step">欢迎回来</span><h2 id="auth-title">登录知擎</h2><p class="auth-description">使用邮箱和密码，进入你的团队工作空间。</p>
          <p v-if="authState.notice && !formError" class="auth-notice" role="status">{{ authState.notice }}</p>
          <form @submit.prevent="login">
            <label class="auth-field" for="login-email">邮箱<span class="auth-input"><Mail :size="18" /><input id="login-email" v-model="email" type="email" name="email" autocomplete="username" placeholder="name@company.com" maxlength="255" required :disabled="submitting" /></span></label>
            <label class="auth-field" for="login-password">密码<span class="auth-input"><LockKeyhole :size="18" /><input id="login-password" v-model="password" :type="passwordVisible ? 'text' : 'password'" name="password" autocomplete="current-password" placeholder="请输入密码" required maxlength="1024" :disabled="submitting" /><button type="button" :aria-label="passwordVisible ? '隐藏密码' : '显示密码'" :aria-pressed="passwordVisible" @click="passwordVisible = !passwordVisible"><EyeOff v-if="passwordVisible" :size="18" /><Eye v-else :size="18" /></button></span></label>
            <p v-if="formError" class="auth-error" role="alert">{{ formError }}</p>
            <button class="button primary auth-submit" type="submit" :disabled="submitting"><LoaderCircle v-if="submitting" :size="17" class="spin" />{{ submitting ? '正在登录…' : '登录' }}<ArrowRight v-if="!submitting" :size="17" /></button>
          </form>
          <p class="auth-account-note">账号由管理员开通。如需帮助，请联系你的团队管理员。</p>
        </template>
      </section>
      <footer class="auth-footer">知擎 · 团队知识平台</footer>
    </main>
  </div>
</template>
