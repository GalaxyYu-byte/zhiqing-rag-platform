"""登录接口契约及服务端确认的请求身份。"""

from dataclasses import dataclass
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class AuthError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=1024, repr=False)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        value = value.strip().lower()
        if "@" not in value or any(character.isspace() for character in value):
            raise ValueError("请输入有效邮箱")
        return value


class SelectTenantRequest(BaseModel):
    selection_token: str = Field(min_length=20, max_length=256, repr=False)
    tenant_id: int = Field(gt=0)


class AuthUser(BaseModel):
    id: str
    email: str
    display_name: str


class AuthTenant(BaseModel):
    id: str
    code: str
    name: str
    role: str


class LoginResponse(BaseModel):
    selection_token: str
    user: AuthUser
    tenants: list[AuthTenant]


class CurrentIdentityResponse(BaseModel):
    user: AuthUser
    tenant: AuthTenant


class TokenResponse(CurrentIdentityResponse):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 900


@dataclass(frozen=True)
class AuthIdentity:
    user_id: int
    tenant_id: int
    member_id: int
    department_id: int | None
    role: str
    clearance: int
    auth_version: int
    tenant_auth_version: int
    session_key: UUID
    user: AuthUser
    tenant: AuthTenant
