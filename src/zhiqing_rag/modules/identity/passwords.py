"""兼容开发账号的 scrypt 密码哈希；限制参数以避免损坏哈希耗尽资源。"""

import base64
import binascii
import hashlib
import hmac
import secrets


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return "$".join([
        "scrypt", "16384", "8", "1",
        base64.urlsafe_b64encode(salt).decode(), base64.urlsafe_b64encode(digest).decode(),
    ])


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt_text, digest_text = encoded.split("$")
        if scheme != "scrypt" or (n, r, p) != ("16384", "8", "1"):
            return False
        salt = base64.b64decode(salt_text, altchars=b"-_", validate=True)
        digest = base64.b64decode(digest_text, altchars=b"-_", validate=True)
        if len(salt) != 16 or len(digest) != 32:
            return False
        actual = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
        return hmac.compare_digest(actual, digest)
    except (ValueError, TypeError, binascii.Error):
        return False


# 未知账号也执行同样的密码校验，避免邮箱存在与否的明显时间差。
DUMMY_PASSWORD_HASH = hash_password(secrets.token_urlsafe(32))
