"""在 multipart 解析之前限制上传请求体，兼容无 Content-Length 的流式请求。"""

from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyTooLarge(HTTPException):
    def __init__(self):
        super().__init__(413, detail={"code": "FILE_TOO_LARGE", "message": "上传请求超过大小限制"})


class UploadBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"].rstrip("/") != "/api/documents/upload"
        ):
            return await self.app(scope, receive, send)
        response = JSONResponse(
            {"detail": {"code": "FILE_TOO_LARGE", "message": "上传请求超过大小限制"}},
            status_code=413,
        )
        headers = dict(scope.get("headers", []))
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            content_length = 0
        if content_length > self.max_bytes:
            return await response(scope, receive, send)
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await response(scope, receive, send)
