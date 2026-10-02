"""Request body size limits, enforced while the body streams in (not after it has been buffered)."""
from __future__ import annotations

from fastapi import HTTPException
from starlette.responses import JSONResponse

_BODY_METHODS = ("POST", "PUT", "PATCH")
UPLOAD_PATH = "/api/scenes"


class BodyLimitMiddleware:
    """413 for a request body above ``upload_limit`` (POST /api/scenes) or ``default_limit`` (everything else).

    An honest Content-Length is refused up front; otherwise the bytes are counted as the app reads them and an
    HTTPException(413) is raised from ``receive`` (FastAPI re-raises it as is, so the client gets a clean 413).
    """

    def __init__(self, app, upload_limit: int, default_limit: int):
        self.app = app
        self.upload_limit = upload_limit
        self.default_limit = default_limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in _BODY_METHODS:
            return await self.app(scope, receive, send)
        limit = self.upload_limit if scope["path"].rstrip("/") == UPLOAD_PATH else self.default_limit
        detail = f"request body too large (limit {limit / (1024 * 1024):g} MB)"
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > limit:
            return await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)
        received = 0

        async def counting_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(status_code=413, detail=detail)
            return message

        await self.app(scope, counting_receive, send)
