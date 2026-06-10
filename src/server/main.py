import sys
import os
import asyncio
import re

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

# Add project root to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


from server.api import router as api_router  # noqa: E402  (import follows sys.path setup above)

# Configure logging. Application logs stay quiet (warnings/errors only) in
# normal operation; set INVESTA_LOG_LEVEL=INFO or DEBUG when debugging.
logging.basicConfig(
    level=os.getenv("INVESTA_LOG_LEVEL", "WARNING").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from db_utils import initialize_database, initialize_global_database
    from server.refresh_worker import refresh_loop, index_refresh_loop
    from server.portfolio_service import warm_summary_caches
    initialize_database()
    initialize_global_database()

    # Kick off the periodic background workers.
    refresh_task = asyncio.create_task(refresh_loop(), name="metadata-refresh")
    # Keep the header index-quote cache warm so /summary never blocks on it.
    index_task = asyncio.create_task(index_refresh_loop(), name="index-refresh")
    # Pre-compute summaries for recently active users so the first dashboard
    # load after a restart is served from cache.
    warm_task = asyncio.create_task(warm_summary_caches(), name="summary-warmup")

    try:
        yield
    finally:
        # Graceful shutdown: cancel the background workers and drain the precalc pool.
        for task in (refresh_task, index_task, warm_task):
            task.cancel()
        for task in (refresh_task, index_task, warm_task):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        from server.portfolio_service import _PRECALC_POOL
        _PRECALC_POOL.shutdown(wait=False)

app = FastAPI(title="Investa API", description="Backend for Investa PWA", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: log the full stack trace, return a clean 500.

    Routes should not need their own `except Exception` catch-alls — anything
    unexpected lands here with full context in the server log.
    """
    logging.error(
        "Unhandled error on %s %s", request.method, request.url.path, exc_info=exc
    )
    # Responses from an Exception handler bypass the middleware stack, so CORS
    # headers must be added by hand or browsers report the 500 as a CORS error.
    headers = {}
    origin = request.headers.get("origin")
    if origin and (
        origin == "null" or origin in _extra_origins or re.match(_LOCAL_ORIGIN_REGEX, origin)
    ):
        headers["Access-Control-Allow-Origin"] = origin
    return JSONResponse(
        status_code=500, content={"detail": "Internal server error"}, headers=headers
    )

# CORS: restrict to the origins Investa is actually served from, instead of "*",
# so a random website can't make API requests with a stolen bearer token.
# Default coverage: localhost, private LAN ranges (RFC 1918), Tailscale (CGNAT
# IPs and *.ts.net hostnames), .local mDNS names — any port — plus the literal
# "null" origin the Electron desktop app sends when loading the UI via file://.
# Extra origins (e.g. a public domain) go in the CORS_ALLOW_ORIGINS env var,
# comma-separated.
_LOCAL_ORIGIN_REGEX = (
    r"^https?://("
    r"localhost|127\.0\.0\.1|\[::1\]"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r"|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}"
    r"|[\w.-]+\.ts\.net"
    r"|[\w-]+\.local"
    r")(:\d+)?$"
)
_extra_origins = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]
# Compress large JSON payloads (summary/history/screener) — big win on LAN/Tailscale.
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", *_extra_origins],
    allow_origin_regex=_LOCAL_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



# API Routes

app.include_router(api_router, prefix="/api")

# Mount at root as well to handle Tailscale Serve stripping the /api prefix
app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Investa API is running"}

if __name__ == "__main__":
    import uvicorn
    
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - " + log_config["formatters"]["access"]["fmt"]
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - " + log_config["formatters"]["default"]["fmt"]

    # reload=False for debugging stability vs potential thread deadlock issues with StatReload
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server.main:app", host="0.0.0.0", port=port, reload=False, workers=1, log_config=log_config)
