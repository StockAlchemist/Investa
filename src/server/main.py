import sys
import os
import asyncio

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


from server.api import router as api_router  # noqa: E402  (import follows sys.path setup above)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from db_utils import initialize_database, initialize_global_database
    from server.refresh_worker import refresh_loop, index_refresh_loop
    initialize_database()
    initialize_global_database()

    # Kick off the periodic background workers.
    refresh_task = asyncio.create_task(refresh_loop(), name="metadata-refresh")
    # Keep the header index-quote cache warm so /summary never blocks on it.
    index_task = asyncio.create_task(index_refresh_loop(), name="index-refresh")

    try:
        yield
    finally:
        # Graceful shutdown: cancel the background workers and drain the precalc pool.
        for task in (refresh_task, index_task):
            task.cancel()
        for task in (refresh_task, index_task):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        from server.api import _PRECALC_POOL
        _PRECALC_POOL.shutdown(wait=False)

app = FastAPI(title="Investa API", description="Backend for Investa PWA", lifespan=lifespan)

# Configure CORS to allow requests from the frontend (likely localhost:3000)
# Configure CORS to allow requests from the frontend
# Bearer-token auth — credentials (cookies) not needed, so wildcard origins are safe.
# allow_credentials=True + allow_origins=["*"] is rejected by browsers per the CORS spec.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
