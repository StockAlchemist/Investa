import sys
import os

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


from server.api import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from db_utils import initialize_database, initialize_global_database
    # Ensure DB is initialized
    initialize_database()
    initialize_global_database()
    yield

app = FastAPI(title="Investa API", description="Backend for Investa PWA", lifespan=lifespan)

# Configure CORS to allow requests from the frontend (likely localhost:3000)
# Configure CORS to allow requests from the frontend
origins = ["*"] # Allow all for local networking convenience

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
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
    uvicorn.run("server.main:app", host="0.0.0.0", port=port, reload=True, workers=1, log_config=log_config)
