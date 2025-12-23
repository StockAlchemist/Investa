import sys
import os
print(f"DEBUG: Executing main.py from {__file__}")
print(f"DEBUG: CWD is {os.getcwd()}")
print(f"DEBUG: sys.path is {sys.path}")
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to sys.path to allow importing from src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from server.api import router as api_router
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="Investa API", description="Backend for Investa PWA")

# Configure CORS to allow requests from the frontend (likely localhost:3000)
# Configure CORS to allow requests from the frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173", # Vite default
    "http://127.0.0.1:5173",
    "http://100.66.59.98:3000", # User's Tailscale/Network IP
    "http://100.66.59.98:5173",
    "http://muon.tail33e9df.ts.net:3000", # Possible hostname
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
async def root():
    return {"message": "Investa API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
