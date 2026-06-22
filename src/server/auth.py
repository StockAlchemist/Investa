from datetime import datetime, timedelta, timezone
from typing import Optional
import bcrypt
import jwt
from pydantic import BaseModel, ConfigDict
import config

# --- Password Hashing ---

def _password_bytes(password: str) -> bytes:
    # bcrypt only reads the first 72 bytes; truncate explicitly so behaviour
    # matches the passlib setup that produced existing hashes (newer bcrypt
    # versions raise on longer inputs instead of truncating).
    return password.encode("utf-8")[:72]

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a stored bcrypt hash."""
    try:
        return bcrypt.checkpw(_password_bytes(plain_password), hashed_password.encode("utf-8"))
    except ValueError:
        # Malformed/non-bcrypt hash in the DB — treat as no match.
        return False

def get_password_hash(password: str) -> str:
    """Hashes a password using bcrypt (same $2b$, 12 rounds as the old passlib setup)."""
    return bcrypt.hashpw(_password_bytes(password), bcrypt.gensalt(rounds=12)).decode("utf-8")

# --- JWT Tokens ---

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[int] = None

class User(BaseModel):
    id: int
    username: str
    alias: Optional[str] = None
    is_active: bool = True
    created_at: str

    model_config = ConfigDict(from_attributes=True)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.AUTH_SECRET_KEY, algorithm=config.AUTH_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[TokenData]:
    """Decodes a JWT access token."""
    try:
        payload = jwt.decode(token, config.AUTH_SECRET_KEY, algorithms=[config.AUTH_ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("id")
        if username is None or user_id is None:
            return None
        return TokenData(username=username, user_id=user_id)
    except jwt.PyJWTError:
        # Covers invalid signature, expired token, and malformed input.
        return None
