"""Auth routes: registration, login, profile, password management."""

import logging
import os
import shutil
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

import config
from server.auth import (
    Token, User, create_access_token, get_password_hash, verify_password
)
from server.dependencies import get_current_user, get_global_db_connection
from server.rate_limit import (
    enforce_limit, get_client_ip,
    failed_auth_limiter, login_ip_limiter, register_ip_limiter,
)

router = APIRouter()


class UserCreate(BaseModel):
    username: str
    password: str

class UserPasswordUpdate(BaseModel):
    current_password: str
    new_password: str


@router.post("/auth/register", response_model=User)
def register(user: UserCreate, request: Request, conn: sqlite3.Connection = Depends(get_global_db_connection)):
    # conn obtained from dependency is for GLOBAL DB (Users)
    client_ip = get_client_ip(request)
    enforce_limit(register_ip_limiter, client_ip, "registration attempts")
    register_ip_limiter.record(client_ip)

    try:
        cursor = conn.cursor()

        # Check if username exists
        cursor.execute("SELECT id FROM users WHERE username = ?", (user.username,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Username already registered")

        # Hash password
        hashed_pw = get_password_hash(user.password)
        created_at = datetime.now().isoformat()

        cursor.execute(
            "INSERT INTO users (username, hashed_password, created_at) VALUES (?, ?, ?)",
            (user.username, hashed_pw, created_at)
        )
        new_user_id = cursor.lastrowid
        if new_user_id is None:
            raise HTTPException(status_code=500, detail="Failed to create user record")
        conn.commit()

        # --- Initialize User Isolation ---
        # Create user directory and initialize their portfolio DB
        user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, user.username)
        try:
             os.makedirs(user_data_dir, exist_ok=True)

             # Initialize Portfolio DB
             user_db_path = os.path.join(user_data_dir, config.PORTFOLIO_DB_FILENAME)
             from db_utils import initialize_database

             # We initialize the DB (creates tables)
             user_conn = initialize_database(user_db_path)
             if user_conn:
                 user_conn.close()

             logging.info(f"Initialized isolated environment for user {user.username}")

        except Exception as e:
             # Rollback user creation if environment setup fails?
             # Ideally yes, but global DB commit already happened.
             # We log critical error.
             logging.error(f"Failed to initialize user environment for {user.username}: {e}")
             # Proceed? Or fail? The user exists but has no DB.
             # Let's try to fail harder or just logging.
             # Re-raising might be better so user knows it failed.
             raise HTTPException(status_code=500, detail="Failed to initialize user data environment")

        return User(id=new_user_id, username=user.username, is_active=True, created_at=created_at)

    except HTTPException:
        raise
    except Exception as e:
        # conn via dependency is closed by dependency, but we can try rollback if active transaction
        # But usually exception triggers 500.
        logging.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/auth/login", response_model=Token)
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), conn: sqlite3.Connection = Depends(get_global_db_connection)):
    # conn is GLOBAL DB
    client_ip = get_client_ip(request)
    user_key = f"login:{form_data.username.lower()}"
    enforce_limit(failed_auth_limiter, user_key, "failed login attempts")
    enforce_limit(login_ip_limiter, client_ip, "login attempts")
    login_ip_limiter.record(client_ip)

    cursor = conn.cursor()
    cursor.execute("SELECT id, username, hashed_password FROM users WHERE username = ?", (form_data.username,))
    row = cursor.fetchone()

    if not row or not verify_password(form_data.password, row[2]):
        failed_auth_limiter.record(user_key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    failed_auth_limiter.reset(user_key)

    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": row[1], "id": row[0]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/auth/me", response_model=User)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

class UpdateUserProfile(BaseModel):
    alias: Optional[str] = None

@router.patch("/auth/me", response_model=User)
def update_user_profile(
    profile_data: UpdateUserProfile,
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_global_db_connection)
):
    try:
        cursor = conn.cursor()

        # Only update alias for now
        if profile_data.alias is not None:
             logging.info(f"Updating alias for {current_user.username} to: '{profile_data.alias}'")
             cursor.execute("UPDATE users SET alias = ? WHERE id = ?", (profile_data.alias, current_user.id))
             conn.commit()
             current_user.alias = profile_data.alias

        return current_user
    except Exception as e:
        logging.error(f"Error updating profile for {current_user.username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

@router.delete("/auth/me")
def delete_user_me(
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_global_db_connection)
):
    try:
        # 1. Delete user from GLOBAL DB
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (current_user.id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="User not found")
        conn.commit()

        # 2. Delete user data directory
        user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, current_user.username)
        if os.path.exists(user_data_dir):
            try:
                shutil.rmtree(user_data_dir)
                logging.info(f"Deleted data directory for user {current_user.username}")
            except Exception as e:
                logging.error(f"Failed to delete data directory for {current_user.username}: {e}")
                # We continue as the user is effectively deleted from the system

        return {"status": "success", "message": f"User {current_user.username} deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting user {current_user.username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete user")


@router.post("/auth/change-password")
def change_password(
    password_data: UserPasswordUpdate,
    current_user: User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_global_db_connection)
):
    pw_key = f"change-password:{current_user.username.lower()}"
    enforce_limit(failed_auth_limiter, pw_key, "incorrect password attempts")
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT hashed_password FROM users WHERE id = ?", (current_user.id,))
        row = cursor.fetchone()

        if not row:
             raise HTTPException(status_code=404, detail="User not found")

        stored_hash = row[0]

        if not verify_password(password_data.current_password, stored_hash):
            failed_auth_limiter.record(pw_key)
            raise HTTPException(status_code=400, detail="Incorrect current password")
        failed_auth_limiter.reset(pw_key)

        hashed_new_pw = get_password_hash(password_data.new_password)

        cursor.execute("UPDATE users SET hashed_password = ? WHERE id = ?", (hashed_new_pw, current_user.id))
        conn.commit()

        return {"status": "success", "message": "Password updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error changing password for {current_user.username}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update password")
