# tests/test_auth_security.py

import os
import stat
import sys

# --- Add src directory to sys.path for module import ---
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# --- End Path Addition ---

import config
from server.auth import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)
from server.rate_limit import SlidingWindowLimiter


# --- Password hashing (bcrypt, formerly passlib) ---

# Hash produced by the old passlib CryptContext(schemes=["bcrypt"]) setup for
# "correct horse battery staple" — existing users' stored hashes must keep working.
LEGACY_PASSLIB_HASH = "$2b$12$hhgdjHBW792znFfxCdyql.ns8Ba68CbION.9fOUAGUjIc88PJjTjq"


def test_hash_verify_roundtrip():
    hashed = get_password_hash("s3cret-pass")
    assert hashed.startswith("$2b$12$")
    assert verify_password("s3cret-pass", hashed)
    assert not verify_password("wrong-pass", hashed)


def test_legacy_passlib_hash_still_verifies():
    assert verify_password("correct horse battery staple", LEGACY_PASSLIB_HASH)
    assert not verify_password("wrong password", LEGACY_PASSLIB_HASH)


def test_long_password_truncated_like_passlib():
    # passlib truncated at 72 bytes; the bcrypt port must do the same so users
    # with very long passwords can still log in.
    long_pw = "x" * 100
    hashed = get_password_hash(long_pw)
    assert verify_password(long_pw, hashed)
    assert verify_password("x" * 72, hashed)  # same first 72 bytes


def test_malformed_stored_hash_is_no_match():
    assert not verify_password("anything", "not-a-bcrypt-hash")
    assert not verify_password("anything", "")


# --- JWT access tokens (PyJWT) ---


def test_jwt_roundtrip_returns_claims():
    token = create_access_token({"sub": "alice", "id": 7})
    assert isinstance(token, str)
    data = decode_access_token(token)
    assert data is not None
    assert data.username == "alice"
    assert data.user_id == 7


def test_jwt_expired_token_is_rejected():
    from datetime import timedelta

    token = create_access_token({"sub": "bob", "id": 1}, expires_delta=timedelta(seconds=-1))
    assert decode_access_token(token) is None


def test_jwt_tampered_signature_is_rejected():
    token = create_access_token({"sub": "carol", "id": 2})
    parts = token.split(".")
    parts[2] = ("A" if parts[2][0] != "A" else "B") + parts[2][1:]
    assert decode_access_token(".".join(parts)) is None


def test_jwt_wrong_key_is_rejected():
    # A token signed with a different secret must not validate against ours.
    foreign = jwt_encode_with_key({"sub": "dave", "id": 3}, "a-different-secret-key-" + "0" * 48)
    assert decode_access_token(foreign) is None


def test_jwt_missing_claims_returns_none():
    # Valid signature but no sub/id — decode_access_token should reject it.
    token = create_access_token({"foo": "bar"})
    assert decode_access_token(token) is None


def jwt_encode_with_key(claims, key):
    import jwt

    return jwt.encode(claims, key, algorithm=config.AUTH_ALGORITHM)


# --- Rate limiter ---

def test_limiter_allows_under_limit():
    limiter = SlidingWindowLimiter(max_events=3, window_seconds=60)
    for _ in range(2):
        limiter.record("k")
    assert limiter.retry_after("k") is None


def test_limiter_blocks_at_limit_and_reports_retry_after():
    limiter = SlidingWindowLimiter(max_events=3, window_seconds=60)
    for _ in range(3):
        limiter.record("k")
    retry = limiter.retry_after("k")
    assert retry is not None
    assert 0 < retry <= 60


def test_limiter_keys_are_independent():
    limiter = SlidingWindowLimiter(max_events=1, window_seconds=60)
    limiter.record("alice")
    assert limiter.retry_after("alice") is not None
    assert limiter.retry_after("bob") is None


def test_limiter_reset_clears_key():
    limiter = SlidingWindowLimiter(max_events=1, window_seconds=60)
    limiter.record("k")
    assert limiter.retry_after("k") is not None
    limiter.reset("k")
    assert limiter.retry_after("k") is None


def test_limiter_window_expiry(monkeypatch):
    import server.rate_limit as rl

    fake_now = [1000.0]
    monkeypatch.setattr(rl.time, "monotonic", lambda: fake_now[0])

    limiter = SlidingWindowLimiter(max_events=2, window_seconds=60)
    limiter.record("k")
    limiter.record("k")
    assert limiter.retry_after("k") is not None

    fake_now[0] += 61
    assert limiter.retry_after("k") is None


# --- Auth secret key resolution ---

def test_secret_key_env_var_wins(monkeypatch):
    monkeypatch.setenv("AUTH_SECRET_KEY", "from-env")
    assert config._resolve_auth_secret_key() == "from-env"


def test_secret_key_generated_and_persisted(monkeypatch, tmp_path):
    monkeypatch.delenv("AUTH_SECRET_KEY", raising=False)
    monkeypatch.setattr(config, "get_app_data_dir", lambda: str(tmp_path))

    key = config._resolve_auth_secret_key()
    assert len(key) == 64  # 32 random bytes, hex-encoded

    key_path = tmp_path / config.CONFIG_DIR / "auth_secret.key"
    assert key_path.read_text().strip() == key
    # Owner read/write only
    assert stat.S_IMODE(os.stat(key_path).st_mode) == 0o600

    # Second resolution loads the same persisted key
    assert config._resolve_auth_secret_key() == key


def test_no_hardcoded_secret_in_source():
    config_src = open(os.path.join(src_dir, "config.py"), encoding="utf-8").read()
    assert "09d25e094faa" not in config_src
