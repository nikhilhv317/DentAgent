"""
Auth layer — JWT + bcrypt.

Simple and appropriate for a single-clinic app:
  - bcrypt hashes passwords (never stored plain)
  - JWT tokens signed with a secret key (HS256, 8-hour expiry)
  - FastAPI dependency `require_auth` verifies the token on every protected route
  - Token is carried in an Authorization: Bearer header (set by the frontend)

One role for now (all staff = same access). Role column is in the schema for
when you want to split later — no code changes needed in the routes, just add
a role check to require_auth.

FIRST-RUN ADMIN: the `admin` account's password comes from the ADMIN_PASSWORD
env var, or is randomly generated and printed to the console once (see
db.seed_staff). There is no built-in default password.

JWT_SECRET: set it to a long random string. If it's unset, a random secret is
generated per process — safe, but everyone is logged out on every restart.
"""
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import bcrypt

import db

# Never fall back to a fixed, published secret — anyone who knows it can forge
# a token for any user. The old dev default is rejected explicitly.
_KNOWN_WEAK_SECRETS = {"", "dentagent-dev-secret-change-in-prod", "dentavis-dev-secret-change-in-prod",
                       "change-me-to-a-long-random-string"}
SECRET_KEY = os.getenv("JWT_SECRET", "")
if SECRET_KEY in _KNOWN_WEAK_SECRETS or len(SECRET_KEY) < 32:
    if SECRET_KEY:
        print("[auth] JWT_SECRET is a known default or shorter than 32 chars — ignoring it.",
              file=sys.stderr)
    print("[auth] Using a random per-process JWT secret; logins won't survive a restart. "
          "Set JWT_SECRET to a long random string to fix.", file=sys.stderr)
    SECRET_KEY = secrets.token_urlsafe(48)
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 8

bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except Exception:
        return False


ROLES = ("admin", "clinician", "front_desk")


def create_token(username: str, display_name: str, role: str = "clinician",
                 clinic_id: int = 1) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS)
    return jwt.encode({"sub": username, "name": display_name, "role": role, "clinic": clinic_id,
                       "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


def authenticate(username: str, password: str):
    """Return staff record if credentials are valid, else None."""
    user = db.get_staff_by_username(username)
    if user and verify_password(password, user["password_hash"]):
        return user
    return None


def require_auth(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    """FastAPI dependency — inject into any route to require a valid token."""
    if not creds:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            detail="Not authenticated",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = decode_token(creds.credentials)
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired token",
                            headers={"WWW-Authenticate": "Bearer"})
    # The account must still exist and be active, and its CURRENT role/clinic win
    # over what the token says — deactivating someone or changing their role takes
    # effect immediately, not when their 8-hour token expires.
    staff = db.get_staff_by_username(payload.get("sub", ""))
    if not staff:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Account disabled",
                            headers={"WWW-Authenticate": "Bearer"})
    payload["role"] = staff.get("role") or "clinician"
    payload["clinic"] = staff.get("clinic_id") or 1
    return payload   # {"sub", "name", "role", "clinic"}


def require_role(*roles: str):
    """Dependency factory: require_role("admin", "clinician"). Admin always passes."""
    def check(user: dict = Depends(require_auth)):
        if user.get("role") != "admin" and user.get("role") not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                detail=f"Needs role: {' or '.join(('admin',) + roles)}")
        return user
    return check


require_admin = require_role()                  # admin only
require_clinician = require_role("clinician")   # clinical decisions: approvals, sign-off, deleting charts
