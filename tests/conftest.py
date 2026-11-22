"""
Shared test setup: every test session runs against a throwaway database seeded
with the standard roster, so tests never touch dpms.db. The env vars must be set
BEFORE app/db are imported, which is why this module does it at import time.
"""
import os
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_TMP = tempfile.mkdtemp(prefix="dentagent-test-")
os.environ["DPMS_DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["ADMIN_PASSWORD"] = "admin-test-pw"
os.environ.setdefault("JWT_SECRET", "test-secret-" + "x" * 40)

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
import db  # noqa: E402


@pytest.fixture(scope="session")
def client():
    return TestClient(app_module.app)   # not a context manager: no scheduler threads


def _login(client, username, password):
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


@pytest.fixture(scope="session")
def admin(client):
    return _login(client, "admin", "admin-test-pw")


@pytest.fixture(scope="session")
def staff(client):
    """A front-desk login (the least-privileged role)."""
    try:
        db.create_staff("desk", "Front Desk", "desk-test-pw", role="front_desk")
    except ValueError:
        pass
    return _login(client, "desk", "desk-test-pw")


@pytest.fixture(scope="session")
def clinician(client):
    try:
        db.create_staff("drsmile", "Dr. Smile", "clin-test-pw", role="clinician")
    except ValueError:
        pass
    return _login(client, "drsmile", "clin-test-pw")


@pytest.fixture
def tmp_models(monkeypatch, tmp_path):
    """Point the retraining agent at a temp folder so tests never overwrite models/."""
    from ml import retrain
    monkeypatch.setattr(retrain, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(retrain, "ACTIVE_PATH", str(tmp_path / "noshow.pkl"))
    return tmp_path
