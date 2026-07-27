"""Usernames are path segments, so they must be constrained.

`user_data_dir_for` joins the username onto the users root, and account
deletion then calls shutil.rmtree on the result. os.path.join does not
sanitise: "../../x" walks out of the root and an absolute username discards
the base entirely. Registration creates the directory and a SQLite file there;
deletion recursively removes it.

Two layers are tested: the Pydantic constraint that rejects such names at the
edge, and the containment check that still refuses to act on a stored row
predating that constraint.
"""

import os
import sys

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config
from server.routes.auth import UserCreate, _safe_user_data_dir


# --- Edge validation -----------------------------------------------------


@pytest.mark.parametrize("name", ["alice", "bob_2", "a.b-c", "Kit99", "abc", "a" * 32])
def test_ordinary_usernames_are_accepted(name):
    assert UserCreate(username=name, password="pw").username == name


@pytest.mark.parametrize(
    "name",
    [
        "../../../etc/passwd",   # traversal
        "..",                    # traversal, bare
        "a/../../b",             # traversal, embedded
        "/absolute/path",        # discards the join base
        "alice/bob",             # separator
        "alice\\bob",            # separator, Windows-style
        "ab",                    # too short
        "a" * 33,                # too long
        "",                      # empty
        ".hidden",               # leading dot -> dotfile dir
        "alice.",                # trailing dot
        "-alice",                # leading hyphen
        "alice bob",             # whitespace
        "alice\x00",             # NUL truncation
        # Python's re treats `$` as matching before a trailing newline, so a
        # hand-rolled re.match validator would accept these. Pydantic's engine
        # does not — pin that down.
        "alice\n",
        "alice\n../../escape",
    ],
)
def test_path_unsafe_usernames_are_rejected(name):
    with pytest.raises(ValidationError):
        UserCreate(username=name, password="pw")


# --- Containment check ---------------------------------------------------


@pytest.fixture
def app_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "get_app_data_dir", lambda: str(tmp_path))
    (tmp_path / config.USERS_DIR).mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_contained_username_resolves_inside_users_root(app_data):
    resolved = _safe_user_data_dir("alice")

    users_root = os.path.realpath(os.path.join(str(app_data), config.USERS_DIR))
    assert resolved == os.path.join(users_root, "alice")


@pytest.mark.parametrize("name", ["../escape", "../../escape", "/tmp/escape", "a/../../escape"])
def test_traversing_username_is_refused(app_data, name):
    """Defence in depth: a row written before USERNAME_PATTERN existed must not
    reach makedirs or rmtree."""
    with pytest.raises(HTTPException) as exc:
        _safe_user_data_dir(name)

    assert exc.value.status_code == 400


def test_username_resolving_to_the_users_root_itself_is_refused(app_data):
    """"." resolves to the root — rmtree there would wipe every user."""
    with pytest.raises(HTTPException) as exc:
        _safe_user_data_dir(".")

    assert exc.value.status_code == 400
