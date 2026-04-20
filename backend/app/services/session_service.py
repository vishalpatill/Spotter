import time

SESSION_TTL_SECONDS = 60 * 30

_SESSIONS = {}


def _new_session():
    return {
        "state": "up",
        "reps": 0,
        "bad_count": 0,
        "last_exercise": None,
        "history": [],
        "last_seen": time.time(),
    }


def get_session(session_id):
    session = _SESSIONS.setdefault(session_id, _new_session())
    session["last_seen"] = time.time()
    return session


def remove_old_sessions():
    cutoff = time.time() - SESSION_TTL_SECONDS
    expired_ids = [
        session_id
        for session_id, session in _SESSIONS.items()
        if session.get("last_seen", 0) < cutoff
    ]

    for session_id in expired_ids:
        del _SESSIONS[session_id]
