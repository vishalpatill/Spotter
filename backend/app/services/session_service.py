import time
from typing import Dict

_SESSIONS: Dict[str, Dict] = {}
_CLEANUP_THRESHOLD = 60 * 60

def get_session(session_id: str) -> Dict:
    now = time.time()
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = {
            "created_at": now,
            "last_seen": now,
            "reps": 0,
            "state": "up",
            "bad_count": 0,
            "last_exercise": None,
            "history": []
        }
    else:
        _SESSIONS[session_id]["last_seen"] = now
    return _SESSIONS[session_id]

def remove_old_sessions():
    now = time.time()
    to_remove = [k for k, v in _SESSIONS.items() if now - v.get("last_seen", 0) > _CLEANUP_THRESHOLD]
    for k in to_remove:
        del _SESSIONS[k]
