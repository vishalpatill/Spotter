/**
 * spotterApi.js
 * All calls to the Spotter AI backend (friend's FastAPI server).
 *
 * Base URL is read from .env:  VITE_API_URL=http://<friend-laptop-ip>:8000
 * Falls back to localhost for local dev.
 */

const BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Health ───────────────────────────────────────────────────────────────────
export async function checkHealth() {
  const res = await fetch(`${BASE}/`);
  return res.json();
}

// ── Session lifecycle ────────────────────────────────────────────────────────

/** POST /session/start  →  { session_id, exercise, message } */
export async function startSession(exercise = "squat") {
  const res = await fetch(`${BASE}/session/start?exercise=${exercise}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(`startSession failed: ${res.status}`);
  return res.json();
}

/** DELETE /session/{id}  →  { message } */
export async function endSession(sessionId) {
  await fetch(`${BASE}/session/${sessionId}`, { method: "DELETE" });
}

/** GET /session/{id}/summary  →  summary object */
export async function getSessionSummary(sessionId) {
  const res = await fetch(`${BASE}/session/${sessionId}/summary`);
  if (!res.ok) throw new Error(`summary failed: ${res.status}`);
  return res.json();
}

// ── Frame analysis ───────────────────────────────────────────────────────────

/**
 * POST /session/frame  (multipart: file + session_id)
 *
 * @param {Blob}   jpegBlob   – JPEG blob from canvas.toBlob()
 * @param {string} sessionId  – from startSession()
 * @returns {Promise<FrameResult>}
 *
 * FrameResult shape:
 * {
 *   ok, session_id, frame_count,
 *   reps, stage,
 *   form,              // "GOOD" | "BAD" | "UNKNOWN"
 *   form_confidence,
 *   exercise_detected,
 *   posture_quality,
 *   angles: { left_knee, right_knee, left_hip, right_hip, avg_knee, … },
 *   danger_alerts: [{ code, message }],
 *   adaptive_feedback,  // string | null
 *   landmarks,          // array of { x, y } (normalised)
 * }
 */
export async function sendFrame(jpegBlob, sessionId) {
  const form = new FormData();
  form.append("file", jpegBlob, "frame.jpg");
  form.append("session_id", sessionId);

  const res = await fetch(`${BASE}/session/frame`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) throw new Error(`sendFrame failed: ${res.status}`);
  return res.json();
}
