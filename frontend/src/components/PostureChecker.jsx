import { useEffect, useRef, useState, useCallback } from "react";
import FeedbackPanel from "./FeedbackPanel";
import { drawSkeleton } from "../utils/skeletonDrawer";
import { startSession, sendFrame, endSession, getSessionSummary } from "../api/spotterApi";

const FRAME_INTERVAL_MS = 120; // ~8 fps  — matches "5-10fps" recommendation

export default function PostureChecker({ exercise, onBack, onEnd }) {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const offscreen   = useRef(document.createElement("canvas"));
  const intervalRef = useRef(null);
  const streamRef   = useRef(null);
  const sessionRef  = useRef(null);
  const sendingRef  = useRef(false);

  const [feedback,    setFeedback]    = useState(null);
  const [apiStatus,   setApiStatus]   = useState("connecting");
  const [repCount,    setRepCount]    = useState(0);
  const [sessionTime, setSessionTime] = useState(0);
  const [ending,      setEnding]      = useState(false);

  const formatTime = (s) =>
    `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;

  useEffect(() => {
    const t = setInterval(() => setSessionTime((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);

  const captureBlob = useCallback(() => {
    return new Promise((resolve) => {
      const video = videoRef.current;
      if (!video || video.readyState < 2) return resolve(null);
      const oc  = offscreen.current;
      oc.width  = 640;
      oc.height = 360;
      oc.getContext("2d").drawImage(video, 0, 0, 640, 360);
      oc.toBlob(resolve, "image/jpeg", 0.7);
    });
  }, []);

  const sendOneFrame = useCallback(async () => {
    if (sendingRef.current || !sessionRef.current) return;
    sendingRef.current = true;
    try {
      const blob = await captureBlob();
      if (!blob) return;
      const data = await sendFrame(blob, sessionRef.current);
      setApiStatus("ok");
      setFeedback(data);
      if (data.reps !== undefined) setRepCount(data.reps);

      const canvas = canvasRef.current;
      const video  = videoRef.current;
      if (canvas && video && data.landmarks?.length) {
        const ctx = canvas.getContext("2d");
        canvas.width  = video.videoWidth  || 640;
        canvas.height = video.videoHeight || 360;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const lms = data.landmarks.map((lm) => ({ ...lm, visibility: 1 }));
        drawSkeleton(ctx, lms, canvas.width, canvas.height, data.form === "GOOD");
      }
    } catch (err) {
      console.error("Frame send error:", err);
      setApiStatus("error");
    } finally {
      sendingRef.current = false;
    }
  }, [captureBlob]);

  useEffect(() => {
    let active = true;
    async function init() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, facingMode: "user" },
          audio: false,
        });
        if (!active) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (err) {
        console.error("Camera error:", err);
        setApiStatus("error");
        return;
      }
      try {
        const { session_id } = await startSession(exercise.id);
        sessionRef.current   = session_id;
        setApiStatus("ok");
      } catch (err) {
        console.error("Session start error:", err);
        setApiStatus("error");
        return;
      }
      intervalRef.current = setInterval(sendOneFrame, FRAME_INTERVAL_MS);
    }
    init();
    return () => {
      active = false;
      clearInterval(intervalRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [exercise, sendOneFrame]);

  const handleEnd = useCallback(async () => {
    setEnding(true);
    clearInterval(intervalRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    try {
      const sid = sessionRef.current;
      if (sid) {
        const summary = await getSessionSummary(sid);
        await endSession(sid);
        onEnd(summary);
      }
    } catch (err) {
      console.error("End session error:", err);
      onEnd(null);
    }
  }, [onEnd]);

  const statusMeta = {
    connecting: { color: "#f59e0b", label: "Connecting…" },
    ok:         { color: "#22c55e", label: "Live" },
    error:      { color: "#ef4444", label: "API Error" },
  }[apiStatus];

  const isCorrect = feedback?.form === "GOOD";

  return (
    <div className="checker-screen">
      <div className="top-bar">
        <button className="back-btn" onClick={onBack}>← Back</button>
        <h2>{exercise.icon} {exercise.name}</h2>
        <div className="status-row">
          <span className="status-dot" style={{ background: statusMeta.color }} />
          <span style={{ color: statusMeta.color, fontSize: 13 }}>{statusMeta.label}</span>
          <span className="session-timer">⏱ {formatTime(sessionTime)}</span>
          <button className="end-btn" onClick={handleEnd} disabled={ending || apiStatus === "connecting"}>
            {ending ? "Saving…" : "End & Summary"}
          </button>
        </div>
      </div>

      <div className="main-layout">
        <div className="video-wrapper">
          <video ref={videoRef} className="video-feed" muted playsInline />
          <canvas ref={canvasRef} className="skeleton-canvas" />

          <div className="rep-badge">
            <span className="rep-label">REPS</span>
            <span className="rep-number">{repCount}</span>
          </div>

          {feedback?.stage && (
            <div className="stage-pill">{feedback.stage.toUpperCase()}</div>
          )}

          {feedback?.ok && (
            <div className={`form-pill ${isCorrect ? "good" : "bad"}`}>
              {isCorrect ? "✅ Good Form" : "⚠️ Fix Form"}
            </div>
          )}

          {apiStatus === "error" && (
            <div className="error-overlay">
              <span>⚠️ Cannot reach Spotter backend</span>
              <small>Check VITE_API_URL in .env and ensure the server is running on port 8000</small>
            </div>
          )}
        </div>

        <FeedbackPanel feedback={feedback} exercise={exercise} />
      </div>
    </div>
  );
}
