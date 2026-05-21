export default function FeedbackPanel({ feedback, exercise }) {
  if (!feedback) {
    return (
      <div className="feedback-panel">
        <div className="feedback-loading">
          <div className="spinner" />
          <p>Starting session…</p>
        </div>
      </div>
    );
  }

  if (!feedback.ok) {
    return (
      <div className="feedback-panel">
        <div className="no-detect">
          <span className="no-detect-icon">👤</span>
          <p>No person detected — step into frame</p>
        </div>
      </div>
    );
  }

  const {
    form = "UNKNOWN",
    form_confidence = 0,
    posture_quality = "unknown",
    angles = {},
    danger_alerts = [],
    adaptive_feedback,
    exercise_detected,
  } = feedback;

  const isGood    = form === "GOOD";
  const isUnknown = form === "UNKNOWN";

  return (
    <div className="feedback-panel">
      <div className={`feedback-banner ${isUnknown ? "unknown" : isGood ? "correct" : "incorrect"}`}>
        {isUnknown ? (
          <p>🔄 Analysing form… keep moving</p>
        ) : (
          <>
            <p>{isGood ? "✅ Good Form!" : "⚠️ Fix Your Form"}</p>
            <span className="confidence-badge">
              {(form_confidence * 100).toFixed(0)}% confidence
            </span>
          </>
        )}
      </div>

      {posture_quality !== "unknown" && (
        <div className={`quality-chip quality-${posture_quality.toLowerCase()}`}>
          Posture: {posture_quality}
        </div>
      )}

      {adaptive_feedback && (
        <div className="adaptive-banner">
          🧠 {adaptive_feedback}
        </div>
      )}

      {danger_alerts.length > 0 && (
        <div className="issues-section">
          <h4>Corrections needed:</h4>
          <ul className="issues-list">
            {danger_alerts.map((alert) => (
              <li key={alert.code} className="issue-item">
                ⚠️ {alert.message}
              </li>
            ))}
          </ul>
        </div>
      )}

      {Object.keys(angles).length > 0 && (
        <div className="angles-section">
          <h4>Joint Angles</h4>
          <div className="angles-grid">
            {Object.entries(angles)
              .filter(([k]) => k !== "avg_knee")
              .map(([name, value]) => (
                <AngleCard key={name} name={name} value={value} />
              ))}
            {angles.avg_knee !== undefined && (
              <AngleCard name="avg_knee" value={angles.avg_knee} highlight />
            )}
          </div>
        </div>
      )}

      {exercise_detected && exercise_detected !== "unknown" && (
        <div className="detected-chip">
          🎯 Detected: {exercise_detected}
        </div>
      )}

      <div className="tips-section">
        <h4>💡 Squat Tips</h4>
        <ul className="tips-list">
          <li>• Drive through your heels on the way up.</li>
          <li>• Keep your chest tall and back straight.</li>
          <li>• Knees track over your toes — don't let them cave in.</li>
          <li>• Aim for thighs parallel to the floor at the bottom.</li>
        </ul>
      </div>
    </div>
  );
}

const SQUAT_RANGES = {
  left_knee:  [80, 120],
  right_knee: [80, 120],
  left_hip:   [55, 100],
  right_hip:  [55, 100],
  avg_knee:   [80, 120],
};

function AngleCard({ name, value, highlight }) {
  const label      = name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  const [min, max] = SQUAT_RANGES[name] ?? [0, 180];
  const inRange    = value >= min && value <= max;
  return (
    <div className={`angle-card ${inRange ? "in-range" : "out-range"} ${highlight ? "highlight" : ""}`}>
      <span className="angle-value">{value}°</span>
      <span className="angle-label">{label}</span>
      <span className="angle-target">{min}–{max}°</span>
    </div>
  );
}
