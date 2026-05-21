/**
 * SessionSummary
 * Shown after the user clicks "End & Summary".
 * Consumes the GET /session/{id}/summary response:
 * {
 *   session_id, exercise,
 *   total_reps, duration_seconds,
 *   avg_form_score,         // 0-1
 *   common_dangers,         // [code, ...]
 *   final_grade,            // "A" | "B" | "C" | "D"
 * }
 */

const DANGER_LABELS = {
  back_rounding:   "Back rounding — chest up!",
  not_deep_enough: "Not going deep enough",
  imbalance:       "Uneven leg balance",
};

const GRADE_META = {
  A: { emoji: "🏆", color: "#22c55e", label: "Excellent!" },
  B: { emoji: "💪", color: "#84cc16", label: "Great work!" },
  C: { emoji: "👍", color: "#f59e0b", label: "Keep it up!" },
  D: { emoji: "🔄", color: "#ef4444", label: "Needs work" },
};

export default function SessionSummary({ summary, exercise, onRestart }) {
  if (!summary) {
    return (
      <div className="summary-screen">
        <div className="summary-card">
          <h2>Session complete!</h2>
          <p style={{ color: "var(--muted)", marginTop: "0.5rem" }}>
            Could not load summary data. The session may have ended unexpectedly.
          </p>
          <button className="restart-btn" onClick={onRestart}>← Back to exercises</button>
        </div>
      </div>
    );
  }

  const {
    total_reps       = 0,
    duration_seconds = 0,
    avg_form_score   = 0,
    common_dangers   = [],
    final_grade      = "C",
  } = summary;

  const grade    = GRADE_META[final_grade] ?? GRADE_META["C"];
  const mins     = Math.floor(duration_seconds / 60);
  const secs     = Math.round(duration_seconds % 60);
  const scoreStr = `${(avg_form_score * 100).toFixed(0)}%`;

  return (
    <div className="summary-screen">
      <div className="summary-card">
        {/* Header */}
        <div className="summary-header">
          <span className="summary-grade-emoji">{grade.emoji}</span>
          <h1 style={{ color: grade.color }}>Grade {final_grade}</h1>
          <p className="summary-subtitle">{grade.label}</p>
        </div>

        {/* Stats row */}
        <div className="summary-stats">
          <StatBox label="Reps" value={total_reps} icon="🔁" />
          <StatBox label="Duration" value={`${mins}m ${secs}s`} icon="⏱" />
          <StatBox label="Avg Form" value={scoreStr} icon="📊" color={
            avg_form_score >= 0.85 ? "#22c55e"
            : avg_form_score >= 0.60 ? "#f59e0b"
            : "#ef4444"
          } />
        </div>

        {/* Form score bar */}
        <div className="score-bar-wrapper">
          <div className="score-bar-label">
            <span>Form Score</span>
            <span style={{ color: grade.color }}>{scoreStr}</span>
          </div>
          <div className="score-bar-track">
            <div
              className="score-bar-fill"
              style={{ width: scoreStr, background: grade.color }}
            />
          </div>
        </div>

        {/* Common issues */}
        {common_dangers.length > 0 && (
          <div className="summary-issues">
            <h4>Common issues this session:</h4>
            <ul>
              {common_dangers.map((code) => (
                <li key={code}>⚠️ {DANGER_LABELS[code] ?? code}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Advice */}
        <div className="summary-advice">
          <SummaryAdvice grade={final_grade} dangers={common_dangers} />
        </div>

        {/* Actions */}
        <div className="summary-actions">
          <button className="restart-btn primary" onClick={onRestart}>
            🔁 New Session
          </button>
        </div>
      </div>
    </div>
  );
}

function StatBox({ label, value, icon, color }) {
  return (
    <div className="stat-box">
      <span className="stat-icon">{icon}</span>
      <span className="stat-value" style={color ? { color } : {}}>{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  );
}

function SummaryAdvice({ grade, dangers }) {
  if (grade === "A") return <p>🔥 Outstanding session! Your form was consistent throughout. Keep it up!</p>;
  if (grade === "B") {
    if (dangers.includes("not_deep_enough")) return <p>💪 Solid session! Focus on going a bit deeper next time to maximise results.</p>;
    return <p>💪 Great session! Small consistency improvements will get you to an A.</p>;
  }
  if (grade === "C") {
    if (dangers.includes("back_rounding")) return <p>👍 Work in progress! Prioritise keeping your chest up — try reducing weight and practising slower reps.</p>;
    return <p>👍 Good effort! Focus on one correction at a time. Slow down and be deliberate.</p>;
  }
  return <p>🔄 Don't give up! Consider practising bodyweight squats in front of a mirror to build muscle memory.</p>;
}
