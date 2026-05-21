

const EXERCISES = [
  {
    id: "squat",
    name: "Squat",
    icon: "🦵",
    description: "Lower body compound movement",
    tips: ["Full body in frame", "Feet shoulder-width apart", "Arms forward for balance"],
  },
];

export default function ExerciseSelector({ onSelect }) {
  return (
    <div className="selector-screen">

      {/* ── Ambient glow blobs ── */}
      <div className="glow glow-left" />
      <div className="glow glow-right" />

      {/* ── Header ── */}
      <div className="selector-header">
        <div className="logo-row">
          <span className="logo-icon">🤸 </span>
          <h1>Spotter</h1>
        </div>
        <p className="tagline">AI-powered real-time posture correction</p>
        <div className="badge-row">
          <span className="badge">⚡ Live Feedback</span>
          <span className="badge">🧠 LSTM Model</span>
          <span className="badge">📐 Joint Angles</span>
        </div>
      </div>

      {/* ── Exercise card ── */}
      <div className="exercise-grid">
        {EXERCISES.map((ex) => (
          <button key={ex.id} className="exercise-card" onClick={() => onSelect(ex)}>
            <div className="card-icon-wrap">
              <span className="exercise-icon">{ex.icon}</span>
            </div>
            <h2>{ex.name}</h2>
            <p className="exercise-desc">{ex.description}</p>
            <ul className="exercise-tips">
              {ex.tips.map((tip, i) => (
                <li key={i}>
                  <span className="tip-dot" />
                  {tip}
                </li>
              ))}
            </ul>
            <div className="start-btn">
              Start Session
              <span className="start-arrow">→</span>
            </div>
          </button>
        ))}
      </div>

      <p className="disclaimer">📷 Camera access required &nbsp;·&nbsp; Works best in good lighting</p>
    </div>
  );
}
