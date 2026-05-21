import { useState } from "react";
import ExerciseSelector from "./components/ExerciseSelector";
import PostureChecker   from "./components/PostureChecker";
import SessionSummary   from "./components/SessionSummary";
import "./App.css";

// screens: "select" | "workout" | "summary"
export default function App() {
  const [screen,   setScreen]   = useState("select");
  const [exercise, setExercise] = useState(null);
  const [summary,  setSummary]  = useState(null);

  function handleExerciseSelect(ex) {
    setExercise(ex);
    setScreen("workout");
  }

  function handleWorkoutEnd(summaryData) {
    setSummary(summaryData);
    setScreen("summary");
  }

  function handleRestart() {
    setSummary(null);
    setScreen("select");
  }

  return (
    <div className="app">
      {screen === "select"  && <ExerciseSelector onSelect={handleExerciseSelect} />}
      {screen === "workout" && (
        <PostureChecker
          exercise={exercise}
          onBack={() => setScreen("select")}
          onEnd={handleWorkoutEnd}
        />
      )}
      {screen === "summary" && (
        <SessionSummary
          summary={summary}
          exercise={exercise}
          onRestart={handleRestart}
        />
      )}
    </div>
  );
}
