# 🏋️ Spotter — Frontend

React frontend for the **Spotter AI** posture correction app.  
Connects to the Spotter FastAPI backend for real-time squat form analysis.

---

## What This Does

- Opens the webcam and sends frames to your backend at ~8fps
- Displays real-time form feedback (GOOD / BAD), joint angles, danger alerts
- Counts reps and tracks stage (UP / DOWN)
- Shows adaptive coaching tips after repeated bad form
- End-of-session summary with grade, score, and common mistakes

---

## Tech Stack

| Layer     | Tech                        |
|-----------|-----------------------------|
| Frontend  | React 18 + Vite             |
| Styling   | Plain CSS (no Tailwind)     |
| Transport | REST — multipart/form-data  |
| API       | Your FastAPI backend        |

---

## Project Structure

```
src/
├── api/
│   └── spotterApi.js          ← All API calls live here
├── components/
│   ├── ExerciseSelector.jsx   ← Landing screen
│   ├── PostureChecker.jsx     ← Webcam + frame sender + skeleton overlay
│   ├── FeedbackPanel.jsx      ← Right panel: form verdict, angles, alerts
│   └── SessionSummary.jsx     ← Post-workout grade screen
├── utils/
│   └── skeletonDrawer.js      ← Draws MediaPipe skeleton on canvas
├── App.jsx                    ← Screen router (select → workout → summary)
└── App.css                    ← All styles
```

---

## How to Run

### 1. Install dependencies
```bash
npm install
```

### 2. Point it at your backend
Create a `.env` file in the project root:
```
VITE_API_URL=http://<your-laptop-ip>:8000
```

If running locally on the same machine:
```
VITE_API_URL=http://localhost:8000
```

To find your IP on the same WiFi:
```bash
# Mac / Linux
ifconfig | grep "inet "

# Windows
ipconfig | findstr "IPv4"
```

### 3. Start the dev server
```bash
npm run dev
```

Open **http://localhost:5173**

---

## Backend API Contract

The frontend talks to these 4 endpoints on your FastAPI server.  
All of these already exist in `backend/app/ai/main.py`.

### `POST /session/start`
Called once when the user clicks "Start".

```
Query param: ?exercise=squat
Response:    { session_id, exercise, message }
```

---

### `POST /session/frame`
Called every ~120ms with a JPEG frame from the webcam.

```
Body: multipart/form-data
  - file:       JPEG image blob
  - session_id: string from /session/start

Response:
{
  ok,                  // true | false
  session_id,
  reps,                // int
  stage,               // "up" | "down"
  form,                // "GOOD" | "BAD" | "UNKNOWN"
  form_confidence,     // 0.0 - 1.0
  posture_quality,     // "good" | "ok" | "bad" | "unknown"
  angles: {
    left_knee, right_knee,
    left_hip,  right_hip,
    avg_knee
  },
  danger_alerts: [
    { code: "back_rounding",   message: "..." },
    { code: "not_deep_enough", message: "..." },
    { code: "imbalance",       message: "..." }
  ],
  adaptive_feedback,   // string | null  (shows after 3 bad reps in a row)
  landmarks,           // array of { x, y } normalised 0-1
  exercise_detected,   // string
}
```

---

### `GET /session/{session_id}/summary`
Called when user clicks "End & Summary".

```
Response:
{
  session_id,
  exercise,
  total_reps,
  duration_seconds,
  avg_form_score,     // 0.0 - 1.0
  common_dangers,     // ["back_rounding", ...]
  final_grade,        // "A" | "B" | "C" | "D"
}
```

---

### `DELETE /session/{session_id}`
Called after summary is fetched to clean up server memory.

```
Response: { message: "Session ended" }
```

---

## Starting Your Backend

From the SPOTTER root directory:

```bash
uvicorn backend.app.ai.main:app --reload --host 0.0.0.0 --port 8000
```

Make sure the frontend `.env` points to your machine's IP if running on separate laptops on the same WiFi.

---

## CORS

Your backend already has CORS enabled for all origins:
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```
No changes needed.

---

## Adding More Exercises

Right now only **Squat** is wired up because that's what your backend supports.  
When you add plank / bicep curl to the backend, update these two files:

**`src/components/ExerciseSelector.jsx`** — add to the `EXERCISES` array:
```js
{ id: "plank",      name: "Plank",      icon: "🏋️", description: "...", tips: [...] },
{ id: "bicep_curl", name: "Bicep Curl", icon: "💪", description: "...", tips: [...] },
```

**`src/api/spotterApi.js`** — pass the exercise id when starting a session:
```js
POST /session/start?exercise=plank
```

---

## Common Issues

| Problem | Fix |
|---|---|
| Blank screen / API Error overlay | Check `VITE_API_URL` in `.env` and make sure backend is running |
| CORS error in browser console | Check `allow_origins=["*"]` is set in your FastAPI CORS middleware |
| `No person detected` on screen | Step back so full body is visible to the webcam |
| Skeleton not drawing | Backend `landmarks` must be an array of `{ x, y }` normalised 0-1 |
| Form stays UNKNOWN | Need 20 frames buffered before LSTM runs — keep moving for a few seconds |
