"""
webcam_test.py  —  Spotter AI Live Demo  (v5 – fully fixed)
============================================================
Run:  python webcam_test.py

USER KEYS  (shown on screen):
    Q       quit
    R       reset rep counter
    S       save screenshot
    SPACE   pause / resume
    V       toggle voice on / off

DEVELOPER KEYS  (not shown on screen):
    C  — calibrate: stand upright in frame then press C to lock YOUR standing
         knee angle as the idle baseline (instead of the generic 158° default)
    L  — flip model labels: if the ML model calls GOOD when it should be BAD
         (or vice versa), press L once to invert its output
    D  — debug overlay: shows raw model output, vote%, stage, avg_knee, etc.

WHAT EACH VISUAL ELEMENT MEANS:
    Top badge        form quality — ONLY visible during active squat descent
                     "Ready" when standing, "Step into frame" when body missing
    Red vignette     full-screen pulsing red border = BAD FORM during squat
    Green vignette   subtle green border = GOOD FORM during squat
    ⚠ alerts         bottom-of-screen — specific faults (back round, shallow…)
    REPS panel       top-left — counted only when full body visible + hit depth
    Angle bars       top-right — live joint angles, green/yellow/red coded
    Depth bar        right edge — how deep your squat is as a percentage
    Orange overlay   your full body is not in frame — step further back

VOICE:
    Fires AT MOST ONCE per descent at the deepest point.
    "Good form" / "Check your form" — once per squat, not repeated.
    Each danger cue (chest up, go deeper, balance) also once per squat.
    Rep number spoken the moment the rep completes.
"""

import sys, os, time, cv2, numpy as np, threading, queue
from collections import deque
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app.ai.pose.mediapipe_engine import detect_pose
from backend.app.ai.ml.model_loader        import predict_sequence
from backend.app.ai.pose.rep_counter       import RepCounter

SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
REQUIRED_LANDMARKS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP",      "RIGHT_HIP",
    "LEFT_KNEE",     "RIGHT_KNEE",
    "LEFT_ANKLE",    "RIGHT_ANKLE",
]
MIN_REP_DEPTH        = 145   # knee angle must reach this to count a rep
IDLE_THRESHOLD       = 158   # above this = standing, no form judgment
VOICE_GATE_THRESHOLD = 142   # voice only fires when knee is below this angle

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_GREEN  = (50,  205,  50)
C_RED    = (30,   30, 220)
C_YELLOW = (0,   200, 255)
C_WHITE  = (255, 255, 255)
C_BLACK  = (0,     0,   0)
C_GREY   = (110, 110, 110)
C_ORANGE = (0,   150, 255)
C_PANEL  = (22,   22,  22)
C_CYAN   = (230, 210,   0)
C_TEAL   = (160, 200,  80)

BONES = [
    ("LEFT_SHOULDER","RIGHT_SHOULDER"), ("LEFT_SHOULDER","LEFT_HIP"),
    ("RIGHT_SHOULDER","RIGHT_HIP"),     ("LEFT_HIP","RIGHT_HIP"),
    ("LEFT_HIP","LEFT_KNEE"),           ("LEFT_KNEE","LEFT_ANKLE"),
    ("RIGHT_HIP","RIGHT_KNEE"),         ("RIGHT_KNEE","RIGHT_ANKLE"),
    ("LEFT_SHOULDER","LEFT_ELBOW"),     ("LEFT_ELBOW","LEFT_WRIST"),
    ("RIGHT_SHOULDER","RIGHT_ELBOW"),   ("RIGHT_ELBOW","RIGHT_WRIST"),
]


# ══════════════════════════════════════════════════════════════════════════════
# VOICE ENGINE  — no internal cooldown; caller controls exactly when it fires
# ══════════════════════════════════════════════════════════════════════════════
class VoiceEngine:
    def __init__(self):
        self._q      = queue.Queue()
        self._on     = True
        self._kind   = self._detect()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _detect(self):
        import shutil, platform
        if platform.system() == "Darwin" and shutil.which("say"):
            return "say"
        try:
            import pyttsx3; return "pyttsx3"
        except ImportError:
            return None

    def _worker(self):
        engine = None
        if self._kind == "pyttsx3":
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
        while True:
            phrase = self._q.get()
            if phrase is None: break
            try:
                if self._kind == "say":
                    import subprocess
                    subprocess.run(["say", "-r", "175", phrase],
                                   capture_output=True, timeout=6)
                elif engine:
                    engine.say(phrase); engine.runAndWait()
            except Exception:
                pass

    def speak(self, phrase):
        """Queue phrase — no internal cooldown; caller guarantees per-descent firing."""
        if not self._on or not self._kind: return
        if self._q.qsize() < 2:
            self._q.put(phrase)

    def toggle(self):
        self._on = not self._on
        return self._on

    def shutdown(self):
        self._q.put(None)


# ══════════════════════════════════════════════════════════════════════════════
# BODY COMPLETENESS  — all 8 key landmarks + anatomical plausibility
# ══════════════════════════════════════════════════════════════════════════════
def check_body_complete(lm):
    if not all(k in lm for k in REQUIRED_LANDMARKS):
        return False
    try:
        # Y grows downward in image space
        if not (lm["LEFT_HIP"][1]  < lm["LEFT_KNEE"][1]  < lm["LEFT_ANKLE"][1]):  return False
        if not (lm["RIGHT_HIP"][1] < lm["RIGHT_KNEE"][1] < lm["RIGHT_ANKLE"][1]): return False
        if lm["LEFT_SHOULDER"][1]  >= lm["LEFT_HIP"][1]:  return False
        if lm["RIGHT_SHOULDER"][1] >= lm["RIGHT_HIP"][1]: return False
        if abs(lm["LEFT_HIP"][0]   - lm["RIGHT_HIP"][0]) < 20: return False
    except Exception:
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# MATHS
# ══════════════════════════════════════════════════════════════════════════════
def angle_at(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def compute_angles(lm):
    lk = angle_at(lm["LEFT_HIP"],      lm["LEFT_KNEE"],  lm["LEFT_ANKLE"])
    rk = angle_at(lm["RIGHT_HIP"],     lm["RIGHT_KNEE"], lm["RIGHT_ANKLE"])
    lh = angle_at(lm["LEFT_SHOULDER"], lm["LEFT_HIP"],   lm["LEFT_KNEE"])
    rh = angle_at(lm["RIGHT_SHOULDER"],lm["RIGHT_HIP"],  lm["RIGHT_KNEE"])
    return {"left_knee":lk, "right_knee":rk,
            "left_hip":lh,  "right_hip":rh,  "avg_knee":(lk+rk)/2}

def build_features(a):
    lk, rk, lh, rh = a["left_knee"], a["right_knee"], a["left_hip"], a["right_hip"]
    return [lk/180, rk/180, lh/180, rh/180,
            abs(lk-rk)/180, abs(lh-rh)/180, ((lk+rk)/2)/180]

def detect_dangers(angles, cal=None):
    """Returns list of (code, message) for specific form faults."""
    alerts = []
    lh, rh  = angles["left_hip"],  angles["right_hip"]
    lk, rk  = angles["left_knee"], angles["right_knee"]
    avg_k   = angles["avg_knee"]
    thr     = (cal - 10) if cal else 155
    if avg_k < thr:
        if lh < 65 or rh < 65:    alerts.append(("back",      "Back rounding — chest up!"))
        if abs(lk - rk) > 22:     alerts.append(("imbalance", "Uneven — balance both legs"))
        if 108 < avg_k < 148:     alerts.append(("shallow",   "Go deeper for full range!"))
    return alerts

def angle_based_form_override(angles):
    """
    Hard angle check — overrides ML when body geometry clearly shows bad form.
    Returns "BAD" or None (defer to ML).
    """
    lh, rh = angles["left_hip"], angles["right_hip"]
    lk, rk = angles["left_knee"], angles["right_knee"]
    if lh < 60 or rh < 60:      return "BAD"   # serious back rounding
    if abs(lk - rk) > 30:       return "BAD"   # severe knee imbalance
    return None

def decide_form(last_label, conf, angles, flipped, cal, stage):
    """
    Returns (form_str, confidence, reason).
      IDLE    — standing / not squatting
      UNKNOWN — warming up or low confidence
      GOOD    — good form during descent
      BAD     — bad form during descent
    """
    avg_k = angles["avg_knee"]
    thr   = (cal - 8) if cal else IDLE_THRESHOLD

    if avg_k > thr or stage != "down":
        return "IDLE", 1.0, "idle"

    # Hard angle override first
    override = angle_based_form_override(angles)
    if override == "BAD":
        return "BAD", 0.93, "angles"

    if conf < 0.65:
        return "UNKNOWN", conf, "low-conf"

    label = (1 - last_label) if flipped else last_label
    return ("GOOD" if label == 1 else "BAD"), conf, "model"


# ══════════════════════════════════════════════════════════════════════════════
# GUARDED REP COUNTER
# Only counts reps where:  body fully visible  AND  knee reached MIN_REP_DEPTH
# ══════════════════════════════════════════════════════════════════════════════
class GuardedRepCounter:
    def __init__(self):
        self._inner          = RepCounter()
        self._reached_depth  = False
        self._last_stage     = "up"
        self._guarded_reps   = 0
        self._just_completed = False
        self._new_descent    = False

    def update(self, avg_knee, body_complete):
        self._just_completed = False
        self._new_descent    = False
        if not body_complete: return

        if avg_knee < MIN_REP_DEPTH:
            self._reached_depth = True

        prev = self._last_stage
        self._inner.update(avg_knee)
        cur  = self._inner.get_stage()

        if prev == "up" and cur == "down":
            self._new_descent = True

        if prev == "down" and cur == "up":
            if self._reached_depth:
                self._guarded_reps  += 1
                self._just_completed = True
            self._reached_depth = False

        self._last_stage = cur

    def get_count(self):      return self._guarded_reps
    def get_stage(self):      return self._inner.get_stage()
    def just_completed(self): return self._just_completed
    def new_descent(self):    return self._new_descent

    def reset(self):
        self._inner         = RepCounter()
        self._reached_depth = False
        self._last_stage    = "up"
        self._guarded_reps  = 0
        self._just_completed = False
        self._new_descent    = False


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def _glow_line(frame, p1, p2, col, t=3):
    ov = frame.copy()
    gc = tuple(min(255, int(c * 0.40)) for c in col)
    cv2.line(ov, p1, p2, gc, t+5, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.28, frame, 0.72, 0, frame)
    cv2.line(frame, p1, p2, col, t, cv2.LINE_AA)

def draw_skeleton(frame, lm, form, conf, body_complete):
    if   not body_complete: bone_col = C_ORANGE
    elif form == "GOOD":    bone_col = C_GREEN
    elif form == "BAD":     bone_col = C_RED
    else:                   bone_col = C_GREY

    if form not in ("UNKNOWN","IDLE") and body_complete and conf < 0.85:
        a = max(0.0, min(1.0, (conf - 0.65) / 0.20))
        bone_col = tuple(int(C_GREY[i] + a * (bone_col[i] - C_GREY[i])) for i in range(3))

    for a_, b_ in BONES:
        if a_ in lm and b_ in lm:
            _glow_line(frame,
                       (int(lm[a_][0]), int(lm[a_][1])),
                       (int(lm[b_][0]), int(lm[b_][1])), bone_col)

    for _, (x, y) in lm.items():
        cx, cy = int(x), int(y)
        gc = tuple(min(255, int(c * 0.5)) for c in bone_col)
        cv2.circle(frame, (cx, cy), 11, gc,       -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy),  8, C_WHITE,  -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy),  8, bone_col,  2, cv2.LINE_AA)

def draw_angle_labels(frame, lm, angles):
    for jnt, akey in [("LEFT_KNEE","left_knee"), ("RIGHT_KNEE","right_knee")]:
        if jnt not in lm: continue
        val = angles.get(akey, 0)
        col = C_GREEN if val > 150 else (C_YELLOW if val > 100 else C_RED)
        x, y = int(lm[jnt][0]) + 14, int(lm[jnt][1])
        txt  = f"{int(val)}\u00b0"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
        cv2.rectangle(frame, (x-3, y-th-5), (x+tw+3, y+5), (0,0,0), -1)
        cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, col, 2, cv2.LINE_AA)

def draw_depth_bar(frame, avg_knee, h, w, body_complete):
    bh   = 220;  bx = w - 34;  bt = h//2 - bh//2
    pct  = 1.0 - max(0, min(1, (avg_knee - 75) / 85))
    fill = int(bh * pct)
    col  = (C_GREEN if pct > 0.65 else (C_YELLOW if pct > 0.35 else C_ORANGE)) if body_complete else C_GREY
    cv2.rectangle(frame, (bx, bt), (bx+20, bt+bh), C_PANEL, -1)
    if body_complete:
        cv2.rectangle(frame, (bx, bt+bh-fill), (bx+20, bt+bh), col, -1)
    cv2.rectangle(frame, (bx, bt), (bx+20, bt+bh), C_GREY, 1)
    ty = bt + bh - int(bh * 0.65)
    cv2.line(frame, (bx-5, ty), (bx+25, ty), C_GREEN, 1)
    cv2.putText(frame, "DEPTH", (bx-14, bt-8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GREY, 1)
    if body_complete:
        cv2.putText(frame, f"{int(pct*100)}%", (bx-6, bt+bh+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

def draw_form_badge(frame, form, conf, body_complete, w, pt):
    """
    Top-centre badge — 4 clean states:
      Orange  "Step into frame"      full body not visible
      Grey    "Ready"                standing / idle
      Green   "✓ GOOD FORM  XX%"    squatting + good
      Red     "✗ BAD FORM   XX%"    squatting + bad  (pulsing)
    """
    if not body_complete:
        text = "  Step into frame — back up  "
        col  = C_ORANGE;  bg = (28, 18, 0)
    elif form in ("IDLE", "UNKNOWN"):
        text = "  Ready  "
        col  = C_GREY;    bg = (22, 22, 22)
    elif form == "GOOD":
        text = f"  \u2713 GOOD FORM  {int(conf*100)}%  "
        col  = C_GREEN;   bg = (8, 40, 8)
    else:  # BAD — pulses
        p    = int(20 * abs(np.sin(pt * 3.5)))
        col  = (min(255, 60+p), min(255, 60+p), min(255, 215+p))
        bg   = (0, 0, max(0, 60 + p*2))
        text = f"  \u2717 BAD FORM  {int(conf*100)}%  "

    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.82, 2)
    x0 = w//2 - tw//2 - 18
    cv2.rectangle(frame, (x0+3, 13), (x0+tw+39, 56), C_BLACK, -1)
    cv2.rectangle(frame, (x0,   10), (x0+tw+36, 53), bg,      -1)
    bt = 1 if form in ("IDLE","UNKNOWN") and body_complete else 2
    cv2.rectangle(frame, (x0, 10), (x0+tw+36, 53), col, bt)
    cv2.putText(frame, text, (x0+18, 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, col, 2, cv2.LINE_AA)

def draw_form_vignette(frame, form, pulse_t, h, w):
    """
    Pulsing coloured screen border — RED for bad form, subtle GREEN for good.
    This is the "red overlay" for bad form that makes it unmissable.
    """
    pulse = abs(np.sin(pulse_t * 3.5))
    if form == "BAD":
        alpha = 0.20 + 0.30 * pulse
        col   = (0, 0, 200)       # red in BGR
    elif form == "GOOD":
        alpha = 0.10 + 0.05 * pulse
        col   = (20, 150, 20)     # green in BGR
    else:
        return

    ov = frame.copy()
    border = 28
    cv2.rectangle(ov, (0,0), (w,h), col, border * 2)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

def draw_rep_panel(frame, reps, stage, last_rep_t):
    flash = max(0.0, 1.0 - (time.time() - last_rep_t) / 0.5)
    rc    = tuple(int(C_GREEN[i] + flash * (255 - C_GREEN[i])) for i in range(3))
    cv2.rectangle(frame, (0,0),   (138, 128), C_PANEL, -1)
    cv2.rectangle(frame, (0,0),   (138, 128), C_GREY,   1)
    cv2.putText(frame, "REPS",   (12, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, C_GREY, 1)
    cv2.putText(frame, str(reps),(12, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.8,  rc,    4, cv2.LINE_AA)
    cv2.putText(frame, stage.upper(), (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                C_YELLOW if stage == "down" else C_WHITE, 1)

def draw_angles_panel(frame, angles, w, body_complete):
    pw = 158
    cv2.rectangle(frame, (w-pw, 0), (w, 118), C_PANEL, -1)
    cv2.rectangle(frame, (w-pw, 0), (w, 118), C_GREY,   1)
    for i, (label, key) in enumerate([("L Knee","left_knee"), ("R Knee","right_knee"),
                                       ("L Hip", "left_hip"),  ("R Hip", "right_hip")]):
        val = angles.get(key, 0)
        col = (C_GREEN if val > 150 else (C_YELLOW if val > 100 else C_RED)) if body_complete else C_GREY
        bw  = int(140 * min(val/180, 1.0)) if body_complete else 0
        cv2.rectangle(frame, (w-pw+8, 16+i*25), (w-pw+8+bw, 21+i*25), col, -1)
        txt = f"{label}: {int(val)}\u00b0" if body_complete else f"{label}: --"
        cv2.putText(frame, txt, (w-pw+8, 14+i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

def draw_danger_alerts(frame, dangers, h, w):
    """Red-bordered alert strips along the bottom for specific form faults."""
    for i, (code, msg) in enumerate(dangers):
        y0 = h - 40 - i * 44
        ov = frame.copy()
        cv2.rectangle(ov, (0, y0-32), (w, y0+10), (10, 0, 100), -1)
        cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (0, y0-32), (7, y0+10), C_RED, -1)
        cv2.putText(frame, f"\u26a0  {msg}", (20, y0-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_WHITE, 2, cv2.LINE_AA)

def draw_coaching_tip(frame, msg, h, w):
    if not msg: return
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.70, 2)
    x0 = w//2 - tw//2 - 16;  y = h//2
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y-38), (x0+tw+32, y+12), (0, 55, 10), -1)
    cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
    cv2.rectangle(frame, (x0, y-38), (x0+tw+32, y+12), C_GREEN, 1)
    cv2.putText(frame, msg, (x0+16, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_WHITE, 2, cv2.LINE_AA)

def draw_no_body_overlay(frame, h, w):
    """Orange banner when body is partially detected but not complete."""
    msg = "Full body not in frame — step back"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
    x = w//2 - tw//2
    ov = frame.copy()
    cv2.rectangle(ov, (x-12, h//2-42), (x+tw+12, h//2+14), (28, 18, 0), -1)
    cv2.addWeighted(ov, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x-12, h//2-42), (x+tw+12, h//2+14), C_ORANGE, 2)
    cv2.putText(frame, msg, (x, h//2-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, C_ORANGE, 2, cv2.LINE_AA)

def draw_warmup_bar(frame, n, w):
    if n >= 20: return
    bw = int((w - 40) * n / 20)
    cv2.rectangle(frame, (20, 62), (w-20, 78), C_PANEL, -1)
    cv2.rectangle(frame, (20, 62), (20+bw, 78), C_ORANGE, -1)
    cv2.putText(frame, f"AI warming up... {n}/20 frames",
                (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_ORANGE, 1)

def draw_paused(frame, h, w):
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w,h), C_BLACK, -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, "PAUSED", (w//2-100, h//2-10),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, C_WHITE, 4, cv2.LINE_AA)
    cv2.putText(frame, "Press SPACE to resume", (w//2-155, h//2+40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, C_GREY, 1)

def draw_status_bar(frame, voice_on, voice_kind, h, w):
    """Minimal bottom bar — only user-facing keys."""
    bh = 26
    cv2.rectangle(frame, (0, h-bh), (w, h), (14,14,14), -1)
    voice_txt = "V: Voice ON" if (voice_on and voice_kind) else "V: Voice OFF"
    voice_col = C_TEAL if (voice_on and voice_kind) else C_GREY
    items = [
        ("Q: Quit", C_GREY), ("R: Reset reps", C_GREY),
        ("S: Screenshot", C_GREY), ("SPACE: Pause", C_GREY),
        (voice_txt, voice_col),
    ]
    x = 10
    for txt, col in items:
        cv2.putText(frame, txt, (x, h-7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        x += tw + 24

def draw_debug(frame, raw_label, raw_conf, vote_pct, streak, flipped,
               reason, body_complete, stage, avg_k, voice_active, h, w):
    pw, ph = 400, 110;  x0 = w - pw - 10;  y0 = h - ph - 42
    ov = frame.copy()
    cv2.rectangle(ov,    (x0, y0), (x0+pw, y0+ph), C_BLACK, -1)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0+pw, y0+ph), C_CYAN, 1)
    lines = [
        "DEBUG  (D = hide)",
        f"raw={raw_label}  conf={raw_conf:.3f}  vote%={vote_pct*100:.0f}  streak={streak}",
        f"body={body_complete}  stage={stage}  avg_k={avg_k:.0f}  flipped={flipped}",
        f"voice_gate={voice_active}   reason={reason}",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (x0+8, y0+20+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    C_CYAN if i == 0 else C_WHITE, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("🚀 Spotter AI v5 — fully fixed")
    voice = VoiceEngine()
    print(f"   Voice: {voice._kind or 'none — pip install pyttsx3 or run on macOS'}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No webcam found"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,            30)

    rep_counter  = GuardedRepCounter()
    seq_buf      = deque(maxlen=20)
    pred_buf     = deque(maxlen=15)

    last_pred_t  = 0.0
    raw_label    = 1;   raw_conf = 1.0
    last_label   = 1;   last_label_t = 0.0;  form_conf = 1.0
    consec_bad   = 0;   bad_streak = 0
    coaching_msg = None;  coaching_until = 0.0
    screenshot_n = 0
    paused       = False
    flipped      = False
    debug_on     = False
    cal          = None
    form_reason  = "init"
    vote_pct     = 1.0
    pulse_t      = 0.0
    last_rep_t   = 0.0
    form         = "UNKNOWN"

    # Per-descent voice gate flags
    _form_spoken   = False         # True once "Good/Bad form" spoken this descent
    _dangers_spoken = set()        # set of danger codes already spoken this descent

    angles = {k: 180 for k in ["left_knee","right_knee","left_hip","right_hip","avg_knee"]}

    voice.speak("Spotter ready")
    print("   ✅  Keys: Q  R  S  SPACE  V  |  Dev: C  L  D\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        pulse_t += 0.033

        # ── Pause loop ─────────────────────────────────────────────────────
        if paused:
            draw_paused(frame, h, w)
            cv2.imshow("Spotter AI", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            if k == ord(" "): paused = False
            continue

        # ── Pose ───────────────────────────────────────────────────────────
        lm            = detect_pose(frame)
        body_complete = bool(lm) and check_body_complete(lm)
        stage         = rep_counter.get_stage()
        avg_k         = angles["avg_knee"]
        dangers       = []
        voice_active  = False

        if body_complete:
            angles = compute_angles(lm)
            avg_k  = angles["avg_knee"]

            # Rep counter
            rep_counter.update(avg_k, body_complete=True)
            stage = rep_counter.get_stage()

            # New descent → reset per-descent voice flags
            if rep_counter.new_descent():
                _form_spoken    = False
                _dangers_spoken = set()

            # Rep just finished → speak count, reset flags for next squat
            if rep_counter.just_completed():
                last_rep_t      = time.time()
                voice.speak(str(rep_counter.get_count()))
                _form_spoken    = False
                _dangers_spoken = set()

            # Voice gate: only when knee is genuinely bent and descending
            voice_active = (stage == "down" and avg_k < VOICE_GATE_THRESHOLD)

            # Danger detection (only during descent)
            dangers = detect_dangers(angles, cal) if stage == "down" else []

            # ML prediction
            seq_buf.append(build_features(angles))
            if len(seq_buf) == 20 and time.time() - last_pred_t > 0.3:
                try:
                    label, conf = predict_sequence(list(seq_buf))
                    raw_label = label;  raw_conf = conf
                    if conf > 0.65:
                        pred_buf.append(label)
                    if len(pred_buf) >= 8:
                        votes    = list(pred_buf)
                        vote_pct = votes.count(1) / len(votes)
                        bad_pct  = 1.0 - vote_pct
                        if   vote_pct >= 0.60: proposed = 1
                        elif bad_pct  >= 0.60: proposed = 0
                        else:                  proposed = last_label
                        if proposed == last_label:
                            consec_bad = 0;  confirmed = proposed
                        elif proposed == 0:
                            consec_bad += 1
                            confirmed = last_label if consec_bad < 3 else 0
                        else:
                            consec_bad = 0;  confirmed = proposed
                        if time.time() - last_label_t > 0.6:
                            last_label = confirmed;  last_label_t = time.time()
                    form_conf   = conf
                    last_pred_t = time.time()
                except Exception:
                    pass

            form, display_conf, form_reason = decide_form(
                last_label, form_conf, angles, flipped, cal, stage
            )
            form_conf = display_conf

            # ── Voice — fires at most once per descent for form + once per danger ──
            if voice_active:
                if not _form_spoken:
                    if form == "GOOD":
                        voice.speak("Good form")
                        _form_spoken = True
                    elif form == "BAD":
                        voice.speak("Check your form")
                        _form_spoken = True

                for code, _ in dangers:
                    if code not in _dangers_spoken:
                        if   code == "back":      voice.speak("Chest up, back straight")
                        elif code == "shallow":   voice.speak("Go deeper")
                        elif code == "imbalance": voice.speak("Balance both legs")
                        _dangers_spoken.add(code)

            # Coaching streak
            if (form == "BAD" or dangers) and voice_active:
                bad_streak += 1
            else:
                bad_streak = max(0, bad_streak - 1)
            if bad_streak >= 6 and time.time() > coaching_until and voice_active:
                coaching_msg = (dangers[0][1] if dangers else "Slow down — focus on control")
                voice.speak(coaching_msg)
                coaching_until = time.time() + 5.0
                bad_streak = 0
            if time.time() > coaching_until:
                coaching_msg = None

            draw_skeleton(frame, lm, form, form_conf, body_complete=True)
            draw_angle_labels(frame, lm, angles)
            draw_depth_bar(frame, avg_k, h, w, body_complete=True)
            if stage == "down":
                draw_danger_alerts(frame, dangers, h, w)

        else:
            form = "UNKNOWN";  form_conf = 1.0
            stage  = rep_counter.get_stage()
            dangers = []
            pred_buf.clear()
            last_label     = 1
            form_reason    = "no-body"
            _form_spoken   = False
            _dangers_spoken = set()

            if lm:
                draw_skeleton(frame, lm, "UNKNOWN", 0.0, body_complete=False)
                draw_no_body_overlay(frame, h, w)
            else:
                cv2.putText(frame, "No person detected",
                            (w//2-140, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, C_ORANGE, 2)
            draw_depth_bar(frame, 180, h, w, body_complete=False)

        # ── Screen-edge vignette (drawn BEFORE badge so badge is on top) ──
        if body_complete and form in ("GOOD","BAD") and stage == "down":
            draw_form_vignette(frame, form, pulse_t, h, w)

        # ── Always-on UI ───────────────────────────────────────────────────
        draw_form_badge(frame, form, form_conf, body_complete, w, pulse_t)
        draw_rep_panel(frame, rep_counter.get_count(), stage, last_rep_t)
        draw_angles_panel(frame, angles, w, body_complete)
        if body_complete and stage == "down":
            draw_coaching_tip(frame, coaching_msg, h, w)
        draw_warmup_bar(frame, len(seq_buf), w)

        if cal:
            cv2.putText(frame, f"Cal:{int(cal)}\u00b0",
                        (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_CYAN, 1)

        if debug_on:
            draw_debug(frame, raw_label, raw_conf, vote_pct, bad_streak, flipped,
                       form_reason, body_complete, stage, avg_k, voice_active, h, w)

        draw_status_bar(frame, voice._on, voice._kind, h, w)
        cv2.imshow("Spotter AI", frame)

        # ── Keys ───────────────────────────────────────────────────────────
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("r"):
            rep_counter.reset()
            seq_buf.clear();  pred_buf.clear()
            last_label = 1;   form_conf = 1.0;  raw_label = 1
            bad_streak = 0;   last_rep_t = 0.0
            _form_spoken   = False;  _dangers_spoken = set()
            voice.speak("Counter reset")
        elif k == ord("s"):
            p = SCREENSHOT_DIR / f"screenshot_{screenshot_n:03d}.jpg"
            cv2.imwrite(str(p), frame)
            screenshot_n += 1
            print(f"📸  Saved {p}")
        elif k == ord(" "):
            paused = True
            voice.speak("Paused")
        elif k == ord("c"):   # developer: calibrate
            if body_complete:
                cal = angles["avg_knee"]
                voice.speak("Calibrated")
                print(f"📐  Standing angle calibrated = {int(cal)}°")
            else:
                print("⚠️   Full body not in frame — cannot calibrate")
        elif k == ord("d"):   # developer: debug overlay
            debug_on = not debug_on
        elif k == ord("l"):   # developer: flip model labels
            flipped = not flipped
            pred_buf.clear();  last_label = 1
            print(f"🔀  Label flip = {flipped}")
        elif k == ord("v"):
            on = voice.toggle()
            print(f"🔊  Voice {'ON' if on else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    voice.shutdown()
    print(f"\n📊  Session finished — {rep_counter.get_count()} reps counted")


if __name__ == "__main__":
    main()