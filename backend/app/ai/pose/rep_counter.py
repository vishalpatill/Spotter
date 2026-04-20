import time


class RepCounter:
    def __init__(self):
        self.count = 0
        self.stage = "up"
        self.down_start_time = None
        self.up_start_time = time.time()   # ✅ NEW

        # 🔥 TUNED THRESHOLDS (IMPORTANT)
        self.DOWN_ANGLE = 120
        self.UP_ANGLE = 150

        self.MIN_DOWN_TIME = 0.4
        self.MIN_UP_TIME = 0.3   # ✅ NEW (prevents double counts)

        # smoothing
        self.angle_history = []
        self.SMOOTHING_WINDOW = 5

    def smooth_angle(self, angle):
        self.angle_history.append(angle)
        if len(self.angle_history) > self.SMOOTHING_WINDOW:
            self.angle_history.pop(0)

        return sum(self.angle_history) / len(self.angle_history)

    def update(self, knee_angle):
        current_time = time.time()

        # smooth angle
        knee_angle = self.smooth_angle(knee_angle)

        # --------------------------
        # GOING DOWN
        # --------------------------
        if knee_angle < self.DOWN_ANGLE:
            if self.stage != "down":
                self.stage = "down"
                self.down_start_time = current_time

        # --------------------------
        # COMING UP (COUNT REP)
        # --------------------------
        elif knee_angle > self.UP_ANGLE:
            if self.stage == "down":

                # ensure enough time in squat
                if self.down_start_time and (current_time - self.down_start_time > self.MIN_DOWN_TIME):

                    # ✅ NEW: ensure user actually came up properly
                    if (current_time - self.up_start_time) > self.MIN_UP_TIME:
                        self.count += 1
                        self.stage = "up"
                        self.down_start_time = None
                        self.up_start_time = current_time   # reset

                        return True

        return False

    def get_count(self):
        return self.count

    def get_stage(self):
        return self.stage