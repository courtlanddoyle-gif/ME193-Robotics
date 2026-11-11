"""The lock's decision-making, shared by the laptop and UNO Q versions.

Each camera frame, call  policy.update(probs, now)  with the CNN's 6
probabilities.  Buttons call  policy.start()  and  policy.close().

States:
    locked     waiting for the button; LEDs off
    entering   LEDs on, reading digit 1, 2, 3
    clear      digit accepted; waiting for the hand to leave before the next
    unlocked   correct code; stays open until the close button
    denied     wrong code; shows for a few seconds, then back to locked
    lockout    too many wrong tries in a row; ignores the button for a while

A digit only counts when the CNN says the same count (1-5) with at least
CONFIDENCE probability for HOLD_TIME seconds straight.  Between digits the
hand has to leave (CNN says 0) so holding 3 fingers can't enter 3-3-3.
"""

import json
from pathlib import Path

CONFIDENCE = 0.70
HOLD_TIME = 1.0       # seconds a digit must be steady
CLEAR_TIME = 0.4      # seconds of "no fingers" between digits
ENTRY_TIMEOUT = 20.0  # seconds to finish the whole code
SHOW_TIME = 3.0       # seconds the "denied" result stays up
MAX_TRIES = 3
LOCKOUT_TIME = 10.0


def load_code(path=Path(__file__).parent / "lock_config.json"):
    code = json.loads(Path(path).read_text())["code"]
    assert len(code) == 3 and len(set(code)) == 3 and all(1 <= d <= 5 for d in code), \
        "code must be 3 different numbers from 1 to 5"
    return code


class LockPolicy:
    def __init__(self, code):
        self.code = list(code)
        self.state = "locked"
        self.entered = []
        self.candidate = None      # digit currently being held
        self.since = 0.0           # when the candidate (or clear) started
        self.state_time = 0.0      # when we entered the current state
        self.wrong_tries = 0
        self.hold_progress = 0.0   # 0..1, for the display
        self.message = "press the button"

    def _go(self, state, now, message):
        self.state, self.state_time, self.message = state, now, message
        self.candidate, self.since, self.hold_progress = None, now, 0.0

    def start(self, now):
        """Button 1: wake up and start entering a code."""
        if self.state in ("locked", "denied"):
            self.entered = []
            self._go("entering", now, "show digit 1")

    def close(self, now):
        """Button 2: close and lock the box (or cancel an entry)."""
        if self.state in ("unlocked", "entering", "clear", "denied"):
            self.entered = []
            self._go("locked", now, "locked - press the button")

    def leds_on(self):
        return self.state in ("entering", "clear")

    def update(self, probs, now):
        """Feed one frame's CNN output. Returns an event string or None:
        'digit', 'unlock', 'deny', 'lockout', 'timeout'."""
        best = int(max(range(6), key=lambda k: probs[k]))
        sure = probs[best] >= CONFIDENCE

        if self.state == "lockout":
            if now - self.state_time >= LOCKOUT_TIME:
                self.wrong_tries = 0
                self._go("locked", now, "press the button")
            return None
        if self.state == "denied" and now - self.state_time >= SHOW_TIME:
            self._go("locked", now, "press the button")
            return None
        if self.state in ("entering", "clear") and now - self.state_time >= ENTRY_TIMEOUT:
            self.entered = []
            self._go("locked", now, "too slow - press the button")
            return "timeout"

        if self.state == "clear":
            if best == 0 and sure:
                if now - self.since >= CLEAR_TIME:
                    self.state, self.since = "entering", now
                    self.message = f"show digit {len(self.entered) + 1}"
            else:
                self.since = now
            return None

        if self.state != "entering":
            return None

        if 1 <= best <= 5 and sure:
            if best != self.candidate:
                self.candidate, self.since = best, now
            self.hold_progress = min(1.0, (now - self.since) / HOLD_TIME)
            if self.hold_progress >= 1.0:
                self.entered.append(best)
                if len(self.entered) < 3:
                    self.state, self.since = "clear", now
                    self.candidate, self.hold_progress = None, 0.0
                    self.message = "got it - move your hand away"
                    return "digit"
                return self._check(now)
        else:
            self.candidate, self.hold_progress = None, 0.0
        return None

    def _check(self, now):
        if self.entered == self.code:
            self.wrong_tries = 0
            self._go("unlocked", now, "UNLOCKED - press close to lock")
            return "unlock"
        self.wrong_tries += 1
        if self.wrong_tries >= MAX_TRIES:
            self._go("lockout", now, f"too many tries - wait {LOCKOUT_TIME:.0f} s")
            return "lockout"
        self._go("denied", now, f"WRONG CODE ({MAX_TRIES - self.wrong_tries} tries left)")
        return "deny"
