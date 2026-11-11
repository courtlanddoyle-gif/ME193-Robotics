// Finger Lock - microcontroller side.
//
// Python (python/main.py) does the camera, CNN and lock decisions; this
// sketch just runs the hardware:
//   - draws frames on the 8x13 LED matrix ("the LEDs light up")
//   - runs the two lid motors together through the dual-PWM driver (same
//     driver and pins as the Door2Door car: PWM xA to go one way, xB the
//     other way). Each only turns ~1/4 turn into a hard stop, so they run
//     for a short time limit and then cut power instead of sitting stalled.
//   - reads the optional buttons and limit switches and tells Python
//
// Wiring:
//   driver M1A -> D5, M1B -> D6, motor 1 -> Motor 1 terminals
//   driver M2A -> D9, M2B -> D10, motor 2 -> Motor 2 terminals
//   driver VB+ -> 5V, VB- -> GND
//   If one motor turns the wrong way, swap its two wires at the driver.
//   limit switch "open"   D2 -> switch -> GND   (optional)
//   limit switch "closed" D3 -> switch -> GND   (optional)
//   button 1 "start"      D4 -> button -> GND   (optional)
//   button 2 "close"      D7 -> button -> GND   (optional)
// Everything uses INPUT_PULLUP, so an unconnected pin reads "not pressed",
// and the latch then just runs for its time limit.

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

Arduino_LED_Matrix matrix;
const uint8_t FRAME_SIZE = 8 * 13;
uint8_t frame[FRAME_SIZE] = {0};

const int MOTOR1_A = 5;
const int MOTOR1_B = 6;
const int MOTOR2_A = 9;
const int MOTOR2_B = 10;
// Which way the motors spin to OPEN the lid (+1 or -1). Flip this if both
// go the wrong way; if just one does, swap that motor's wires instead.
const int OPEN_SPIN = -1;
// Motor 2 starts this much later, so the two start-up surges don't add up.
const unsigned long MOTOR2_DELAY_MS = 40;
const int LIMIT_OPEN = 2;
const int LIMIT_CLOSED = 3;
const int BUTTON_START = 4;
const int BUTTON_CLOSE = 7;

// latch move in progress
int moveDir = 0;              // +1 opening, -1 closing, 0 stopped
int moveSpeed = 0;            // target PWM
unsigned long moveStart = 0;
unsigned long moveUntil = 0;  // millis() when the time limit runs out
// Soft start: ramp up over this long. Jumping straight to full PWM pulled
// the board's 5 V down enough to drop the USB webcam.
const unsigned long RAMP_MS = 150;

uint8_t lastInputs = 0;
unsigned long lastInputChange = 0;

void driveMotor(int pinA, int pinB, int speed) {
  speed = constrain(speed, -255, 255);
  analogWrite(pinA, speed > 0 ? speed : 0);
  analogWrite(pinB, speed < 0 ? -speed : 0);
}

void stopMotors() {
  driveMotor(MOTOR1_A, MOTOR1_B, 0);
  driveMotor(MOTOR2_A, MOTOR2_B, 0);
}

// Soft-start speed for a motor that started `t` ms ago.
int ramped(unsigned long t) {
  return t >= RAMP_MS ? moveSpeed : (int)(moveSpeed * t / RAMP_MS);
}

bool pressed(int pin) { return digitalRead(pin) == LOW; }

// Called from Python: dir +1 = open, -1 = close. Runs at `speed` (0-255)
// until that end's limit switch closes or `ms` milliseconds pass.
void latch(int dir, int speed, int ms) {
  int limit = dir > 0 ? LIMIT_OPEN : LIMIT_CLOSED;
  if (pressed(limit)) return;          // already there
  moveDir = dir;
  moveSpeed = speed;
  moveStart = millis();
  moveUntil = moveStart + ms;
}

// Called from Python with a full 8x13 frame (brightness 0-7 per pixel).
void draw(std::vector<uint8_t> newFrame) {
  size_t len = min(newFrame.size(), (size_t)FRAME_SIZE);
  memcpy(frame, newFrame.data(), len);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();

  pinMode(MOTOR1_A, OUTPUT);
  pinMode(MOTOR1_B, OUTPUT);
  pinMode(MOTOR2_A, OUTPUT);
  pinMode(MOTOR2_B, OUTPUT);
  stopMotors();
  pinMode(LIMIT_OPEN, INPUT_PULLUP);
  pinMode(LIMIT_CLOSED, INPUT_PULLUP);
  pinMode(BUTTON_START, INPUT_PULLUP);
  pinMode(BUTTON_CLOSE, INPUT_PULLUP);

  Bridge.begin();
  Bridge.provide("draw", draw);
  Bridge.provide("latch", latch);
}

void loop() {
  matrix.draw(frame);

  // stop the latch at its limit switch or time limit
  if (moveDir != 0) {
    int limit = moveDir > 0 ? LIMIT_OPEN : LIMIT_CLOSED;
    if (pressed(limit) || (long)(millis() - moveUntil) >= 0) {
      stopMotors();
      Bridge.notify("latch_done", moveDir, pressed(limit) ? 1 : 0);
      moveDir = 0;
    } else {
      unsigned long t = millis() - moveStart;
      driveMotor(MOTOR1_A, MOTOR1_B, OPEN_SPIN * moveDir * ramped(t));
      int s2 = t >= MOTOR2_DELAY_MS ? ramped(t - MOTOR2_DELAY_MS) : 0;
      driveMotor(MOTOR2_A, MOTOR2_B, OPEN_SPIN * moveDir * s2);
    }
  }

  // buttons + switches as bits; tell Python when they change (debounced)
  uint8_t inputs = (pressed(BUTTON_START) << 0) | (pressed(BUTTON_CLOSE) << 1) |
                   (pressed(LIMIT_OPEN) << 2) | (pressed(LIMIT_CLOSED) << 3);
  if (inputs != lastInputs && millis() - lastInputChange > 30) {
    lastInputs = inputs;
    lastInputChange = millis();
    Bridge.notify("inputs", (int)inputs);
  }
  delay(5);
}
