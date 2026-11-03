// SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
//
// SPDX-License-Identifier: MPL-2.0

#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;
Arduino_LED_Matrix matrix;

unsigned long previousMillis = 0;
const long interval = 20; // 20ms -> 50Hz sample rate

// HC-SR04 on the inside of the right calf, aimed at the left leg.
// Echo goes through a 1k / 2k divider (5 V -> 3.3 V) before D10.
const int TRIG_PIN = 9;
const int ECHO_PIN = 10;
const unsigned long PING_INTERVAL_MS = 50;  // 20 Hz; lets old echoes die out
const unsigned long MAX_RANGE_US = 25000;    // ~4.3 m; longer pulse = nothing in range
const unsigned long ECHO_TIMEOUT_US = 40000; // a live sensor always ends its pulse by ~38 ms

// distanceCm codes besides a real distance:
//   -1 = sensor never answered the ping (Echo never went high): wiring/power problem
//   -2 = sensor answered but nothing was in range (big room, tall ceiling): normal
const float DIST_NO_RESPONSE = -1.0;
const float DIST_OUT_OF_RANGE = -2.0;

// The echo pulse is timed by an interrupt so the 50 Hz IMU loop never waits
// on pulseIn (up to ~38 ms with nothing in range).
volatile unsigned long echoRiseUs = 0;
volatile unsigned long echoWidthUs = 0;
volatile bool echoHigh = false;
volatile bool echoReady = false;
volatile bool echoRoseSincePing = false;
unsigned long lastPingMs = 0;
unsigned long pingSentUs = 0;
bool waitingForEcho = false;
float distanceCm = DIST_NO_RESPONSE;  // last reading

void echo_isr() {
  if (digitalRead(ECHO_PIN) == HIGH) {
    echoRiseUs = micros();
    echoHigh = true;
    echoRoseSincePing = true;
  } else if (echoHigh) {
    echoWidthUs = micros() - echoRiseUs;
    echoHigh = false;
    echoReady = true;
  }
}

void service_ultrasonic(unsigned long nowMs) {
  if (echoReady) {
    noInterrupts();
    unsigned long width = echoWidthUs;
    echoReady = false;
    interrupts();
    distanceCm = width < MAX_RANGE_US ? width / 58.0 : DIST_OUT_OF_RANGE;
    waitingForEcho = false;
  } else if (waitingForEcho && micros() - pingSentUs > ECHO_TIMEOUT_US) {
    // Still high = sensor is alive, just hearing nothing; never rose = dead.
    distanceCm = echoRoseSincePing ? DIST_OUT_OF_RANGE : DIST_NO_RESPONSE;
    waitingForEcho = false;
  }

  // Don't ping while the last echo is still high, or the readings mix up.
  if (nowMs - lastPingMs >= PING_INTERVAL_MS && !echoHigh) {
    lastPingMs = nowMs;
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    pingSentUs = micros();
    echoRoseSincePing = false;
    waitingForEcho = true;
  }
}

const uint8_t FRAME_SIZE = 8 * 13;

// Simple glyphs so peers watching the demo can see the call without a laptop screen.
uint8_t FRAME_STAND[FRAME_SIZE] = { // centered square
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

uint8_t FRAME_WALK[FRAME_SIZE] = { // single diagonal stride
    7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0
};

uint8_t FRAME_RUN[FRAME_SIZE] = { // double diagonal, faster cadence
    7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 7, 0, 7, 0, 0, 0
};

uint8_t FRAME_BIKE[FRAME_SIZE] = { // wheel
    0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 0,
    0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0,
    0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0
};

// Activity codes, shared with python/main.py
enum Activity { ACT_STAND = 0, ACT_WALK = 1, ACT_RUN = 2, ACT_BIKE = 3 };

void set_activity(int activity) {
  switch (activity) {
    case ACT_STAND: matrix.draw(FRAME_STAND); break;
    case ACT_WALK:  matrix.draw(FRAME_WALK);  break;
    case ACT_RUN:   matrix.draw(FRAME_RUN);   break;
    case ACT_BIKE:  matrix.draw(FRAME_BIKE);  break;
  }
}

void scan_bus(const char *name, TwoWire &bus) {
  Monitor.print("Scanning ");
  Monitor.print(name);
  Monitor.println("...");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    bus.beginTransmission(addr);
    if (bus.endTransmission() == 0) {
      Monitor.print("  found device at 0x");
      Monitor.println(addr, HEX);
      found++;
    }
  }
  if (found == 0) {
    Monitor.print("  nothing responded on ");
    Monitor.println(name);
  }
}

void setup() {
  Monitor.begin(115200);
  Bridge.begin();
  Bridge.provide("set_activity", set_activity);

  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();

  pinMode(TRIG_PIN, OUTPUT);
  digitalWrite(TRIG_PIN, LOW);
  pinMode(ECHO_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(ECHO_PIN), echo_isr, CHANGE);

  Wire.begin();
  Wire1.begin();

  // Don't assume which I2C bus/address the sensor is wired to: try the
  // standard header pins (Wire) and the Qwiic/Modulino connector (Wire1),
  // at both possible addresses (AD0 low = 0x68, AD0 high = 0x69).
  TwoWire *buses[] = { &Wire, &Wire1 };
  const char *bus_names[] = { "Wire", "Wire1" };
  uint8_t addrs[] = { MPU6050_I2CADDR_DEFAULT, 0x69 };
  bool found = false;

  while (!found) {
    // Repeated every loop (not just once at boot) so it's visible even if
    // the Serial Monitor attaches well after the board powered up.
    scan_bus("Wire", Wire);
    scan_bus("Wire1", Wire1);

    for (uint8_t b = 0; b < 2 && !found; b++) {
      for (uint8_t a = 0; a < 2 && !found; a++) {
        if (mpu.begin(addrs[a], buses[b])) {
          found = true;
          Monitor.print("MPU6050 found on ");
          Monitor.print(bus_names[b]);
          Monitor.print(" at 0x");
          Monitor.println(addrs[a], HEX);
        }
      }
    }
    if (!found) {
      Monitor.println("MPU6050 not found on Wire/Wire1 (0x68/0x69), retrying in 3s...");
      delay(3000);
    }
  }

  // Wide ranges so footstrike/pedal peaks don't clip.
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  Monitor.println("MPU6050 ready");
}

void loop() {
  unsigned long currentMillis = millis();

  service_ultrasonic(currentMillis);

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // acceleration in m/s^2, angular velocity in rad/s, distance in cm
    // (the latest ultrasonic reading, repeated until a new one arrives;
    // -1 = sensor not responding, -2 = nothing in range)
    Bridge.notify("imu_sample",
                  a.acceleration.x, a.acceleration.y, a.acceleration.z,
                  g.gyro.x, g.gyro.y, g.gyro.z, distanceCm);
  }
}
