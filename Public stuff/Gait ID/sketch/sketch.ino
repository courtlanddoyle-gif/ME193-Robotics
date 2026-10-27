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

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // acceleration in m/s^2, angular velocity in rad/s
    Bridge.notify("imu_sample",
                  a.acceleration.x, a.acceleration.y, a.acceleration.z,
                  g.gyro.x, g.gyro.y, g.gyro.z);
  }
}
