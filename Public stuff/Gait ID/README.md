# 🚶 Gait ID

An ankle-worn wearable that tells whether you're **standing, walking, running, or biking**, using an Arduino UNO Q and an MPU6050 IMU. It runs two non-neural-net models side by side, and a live web page on your phone shows both calls.

ME-193, Tufts, Fall 2026 — Courtland Doyle

## Hardware

| Part | Connection |
|---|---|
| Arduino UNO Q | Strapped to the ankle, on a USB battery pack |
| MPU6050 breakout | VCC → power, GND → GND, SDA/SCL → the UNO Q's SDA/SCL header pins (next to AREF, the `Wire` bus). AD0 and INT not connected. |

The sensor is strapped just above the ankle. **Wear it the same way every time.** Model 2's tilt features depend on which way the sensor faces.

## How it works

```
MPU6050 ──I2C, 50 Hz──► sketch.ino ──Bridge──► python/main.py ──► LED matrix + web page
                         (MCU side)             (Linux side)
```

1. **`sketch/sketch.ino`** reads the accelerometer (m/s²) and gyro (rad/s) 50 times a second and sends each sample to Python.
2. **`python/main.py`** keeps the last **2 seconds** (100 samples) of each channel. Every 0.2 s it computes features and runs both models.
3. **Gyro calibration:** the MPU6050's gyro has a built-in zero offset. For the first 2 s after the app starts, the sensor must sit still while the code measures that offset and then subtracts it from every sample.
4. The result shows on the **LED matrix** (Model 2) and on a **web page** at `http://<board IP>:7000` that polls `/api/gait` 5 times a second.

### Features (per 2 s window)

| Feature | What it measures |
|---|---|
| `axis_ptp` | Peak-to-peak acceleration on **one axis** (`az`): how hard the foot hits |
| `accel_mag_std`, `accel_mag_ptp` | Spread and peak-to-peak of the total acceleration: overall motion and impact size |
| `gyro_mag_mean`, `gyro_mag_max` | Average and peak rotation speed of the shin |
| `ax_mean`, `ay_mean`, `az_mean` | Average acceleration per axis ≈ **gravity direction**, i.e. which way the shin points |

### Model 1: one axis, peak acceleration only

```
axis_ptp < 2.0   → standing
axis_ptp < 6.5   → biking     (no footstrike)
axis_ptp < 14.0  → walking
otherwise        → running
```

### Model 2: all six IMU channels

```
accel_mag_std < 0.8                                → standing
ay_mean < -2.0                                     → biking   (shin tilted back)
gyro_mag_max < 4.2 and accel_mag_ptp < 17          → biking   (smooth pedal circle, no footstrike)
ay_mean > 1.7  or  accel_mag_ptp > 22              → running  (shin leans forward / hard impact)
otherwise                                          → walking
```

## Results

All thresholds were tuned on ~20 s recordings of each activity (in `data/`). Run `python data/score.py` to reproduce this:

| Recording | Model 1 | Model 2 |
|---|---|---|
| Standing | 95% | 100% |
| Walking in place | 85% | 100% |
| Jogging in place | 87% | 100% |
| Biking, lying on your back | 100% | 100% |
| Biking, sitting in a chair | 100% | 100% |

These are scored on the same data the thresholds were picked from, all from one person, so expect lower accuracy on someone new.

## Which model is better, and why

**Model 2.** Model 1 only knows how hard your foot hits along one axis. That's enough to spot standing (no hit) and biking (no footstrike), but walking and jogging overlap on that one number, so it mixes them up about 15% of the time.

Model 2 also knows **which way your shin points** and **how it rotates**:
- **Jogging** leans the shin forward, so gravity shifts onto the y axis (`ay_mean` ≈ 3.2 vs. 0.6 walking, no overlap).
- **Biking** tilts the shin back, whether you're sitting (`ay_mean` ≈ -3.9) or on your back (≈ -9.5). It also rotates in a smooth circle, so the gyro peak stays low (< 3.8 vs. > 5 when walking).

### What didn't work

The first Model 2 used a "quiet gyro" feature: the fraction of time the foot is nearly still, on the theory that walking has a planted-foot phase and pedaling doesn't. Real data proved it wrong. On an ankle mount, the **shin** keeps rotating over the planted foot, so the feature read ≈ 0 for walking, jogging, *and* biking. It was replaced with the tilt and gyro-peak features above.

## Running it

1. Open the app in **Arduino App Lab** (or copy this folder to `~/ArduinoApps/gait-id` on the board) and start it. Leave the sensor **flat and still** for the first few seconds.
2. On a phone on the same network as the board, open `http://<board IP>:7000`.
3. Optional: make it start on boot, and have the board host its own WiFi, so it works anywhere without a laptop:
   ```bash
   arduino-app-cli properties set default /home/arduino/ArduinoApps/gait-id
   nmcli connection add type wifi ifname wlan0 con-name gait-hotspot autoconnect yes \
     ssid "<network name>" 802-11-wireless.mode ap 802-11-wireless.band bg \
     ipv4.method shared ipv4.addresses 10.42.0.1/24 \
     wifi-sec.key-mgmt wpa-psk wifi-sec.psk "<password>" \
     connection.autoconnect-priority 10
   ```
   Then join that network on a phone and open `http://10.42.0.1:7000`.

`CALIBRATE_LOG = True` in `main.py` prints every feature 5 times a second. That's how the thresholds were tuned:
```bash
arduino-app-cli app logs /home/arduino/ArduinoApps/gait-id --follow --tail 5
```

## Files

| Path | What it is |
|---|---|
| `sketch/sketch.ino` | MCU side: reads the MPU6050, draws LED matrix glyphs |
| `python/main.py` | Linux side: windows, features, both models, web API |
| `assets/` | The live web page |
| `app.yaml` | Arduino App config (uses the `web_ui` brick) |
| `data/*.log`, `data/score.py` | Recordings used for tuning, and a script that scores both models on them |
