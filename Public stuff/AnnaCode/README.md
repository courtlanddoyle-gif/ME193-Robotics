# AnnaCode

Two browser pages for building small neural networks and connecting them to
LEGO Education hardware (motors, color sensor, controller). Nothing to
install: Python runs in the page via [PyScript](https://pyscript.net).

| Page | What it does | Open |
|---|---|---|
| **Builder** (`index.html`) | Wire sensors into a network by hand and set the weights yourself. Outputs drive motors, lights and the beeper. | [Open](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/AnnaCode/) |
| **Trainer** (`trainer.html`) | The same tool plus a dataset and training: collect examples from your hardware, train the network, and watch the fit and loss plots. | [Open](https://chrisbuerginrogers.github.io/ME193-Robotics/Public%20stuff/AnnaCode/trainer.html) |

## What you need

- **Chrome or Edge.** Connecting to hardware uses Web Bluetooth, which Safari
  and Firefox don't have. (The trainer still works there with its on-screen
  virtual controller and motor.)
- A LEGO Education hub that is switched on. Turn off other apps connected to
  it first; a hub only talks to one at a time.
- The first load takes a while (the browser downloads Python and numpy).

## Using it

1. Click **Connect device**, pick the hub in the Bluetooth chooser, and click
   **Pair**. The hub beeps when it connects.
2. The device appears in the Devices list, labelled with its card color, for
   example `🟦    Single Motor`. Connect as many as you like.
3. Add inputs, layers and outputs. On each input or output pick a device and a
   channel (for example `Reflection` on a Color Sensor, or `Speed` on a motor).
   The small number beside the device dropdown is that channel's current value.
4. Press the **play** button (▶) to start the network driving the outputs, and the
   stop button (■) to stop it.
5. *Trainer only:* press **Collect** to record rows from the connected inputs and
   outputs into the training data, then **Step** (one epoch) or **Train 50 epochs**.
   **Randomize weights** starts the network over.

Picking a hub that is already connected shows an alert and does nothing.

## Troubleshooting

- **It connects only some of the time.** The page retries automatically
  (up to five attempts). Open the browser console (Cmd+Opt+J) to see
  `BLE connect attempt N/5 failed` lines.
- **A hub appears in the chooser with no name.** Chrome sometimes does not
  report the name. The page works out the device type from the data the hub
  sends, and falls back to asking you which type it is.
- **Nothing happens on load.** Hard-reload (Cmd+Shift+R). The page needs the
  internet to fetch PyScript and Plotly.
- **Hub won't connect at all.** Switch it off and on, and remove any stale
  entry for it in your computer's Bluetooth settings.

## For editing

Everything is plain files; there is no build step. To run a copy locally:

```
cd "Public stuff/AnnaCode"
python3 -m http.server
```

then open `http://localhost:8000/` (builder) or
`http://localhost:8000/trainer.html` (trainer) in Chrome. `localhost` is
allowed to use Bluetooth.

```
index.html, styles.css, main.py   Builder page and its entry point
Device.py                         Hub connection, device state, output routing
glue/ network/ misc_ui/ plot.py   Builder UI, network model, plotting
JS/ble.js                         Web Bluetooth wrapper (connect with retries)
pyscript.toml                     Builder config: packages and file map
trainer.html, trainer.toml        Trainer page and its config
trainer/                          Trainer sources (same layout, own copies)
```

Notes:

- **Flat Python imports.** PyScript copies each source file into one flat
  folder (`./glue/state.py` becomes `state.py`), so modules import each other
  as `import state`, not `from glue import state`. The mapping lives in the
  `[files]` table of `pyscript.toml` / `trainer.toml`. Add a new file there
  or it won't be found.
- **The two pages have separate copies of `Device.py`.** A fix to Bluetooth
  handling must be made in both `Device.py` and `trainer/Device.py`
  (`JS/ble.js` and `trainer/JS/ble.js` likewise).
- **`legoeducation` is pinned to 1.0.6** in both `.toml` files. Version 1.1.1
  replaced the worker the page patches for the browser, and breaks it.
- **Color Sensor firmware update.** Updated Color Sensors (RPC 2.3, the
  firmware that matches `legoeducation` 1.2.0) add a sensor-mode byte to each
  reading, which 1.0.6 can't parse. `lego_compat.py` (one copy per page)
  replaces the library's notification parser with one that reads both the
  new and the old layout; `Device.py` applies it at import. Readings from
  updated sensors also carry `colorSensorMode` (0 off, 1 color, 2 ambient).
- The builder page still includes the original PyScript analytics snippet;
  the trainer page does not.

## Where it came from

Adapted from the Tufts CEEO apps
[network-builder](https://tuftsceeo.pyscriptapps.com/network-builder/latest/)
and [network-trainer](https://tuftsceeo.pyscriptapps.com/network-trainer/latest/)
(source: [tuftsceeo/NeuralNetworkEducation](https://github.com/tuftsceeo/NeuralNetworkEducation)).
Changes here: more reliable Bluetooth pairing, hub type detection when Chrome
gives no name, card-color labels, an already-connected alert, live values next
to each device dropdown, and a trainer page without the lesson plans.
