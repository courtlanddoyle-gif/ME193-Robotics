import asyncio
import threading
import concurrent.futures
import legoeducation as le
import legoeducation.background_worker as bw
import legoeducation.basic_device as bd
import lego_compat
lego_compat.apply()   # read Color Sensors on the updated firmware
from pyscript import document, window
from pyscript.js_modules import ble
from pyscript.ffi import create_proxy
import json

import state

def _nowait_event_wait(self, timeout=None):
    return self._flag
threading.Event.wait = _nowait_event_wait

def _nowait_future_result(self, timeout=None):
    if self._state == 'FINISHED':
        return self._Future__get_result()
    return None
concurrent.futures.Future.result = _nowait_future_result

_worker_started = False

def _wasm_start_thread(self):
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    self.loop = asyncio.get_event_loop()
    self.loop_ready.set()
    asyncio.ensure_future(_wasm_worker_loop(self))

def _wasm_put_request(self, request):
    asyncio.ensure_future(self.async_put_request(request))

bw.Worker.start_thread = _wasm_start_thread
bw.Worker.put_request  = _wasm_put_request

async def _wasm_worker_loop(worker):
    worker._js_ble_registry = {}
    while True:
        try:
            req = await worker.request_queue.get()
            if req is None:
                break
            topic = req.get('topic')
            if topic == 'send':
                device  = req.get('msg')
                message = req.get('msg2')
                js_ble  = worker._js_ble_registry.get(id(device))
                if js_ble is not None and message is not None:
                    try:
                        await js_ble.send(list(message))
                    except Exception as e:
                        print(f"BLE send error: {e}")
            elif topic == 'connect':
                cb = req.get('msg3')
                if cb:
                    cb(True)
            elif topic == 'disconnect':
                device = req.get('msg')
                js_ble = worker._js_ble_registry.pop(id(device), None)
                if js_ble is not None:
                    js_ble.disconnect()
            elif topic == 'scan':
                cb = req.get('msg2')
                if cb:
                    cb([])
        except Exception as e:
            print(f"Worker loop error: {e}")

bd.my_worker.start_thread()

def flush_pending_commands():
    """Drop any not-yet-sent BLE requests.

    The run loop enqueues a 'send' request every tick (main.py's 50ms
    loop_network) but _wasm_worker_loop sends them strictly one at a
    time, awaiting each BLE write. If BLE round-trips run slower than
    the tick rate, motor-speed updates pile up in request_queue faster
    than they drain. Stop's motor_stop() request would otherwise land
    at the tail of that backlog, so the robot keeps executing stale
    speed commands for a while after Stop is pressed. Call this right
    before issuing the stop command so it isn't stuck behind them.
    """
    q = bd.my_worker.request_queue
    while not q.empty():
        try:
            q.get_nowait()
        except Exception:
            break

_DEVICE_MAP = {
    "Double Motor": le.DoubleMotor,
    "Single Motor": le.SingleMotor,
    "Color Sensor": le.ColorSensor,
    "Controller":   le.Controller,
}

def _dedupe_name(base_name: str, existing_names: list[str]) -> str:
    """Append ' (1)', ' (2)', etc. if base_name is already in use."""
    if base_name not in existing_names:
        return base_name
    i = 1
    while f"{base_name} ({i})" in existing_names:
        i += 1
    return f"{base_name} ({i})"

SERVICE_UUID = '0000fd02-0000-1000-8000-00805f9b34fb'
WRITE_UUID   = '0000fd02-0001-1000-8000-00805f9b34fb'
NOTIFY_UUID = '0000fd02-0002-1000-8000-00805f9b34fb'

_CARD_EMOJI = {
    le.LEGO_COLOR_RED: "🟥", le.LEGO_COLOR_ORANGE: "🟧", le.LEGO_COLOR_YELLOW: "🟨",
    le.LEGO_COLOR_GREEN: "🟩", le.LEGO_COLOR_BLUE: "🟦", le.LEGO_COLOR_PURPLE: "🟪",
    le.LEGO_COLOR_MAGENTA: "🩷", le.LEGO_COLOR_AZURE: "🩵",
}

def _classify_notification(data: bytes):
    """Device type name from one raw BLE message, or None if not a device notification."""
    import struct
    from legoeducation import rpc_message as rm
    if len(data) < 3 or data[0] != rm.DEVICE_NOTIFICATION:
        return None
    payload = data[1:]
    (n,) = struct.unpack('<H', payload[:2])
    note = rm.DeviceNotification(deviceDataByteLength=n, deviceData=payload[2:2 + n])
    items = rm.device_notification_parser(note)
    if any(isinstance(i, rm.ColorSensorNotification) for i in items):
        return "Color Sensor"
    if any(isinstance(i, rm.ControllerNotification) for i in items):
        return "Controller"
    motors = {i.motorBitMask for i in items if isinstance(i, rm.MotorNotification)}
    if len(motors) >= 2:
        return "Double Motor"
    if len(motors) == 1:
        return "Single Motor"
    return None

def ask_device_type():
    """Modal asking which device was picked; resolves to a type name or None."""
    fut = asyncio.get_event_loop().create_future()
    overlay = document.createElement("div")
    overlay.style.cssText = ("position:fixed;inset:0;background:rgba(0,0,0,.45);"
                             "display:flex;align-items:center;justify-content:center;z-index:9999")
    box = document.createElement("div")
    box.style.cssText = ("background:#fff;border-radius:12px;padding:20px;max-width:340px;"
                         "box-shadow:0 8px 30px rgba(0,0,0,.3);font-family:inherit")
    box.innerHTML = ("<div style='font-weight:600;margin-bottom:6px'>Which device did you pick?</div>"
                     "<div style='font-size:13px;color:#555;margin-bottom:12px'>"
                     "The browser did not report its name.</div>")
    proxies = []

    def choose(value):
        def handler(evt):
            overlay.remove()
            for p in proxies:
                p.destroy()
            if not fut.done():
                fut.set_result(value)
        p = create_proxy(handler)
        proxies.append(p)
        return p

    for label in _DEVICE_MAP:
        btn = document.createElement("button")
        btn.textContent = label
        btn.style.cssText = ("display:block;width:100%;margin:6px 0;padding:10px;border-radius:8px;"
                             "border:1px solid #ccc;background:#f6f6f6;cursor:pointer;font-size:14px")
        btn.addEventListener("click", choose(label))
        box.appendChild(btn)
    cancel = document.createElement("button")
    cancel.textContent = "Cancel"
    cancel.style.cssText = "margin-top:6px;border:none;background:none;color:#777;cursor:pointer"
    cancel.addEventListener("click", choose(None))
    box.appendChild(cancel)
    overlay.appendChild(box)
    document.body.appendChild(overlay)
    return fut

def show_reading(plot_id: str, val):
    """Show a node's current device value next to its device dropdown.

    The node's reading element shares its id suffix with its plot
    ("plot-in-3" -> "reading-in-3"), so the plot loops can report the value
    they are already plotting. val=None resets it to a dash.
    """
    el = document.getElementById(plot_id.replace("plot-", "reading-", 1))
    if not el:
        return
    if val is None:
        el.textContent = "—"
    elif isinstance(val, float) and not val.is_integer():
        el.textContent = f"{val:.2f}"
    else:
        el.textContent = str(int(val)) if isinstance(val, (int, float)) else str(val)

class Element:
    def __init__(self):
        self.myble = ble.BLEDevice.new()
        self.hub = None
        self.name = None
        self.plots = []
        self.plot_vars = []
        self.in_list = []
        self.out_list = []
        self.list_done = False
        self.state = None
        self.hardware_state = {"LightColor": le.LEGO_COLOR_WHITE, 
                               "LightPattern": le.LIGHT_PATTERN_BREATHE, 
                               "LightIntensity": 100, 
                               "BeepPattern": le.SOUND_PATTERN_BEEP_SINGLE, 
                               "BeepFrequency": 1,}

    async def _probe_type(self, timeout=4.0):
        """Connect, request notifications and identify the hub from them.

        Leaves the GATT link up on success (returns the type name); returns None
        on any failure.
        """
        found = []

        def on_data(data):
            try:
                kind = _classify_notification(bytes(data.to_py()))
            except Exception as e:
                print(f"Probe parse error: {e}")
                return
            if kind and not found:
                found.append(kind)

        proxy = create_proxy(on_data)
        try:
            self.myble.callback = proxy
            if not await self.myble.connect(SERVICE_UUID, WRITE_UUID, NOTIFY_UUID):
                return None
            from legoeducation.rpc_message import DeviceNotificationRequest
            request = list(DeviceNotificationRequest(100).Serialize())
            for tick in range(int(timeout / 0.05)):
                if found:
                    break
                if tick % 20 == 0:   # (re)send about once a second
                    await self.myble.send(request)
                await asyncio.sleep(0.05)
            print(f"Probe identified device as: {found[0] if found else None}")
            return found[0] if found else None
        except Exception as e:
            print(f"Probe failed: {e}")
            return None
        finally:
            self.myble.callback = None
            proxy.destroy()

    async def connect(self, existing_names: list[str] | None = None,
                      existing_ids: list[str] | None = None):
        print("Scanning...")
        try:
            await self.myble.scan()
        except Exception as e:
            print(f"Scan cancelled or failed: {e}")
            return

        self.ble_id = str(self.myble.device.id)
        if existing_ids and self.ble_id in existing_ids:
            print("That device is already connected; ignoring.")
            window.alert("That device is already connected.")
            self.myble = None
            return

        raw_name = self.myble.name
        print(f"Picked device name={raw_name!r}")

        probed = False
        if not str(raw_name).strip():
            # Chrome sometimes never reports the BLE name. Connect anyway and
            # work out the type from the notification stream; only ask the
            # user if that fails.
            raw_name = await self._probe_type()
            probed = bool(raw_name)
            if not raw_name:
                raw_name = await ask_device_type()
            if not raw_name:
                print("No device type chosen.")
                return

        base_name = str(raw_name)
        self.name = _dedupe_name(base_name, existing_names)

        hub_cls = None
        for key, cls in _DEVICE_MAP.items():
            if key in self.name:
                hub_cls = cls
                self.build_state()
                break

        if hub_cls is None:
            print(f"Unknown device type: {self.name}")
            return

        try:
            self.hub = hub_cls()

            _hub = self.hub
            self._notification_proxy = create_proxy(
                lambda data: asyncio.ensure_future(
                    _hub._device_callback(NOTIFY_UUID, bytes(data.to_py()))
                )
            )
            self.myble.callback = self._notification_proxy

            if probed:
                success = True   # probe already left the GATT link up
            else:
                success = await self.myble.connect(SERVICE_UUID, WRITE_UUID, NOTIFY_UUID)
            if not success:
                print("BLE connect() returned false")
                self.hub = None
                return

            print("BLE GATT connected, setting up hub...")
            
            self.hub.device = self.hub  # <-- THIS is the fix

            registry = getattr(bd.my_worker, '_js_ble_registry', None)
            if registry is not None:
                registry[id(self.hub)] = self.myble 
                print(f"Registered hub id={id(self.hub)}")
            else:
                print("WARNING: worker registry not found")

            self.hub.connected = True
            
            await asyncio.sleep(0)
            
            # Hub sometimes ignores the first request; re-send until the
            # first notification arrives (3 tries x 2 s).
            for attempt in range(1, 4):
                print(f"Sending device_notification_request (try {attempt})...")
                self.hub.device_notification_request(100, blocking=False)
                waited = 0.0
                while not self.hub._first_notification_ready.is_set() and waited < 2.0:
                    await asyncio.sleep(0.05)
                    waited += 0.05
                if self.hub._first_notification_ready.is_set():
                    break
        
            
            if self.hub._first_notification_ready.is_set():
                print("First notification received, sensor values ready.")
            else:
                print("Warning: timed out waiting for first notification, sensor values may still be nan.")
            
            await self._label_with_card_color(existing_names)
            print(f"Hub ready: {self.name}")

            self.hub.set_notification_callback(self.notif_callback)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Hub setup failed: {e}")
            self.hub = None
            
    async def _label_with_card_color(self, existing_names):
        """Prefix the name with a square for the hub's card color.

        Chrome-supplied names already carry one; names we derived ourselves
        (probe / dialog) do not, so read it from the hub's card notification.
        """
        if any(e in self.name for e in _CARD_EMOJI.values()) or "⬜" in self.name:
            return
        for _ in range(50):   # card info can trail the first notification (up to 2.5 s)
            card = getattr(self.hub, "scanned_card", None)
            if card is not None and card.color in _CARD_EMOJI:
                break
            await asyncio.sleep(0.05)
        card = getattr(self.hub, "scanned_card", None)
        color = getattr(card, "color", None)
        print(f"Card notification: color={color} serial={getattr(card, 'serial', None)}")
        emoji = _CARD_EMOJI.get(color)
        if emoji:
            kind = next(k for k in _DEVICE_MAP if k in self.name)
            self.name = _dedupe_name(f"{emoji}    {kind}", existing_names or [])

    async def disconnect(self):
        if self.hub and self.hub.connected:
            self.hub.disconnect()
            self.hub.connected = False
        await asyncio.sleep(0)
        self.hub = None
        self.myble = None   

    def build_state(self):
        if "Double Motor" in self.name:
            self.state = {
                "LeftPosition": None, "RightPosition": None, "LeftAngle": None, "RightAngle": None,
                "LeftSpeed": None, "RightSpeed": None, "RightPower": None, "LeftPower": None,
                "Yaw": None, "Pitch": None, "Roll": None,
                "AccelerationX": None, "AccelerationY": None, "AccelerationZ": None,
                "GyroX": None, "GyroY": None, "GyroZ": None,
            }
        elif "Single Motor" in self.name:
            self.state = {
                "Position": None, "Speed": None, "Angle": None,
            }
        elif "Controller" in self.name:
            self.state = {
                "LeftPercent": None, "RightPercent": None, 
                "LeftAngle": None, "RightAngle": None,
            }
        else:
            self.state = {
                "Reflection": None, "Color": None, 
                "Hue": None, "Saturation": None, "Value": None,
                "Red": None, "Green": None, "Blue": None,
            }

    def get_out_list(self):
        if "Double Motor" in self.name:
            return ["LeftSpeed", "RightSpeed", "BothSpeed", "LightIntensity", "BeepFrequency"]
        elif "Single Motor" in self.name:
            return ["Speed", "LightIntensity", "BeepFrequency"]
        elif "Color Sensor" in self.name:
            return ["LightIntensity", "BeepFrequency"] #["LightColor", "LightPattern", "LightIntensity", "BeepPattern", "BeepFrequency"]
        elif "Controller" in self.name:
            return ["LightIntensity", "BeepFrequency"] #["LightColor", "LightPattern", "LightIntensity", "BeepPattern", "BeepFrequency"]

    def notif_callback(self, data):                    
        parsed_items = le.device_notification_parser(data)
        for parsed_item in parsed_items:
    
            if isinstance(parsed_item, le.MotorNotification):
                if "Double Motor" in self.name:
                    side = "Left" if parsed_item.motorBitMask == le.MOTOR_BITS_LEFT else "Right"
                    self.state[side + "Position"] = parsed_item.position
                    self.state[side + "Angle"] = parsed_item.absolutePosition
                    self.state[side + "Speed"] = parsed_item.speed
                    self.state[side + "Power"] = parsed_item.power
                else:
                    self.state["Position"] = parsed_item.position
                    self.state["Angle"] = parsed_item.absolutePosition
                    self.state["Speed"] = parsed_item.speed
                    self.state["Power"] = parsed_item.power
    
            elif isinstance(parsed_item, le.ImuDeviceNotification):
                self.state["Yaw"] = parsed_item.yaw
                self.state["Pitch"] = parsed_item.pitch
                self.state["Roll"] = parsed_item.roll
                self.state["AccelerationX"] = parsed_item.accelerometerX
                self.state["AccelerationY"] = parsed_item.accelerometerY
                self.state["AccelerationZ"] = parsed_item.accelerometerZ
                self.state["GyroX"] = parsed_item.gyroscopeX
                self.state["GyroY"] = parsed_item.gyroscopeX
                self.state["GyroZ"] = parsed_item.gyroscopeX

            elif isinstance(parsed_item, le.ColorSensorNotification):
                self.state["Reflection"] = parsed_item.reflection
                self.state["Color"] = parsed_item.color
                self.state["Hue"] = parsed_item.hue
                self.state["Saturation"] = parsed_item.saturation
                self.state["Value"] = parsed_item.value
                self.state["Red"] = parsed_item.rawRed
                self.state["Green"] = parsed_item.rawGreen
                self.state["Blue"] = parsed_item.rawBlue

            elif isinstance(parsed_item, le.ControllerNotification):
                self.state["LeftPercent"] = parsed_item.leftPercent
                self.state["RightPercent"] = parsed_item.rightPercent
                self.state["LeftAngle"] = parsed_item.leftAngle
                self.state["RightAngle"] = parsed_item.rightAngle

        # LIVE PLOTTING STUFF
        for graph, var in zip(self.plots, self.plot_vars):
            val = 0
            if var == "BothSpeed":
                val = (self.state["LeftSpeed"] - self.state["RightSpeed"])/2
            elif var in self.hardware_state:
                try:
                    val = self.hardware_state[var]
                except Exception as e:
                    print("Cannot plot. Error: " + e)
            else: 
                val = self.state[var]
            show_reading(graph.id, val)
            update = graph.addPoints(1, [val])
            graph.updatePlot(update)

    def set_speed(self, speed):
        if "Single Motor" in self.name:
            #self.hub.motor_set_speed(speed, blocking=False)
            self.hub.motor_run(speed=speed, blocking=False)
        elif "Double Motor" in self.name:
            #self.hub.movement_set_speed(speed, blocking=False)
            self.hub.movement_move(speed=speed, blocking=False)
        else:
            print("cant set speed for " + self.name)

    def set_speedL(self, speed):
        if "Double Motor" in self.name:
            #self.hub.motor_set_speed(speed, motor=le.MOTOR_LEFT, blocking=False)
            self.hub.motor_run(speed=speed, motor=le.MOTOR_LEFT, blocking=False)
        else:
            print("cant set speedL for " + self.name)

    def set_speedR(self, speed):
        if "Double Motor" in self.name:
            #self.hub.motor_set_speed(speed, motor=le.MOTOR_RIGHT, blocking=False)
            self.hub.motor_run(speed=speed, motor=le.MOTOR_RIGHT, blocking=False)
        else:
            print("cant set speedL for " + self.name)
        
    def stop(self):
        if "Single Motor" in self.name:
            self.hub.motor_stop()
        elif "Double Motor" in self.name:
            self.hub.motor_stop(motor=le.MOTOR_BOTH)
        else:
            print("Cant stop " + self.name)

    def set_light(self, variable, value):
        self.hardware_state[variable] = value
        self.hub.light_color(self.hardware_state["LightColor"], intensity=self.hardware_state["LightIntensity"], blocking=False) #removed pattern=self.hardware_state["LightPattern"],
        
    def set_beep(self, variable, value):
        self.hardware_state[variable] = value
        freq = self.hardware_state["BeepFrequency"] if self.hardware_state["BeepFrequency"] != 0 else 1 # dont want it to beep at 0
        self.hub.beep(frequency=freq, blocking=False)


# ── Lookup / options-list helpers ────────────────────────────────────────────

def device_by_name(name: str) -> "Element | None":
    for d in state.devices:
        if d.name == name:
            return d
    return None

def get_device_options_html() -> str:
    opts = '<option class="dev-dropdown" value="">— select device —</option>'
    for device in state.devices:
        opts += f'<option value="{device.name}">{device.name}</option>'
    return opts

def _channel_options_html(source_fn) -> str:
    """Shared fallback-on-exception shape for get_in_channels_html/
    get_out_channels_html: try to pull the option keys from `source_fn`,
    fall back to a single placeholder option on any error (e.g. the device
    doesn't expose that list yet)."""
    try:
        keys = source_fn()
        return "".join(f'<option value="{key}">{key}</option>' for key in keys)
    except Exception as e:
        print("got error: " + e)
        return '<option value="">— value —</option>'

def get_in_channels_html(device: "Element | None" = None) -> str:
    if device is None:
        return '<option value="">— value —</option>'
    return _channel_options_html(lambda: device.state.keys())

def get_out_channels_html(device: "Element | None" = None) -> str:
    if device is None:
        return '<option value="">— value —</option>'
    return _channel_options_html(device.get_out_list)

# ── Device management (connect/disconnect UI) ────────────────────────────────

async def add_device_chip(dev: "Element"):
    dl = document.getElementById("device-list")

    name = dev.name
    chip = document.createElement("div")
    chip.className = "device-row"
    chip.id = f"chip-{name}"
    chip.innerHTML = (
        f'<div class="device-indicator"></div>'
        f'<span class="device-name">{name}</span>'
        f'<button class="btn-disconnect" title="Disconnect">'
        f'    <svg width="11" height="11" viewBox="0 0 24 24" fill="none"'
        f'         stroke="currentColor" stroke-width="2.5">'
        f'        <path d="M18 6L6 18M6 6l12 12"/>'
        f'    </svg>'
        f'</button>'
    )

    disc = chip.querySelector(".btn-disconnect")

    async def make_disc(dev):
        async def handler(evt):
            import sync
            chip_el = document.getElementById(f"chip-{dev.name}")
            if chip_el:
                chip_el.remove()
            await dev.disconnect()
            state.devices.remove(dev)
            sync.refresh_device_dropdowns()
        return create_proxy(handler)

    disc.addEventListener("click", await make_disc(dev))
    dl.appendChild(chip)
    import sync
    sync.refresh_device_dropdowns()

_connecting = False

async def create_new_device(evt=None):
    # One pairing at a time: overlapping scans race on names and the radio.
    global _connecting
    if _connecting:
        print("Already connecting; ignoring extra click.")
        return
    _connecting = True
    try:
        import sync
        new_dev = Element()
        existing_names = [d.name for d in state.devices]
        existing_ids = [getattr(d, "ble_id", None) for d in state.devices]
        await new_dev.connect(existing_names=existing_names, existing_ids=existing_ids)
        print("out of connect")
        if not new_dev.hub or not new_dev.hub.connected:
            return
        state.devices.append(new_dev)
        await add_device_chip(new_dev)
        sync.refresh_device_dropdowns()
    finally:
        _connecting = False

# ── Virtual controller (no-Bluetooth fallback) ───────────────────────────────
# Stands in for a real Controller+Motor pair on browsers without Web
# Bluetooth (Safari/Firefox, see main.py's check_browser_support()). Exposes
# LeftPercent/RightPercent input channels -- same names a real LEGO
# Controller reports -- driven by two <input type=range> sliders, and a
# "Speed" output channel -- same as a real Motor -- that spins a CSS
# animation instead of a physical motor. Because run_output() dispatches on
# fixed channel names (Device.py:~475), reusing "Speed" here means it needs
# no changes at all: it just calls this class's set_speed() like it would
# any real device's.
class VirtualDevice:
    def __init__(self, name="Virtual Controller"):
        self.name = name
        # "Speed" doubles as this device's own telemetry (like a real
        # motor's reported speed) and its output-write target -- same dual
        # role Element.state plays for a real device.
        self.state = {"LeftPercent": 0.0, "RightPercent": 0.0, "Speed": 0}
        self.plots = []
        self.plot_vars = []

    def get_out_list(self):
        return ["Speed"]

    def set_speed(self, value):
        _set_virtual_motor_speed(self, value)

    def stop(self):
        _set_virtual_motor_speed(self, 0)

def _set_virtual_motor_speed(dev: "VirtualDevice", value):
    # value arrives already clamped to [-100, 100] by run_output()'s "Speed"
    # branch (or is exactly 0, from stop()).
    dev.state["Speed"] = value
    # Output nodes attach their live plot to dev.plots/plot_vars exactly
    # like Input nodes do (bindings.py's on_output_device_change/
    # on_output_channel_change both pass attach_plot=True) -- a real
    # motor's BLE notification feeds that plot via Element.notif_callback's
    # "LIVE PLOTTING STUFF" loop; this is this device's equivalent push.
    _push_virtual_plot_points(dev)

    label = document.getElementById("virtual-motor-value")
    if label:
        label.textContent = str(value)

    blade = document.getElementById("virtual-motor-blade")
    if not blade:
        return
    if value == 0:
        blade.style.animationPlayState = "paused"
        return
    # Matches the real hardware's scale, where a "speed" of 1 is roughly
    # 1 RPM: duration (seconds/revolution) = 60 / RPM.
    duration = 60.0 / abs(value)
    blade.style.animationPlayState = "running"
    blade.style.animationDuration = f"{duration}s"
    blade.style.animationDirection = "reverse" if value < 0 else "normal"

def _push_virtual_plot_points(dev: "VirtualDevice"):
    """Mirrors the "LIVE PLOTTING STUFF" loop in Element.notif_callback --
    a real device's BLE notification pushes a fresh point into every
    live-plot attached to it (dev.plots/dev.plot_vars, wired up by
    bindings.py whenever an Input node picks this device+channel); a slider
    move is this device's equivalent of "new data just arrived"."""
    for graph, var in zip(dev.plots, dev.plot_vars):
        if var not in dev.state:
            continue
        show_reading(graph.id, dev.state[var])
        update = graph.addPoints(1, [dev.state[var]])
        graph.updatePlot(update)

_VIRTUAL_SLIDERS = (
    ("virtual-slider-left",  "virtual-slider-left-value",  "LeftPercent"),
    ("virtual-slider-right", "virtual-slider-right-value", "RightPercent"),
)

# elem_id -> in-flight spring-back Task, so grabbing a slider again mid-animation
# cancels the old return instead of fighting it.
_virtual_slider_anim_tasks: dict[str, "asyncio.Task"] = {}

def _cancel_virtual_slider_anim(elem_id: str):
    task = _virtual_slider_anim_tasks.pop(elem_id, None)
    if task and not task.done():
        task.cancel()

def _apply_virtual_slider_value(elem_id, value_label_id, dev, channel, val):
    slider = document.getElementById(elem_id)
    if slider:
        slider.value = str(val)
    dev.state[channel] = val
    label = document.getElementById(value_label_id)
    if label:
        label.textContent = str(int(round(val)))
    _push_virtual_plot_points(dev)

async def _spring_virtual_slider_to_zero(elem_id, value_label_id, dev, channel, start_value):
    """Physical joystick sliders are spring-loaded and snap back to center
    the instant you let go; this eases the virtual slider back to 0 the same
    way over a quick ~250ms instead of leaving it wherever it was dropped."""
    try:
        steps, duration = 15, 0.25
        for i in range(1, steps + 1):
            eased = 1 - (1 - i / steps) ** 3   # ease-out cubic
            _apply_virtual_slider_value(elem_id, value_label_id, dev, channel, start_value * (1 - eased))
            await asyncio.sleep(duration / steps)
    except asyncio.CancelledError:
        pass
    finally:
        # Only clear our own registration -- if we got cancelled and a new
        # spring-back task already replaced us in the dict, leave it alone.
        if _virtual_slider_anim_tasks.get(elem_id) is asyncio.current_task():
            _virtual_slider_anim_tasks.pop(elem_id, None)

def _bind_virtual_slider(elem_id: str, value_label_id: str, dev: "VirtualDevice", channel: str):
    slider = document.getElementById(elem_id)
    if not slider:
        return

    def on_input(evt):
        _cancel_virtual_slider_anim(elem_id)
        _apply_virtual_slider_value(elem_id, value_label_id, dev, channel, float(evt.target.value))
    slider.addEventListener("input", create_proxy(on_input))

    def on_release(evt):
        current = float(slider.value)
        if current == 0:
            return
        _virtual_slider_anim_tasks[elem_id] = asyncio.ensure_future(
            _spring_virtual_slider_to_zero(elem_id, value_label_id, dev, channel, current))
    slider.addEventListener("change", create_proxy(on_release))

def _zero_virtual_sliders(dev: "VirtualDevice"):
    for elem_id, value_label_id, channel in _VIRTUAL_SLIDERS:
        _cancel_virtual_slider_anim(elem_id)
        _apply_virtual_slider_value(elem_id, value_label_id, dev, channel, 0)

def enable_virtual_controller():
    """Called once at boot when navigator.bluetooth is missing: hides the
    (non-functional) Connect device button, shows the slider/motor panel in
    its place, and registers the virtual device so it appears in every
    Input/Output node's device dropdown immediately -- no "connect" step,
    since there's nothing to connect to."""
    device_list = document.getElementById("device-list")
    if device_list:
        device_list.classList.add("hidden")
    panel = document.getElementById("virtual-controller-panel")
    if panel:
        panel.classList.remove("hidden")

    dev = VirtualDevice()
    state.devices.append(dev)
    for elem_id, value_label_id, channel in _VIRTUAL_SLIDERS:
        _bind_virtual_slider(elem_id, value_label_id, dev, channel)

    zero_btn = document.getElementById("zero-virtual-sliders-btn")
    if zero_btn:
        zero_btn.addEventListener("click", create_proxy(lambda evt: _zero_virtual_sliders(dev)))

    import sync
    sync.refresh_device_dropdowns()

# ── Output routing ───────────────────────────────────────────────────────────

def run_output(variable, dev_name, value):
    device = device_by_name(dev_name)
    if not device:
        return
    if "Speed" in variable:
        if value > 100:
            value = 100
        elif value < -100:
            value = -100
    if variable == "Speed":
        device.set_speed(value)
    elif variable == "LeftSpeed":
        device.set_speedL(value)
    elif variable == "RightSpeed":
        device.set_speedR(value)
    elif variable == "BothSpeed":
        device.set_speed(value)
    elif variable == "LightColor":
        device.set_light(variable, value)
    elif variable == "LightPattern":
        device.set_light(variable, value)
    elif variable == "LightIntensity":
        if value > 100:
            value = 100
        elif value < 0:
            value = 0
        device.set_light(variable, value)
    elif variable == "BeepPattern":
        device.set_beep(variable, value)
    elif variable == "BeepFrequency":
        if value < 0:
            value = 0
        elif value > 2700:
            value = 2700
        device.set_beep(variable, value)
    else:
        print("Cannot set " + str(variable))
