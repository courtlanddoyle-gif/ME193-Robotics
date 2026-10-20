"""Make legoeducation 1.0.6 read Color Sensors on the newer firmware.

The Color Sensor firmware update (RPC 2.3, matching legoeducation 1.2.0)
added one byte to the end of every Color Sensor notification: the sensor
mode (off / color detection / ambient light). legoeducation 1.0.6 still
expects the old 14-byte layout, so after each Color Sensor reading it starts
the next item one byte early. That byte is the mode (1 = color detection),
which it misreads as an IMU notification, and the whole notification fails
to parse: no readings, and hub-type detection fails.

We can't simply upgrade: 1.1.1 and later replace the background worker that
Device.py patches for the browser (see README). So this keeps 1.0.6 and swaps
in a notification parser that understands both layouts. It tries the new
15-byte layout first and falls back to the old 14-byte one, so sensors that
haven't been updated keep working too.

Call apply() once, right after importing legoeducation.
"""

import struct

import legoeducation as le
import legoeducation.basic_device as bd
from legoeducation import rpc_message as rm

COLOR_SENSOR_MODE_OFF = 0
COLOR_SENSOR_MODE_COLOR_DETECTION = 1
COLOR_SENSOR_MODE_AMBIENT_LIGHT = 2

_OLD_SIZE = struct.calcsize(rm.ColorSensorNotification.FormatString)  # 14
_NEW_SIZE = _OLD_SIZE + 1                                              # 15


class ColorSensorNotificationV2(rm.ColorSensorNotification):
    """Color Sensor reading that also carries the sensor mode.

    A subclass, so every existing isinstance(..., ColorSensorNotification)
    check (Device.py, the library's color translation) still matches.
    colorSensorMode is None for readings from older firmware.
    """

    def __init__(self, *fields, colorSensorMode=None):
        super().__init__(*fields)
        self.colorSensorMode = colorSensorMode

    @classmethod
    def Deserialize(cls, data, with_mode=True):
        base = rm.ColorSensorNotification.Deserialize(data)
        mode = struct.unpack_from("<b", data, _OLD_SIZE)[0] if with_mode else None
        return cls(base.color, base.reflection, base.rawRed, base.rawGreen,
                   base.rawBlue, base.hue, base.saturation, base.value,
                   colorSensorMode=mode)


_FIXED_TYPES = {
    rm.INFO_DEVICE_NOTIFICATION:  rm.InfoDeviceNotification,
    rm.IMU_DEVICE_NOTIFICATION:   rm.ImuDeviceNotification,
    rm.CARD_NOTIFICATION:         rm.CardNotification,
    rm.BUTTON_STATE_NOTIFICATION: rm.ButtonStateNotification,
    rm.MOTOR_NOTIFICATION:        rm.MotorNotification,
    rm.CONTROLLER_NOTIFICATION:   rm.ControllerNotification,
    rm.IMU_GESTURE_NOTIFICATION:  rm.ImuGestureNotification,
}


class _BadLayout(Exception):
    pass


def _parse(data, length, color_size, strict):
    """Parse the items in one device notification.

    strict=True raises _BadLayout unless the items exactly fill `length`
    bytes, which is how we tell which Color Sensor layout the hub is using.
    strict=False behaves like the library: stop quietly at anything unknown.
    """
    items = []
    pos = 0
    while pos < length:
        kind = data[pos]
        if kind == rm.COLOR_SENSOR_NOTIFICATION:
            size = color_size
            make = lambda b: ColorSensorNotificationV2.Deserialize(
                b, with_mode=(color_size == _NEW_SIZE))
        elif kind in _FIXED_TYPES:
            cls = _FIXED_TYPES[kind]
            size = struct.calcsize(cls.FormatString)
            make = cls.Deserialize
        else:
            if strict:
                raise _BadLayout(f"unknown item type {kind}")
            print(f"Unknown notification type: {kind}, Data={bytes(data)}")
            break

        end = pos + 1 + size
        if end > len(data):
            if strict:
                raise _BadLayout("item runs past the end of the data")
            break
        items.append(make(data[pos + 1:end]))
        pos = end

    if strict and pos != length:
        raise _BadLayout("items don't fill the notification")
    return items


_preferred = [_NEW_SIZE, _OLD_SIZE]   # whichever worked last is tried first


def device_notification_parser(notification_obj):
    """Drop-in replacement for rpc_message.device_notification_parser."""
    data = notification_obj.deviceData
    length = notification_obj.deviceDataByteLength
    for size in list(_preferred):
        try:
            items = _parse(data, length, size, strict=True)
        except (_BadLayout, struct.error):
            continue
        if size != _preferred[0]:
            _preferred.remove(size)
            _preferred.insert(0, size)
        return items
    return _parse(data, length, _preferred[0], strict=False)


_applied = False


def apply():
    """Install the parser everywhere legoeducation 1.0.6 looks it up."""
    global _applied
    if _applied:
        return
    rm.device_notification_parser = device_notification_parser
    bd.device_notification_parser = device_notification_parser
    bd._raw_device_notification_parser = device_notification_parser
    # le.device_notification_parser is a wrapper (it also translates color
    # numbers) that calls this module-level name.
    le._raw_device_notification_parser = device_notification_parser
    _applied = True
