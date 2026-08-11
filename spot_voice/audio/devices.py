"""Finding the microphone.

The Hollyland Lark M1 receiver shows up on Windows as a generic "USB Audio
Device", and its index moves between reboots, so the device is selected by a
case-insensitive **substring** of its name (``MIC_DEVICE_NAME``) rather than by
index. Every input device is printed at startup so the right substring is easy
to pick.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputDevice:
    """One audio input the OS is offering."""

    index: int
    name: str
    channels: int
    default_samplerate: float
    is_default: bool

    def describe(self) -> str:
        marker = " (system default)" if self.is_default else ""
        return (
            f"[{self.index}] {self.name}{marker} - "
            f"{self.channels} ch @ {self.default_samplerate:.0f} Hz"
        )


class MicrophoneNotFound(RuntimeError):
    """Raised when ``MIC_DEVICE_NAME`` matches nothing."""


def list_input_devices() -> list[InputDevice]:
    """Return every device that can record.

    Returns an empty list when ``sounddevice`` (or PortAudio) is unavailable,
    so text mode still runs on a machine with no audio stack.
    """
    try:
        import sounddevice
    except Exception as exc:  # pragma: no cover - depends on the host
        LOGGER.warning("sounddevice unavailable: %s", exc)
        return []

    try:
        default_input = sounddevice.default.device[0]
    except Exception:  # pragma: no cover
        default_input = None

    devices: list[InputDevice] = []
    for index, info in enumerate(sounddevice.query_devices()):
        if int(info.get("max_input_channels", 0)) <= 0:
            continue
        devices.append(
            InputDevice(
                index=index,
                name=str(info.get("name", f"device {index}")),
                channels=int(info["max_input_channels"]),
                default_samplerate=float(info.get("default_samplerate", 0.0)),
                is_default=(index == default_input),
            )
        )
    return devices


def select_input_device(name_fragment: str, devices: list[InputDevice] | None = None):
    """Resolve ``MIC_DEVICE_NAME`` to a device index.

    Args:
        name_fragment: Case-insensitive substring of the device name. Empty
            selects the system default.
        devices: Device list to search; queried from the OS when omitted.

    Returns:
        The device index, or ``None`` to mean "system default".

    Raises:
        MicrophoneNotFound: When a fragment was given but matched nothing.
    """
    if devices is None:
        devices = list_input_devices()

    if not name_fragment:
        return None

    needle = name_fragment.strip().lower()
    matches = [device for device in devices if needle in device.name.lower()]
    if not matches:
        available = "\n".join(f"  {device.describe()}" for device in devices) or "  (none)"
        raise MicrophoneNotFound(
            f"No input device matching {name_fragment!r}. Available inputs:\n{available}"
        )
    if len(matches) > 1:
        LOGGER.warning(
            "MIC_DEVICE_NAME %r matched %d devices; using %s",
            name_fragment,
            len(matches),
            matches[0].name,
        )
    return matches[0].index
