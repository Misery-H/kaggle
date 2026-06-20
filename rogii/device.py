from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceChoice:
    requested: str
    resolved: str
    cuda_available: bool
    cuda_usable: bool
    reason: str | None = None


def torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"installed": False, "error": str(exc)}

    cuda_available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    cuda_probe: dict[str, Any] = {"attempted": False}

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        cuda_probe = {"attempted": True}
        try:
            tensor = torch.tensor([1.0], device="cuda")
            cuda_probe.update({"ok": bool((tensor + 1).item() == 2.0)})
        except Exception as exc:
            cuda_probe.update({"ok": False, "error": str(exc)})

    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_usable": bool(cuda_probe.get("ok", False)),
        "cuda_probe": cuda_probe,
        "device_count": len(devices),
        "devices": devices,
    }


def choose_device(requested: str = "auto", torch_info: dict[str, Any] | None = None) -> DeviceChoice:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {requested}")

    torch_info = torch_info if torch_info is not None else torch_status()
    cuda_available = bool(torch_info.get("cuda_available"))
    cuda_usable = bool(torch_info.get("cuda_usable"))

    if requested == "cpu":
        return DeviceChoice(requested, "cpu", cuda_available, cuda_usable)
    if requested == "cuda":
        if cuda_usable:
            return DeviceChoice(requested, "cuda", cuda_available, cuda_usable)
        return DeviceChoice(requested, "unavailable", cuda_available, cuda_usable, "CUDA requested but not usable.")
    if cuda_usable:
        return DeviceChoice(requested, "cuda", cuda_available, cuda_usable)
    return DeviceChoice(requested, "cpu", cuda_available, cuda_usable, "CUDA not usable; falling back to CPU.")
