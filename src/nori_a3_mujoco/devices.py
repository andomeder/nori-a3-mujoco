"""Accelerator selection for policy inference: Intel Arc (XPU) first-class, CPU fallback.

Same order and shape as the GPU helpers in andomeder/act-mujoco-manipulation
(src/utils/gpu_utils.py) and andomeder/decision-robot-control (policy_runner.py):

  torch:  CUDA if available -> Intel XPU (torch.xpu, native in PyTorch >= 2.5) -> CPU
  ONNX:   CUDAExecutionProvider -> OpenVINOExecutionProvider(device_type=GPU) -> CPU
          (ONNX Runtime has no XPU provider; an Arc runs ONNX through OpenVINO's GPU device)

Override with NORI_DEVICE=auto|cpu|xpu|xpu:1|cuda. Nothing here imports torch at module import,
so the sim and policy runner work without it.

Intel Extension for PyTorch (IPEX) is optional: set NORI_IPEX=1 and, if it imports, models
are passed through ipex.optimize(). Native torch.xpu does not need it.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

log = logging.getLogger(__name__)


def torch_device(prefer: str | None = None) -> Any:
    """The torch.device to run a policy on. Raises ImportError if torch isn't installed."""
    import torch

    want = (prefer or os.environ.get("NORI_DEVICE", "")).strip().lower()
    if want and want != "auto":
        kind = want.split(":")[0]
        if kind == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            log.warning("NORI_DEVICE=%s but torch.xpu is not available; using CPU", want)
            return torch.device("cpu")
        if kind == "cuda" and not torch.cuda.is_available():
            log.warning("NORI_DEVICE=%s but CUDA is not available; using CPU", want)
            return torch.device("cpu")
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def synchronize(device: Any) -> None:
    """Block until queued kernels finish -- needed for honest latency numbers on GPU."""
    import torch

    kind = getattr(device, "type", str(device))
    if kind == "xpu":
        torch.xpu.synchronize()
    elif kind == "cuda":
        torch.cuda.synchronize()


def maybe_ipex_optimize(model: Any, device: Any, dtype: Any = None) -> Any:
    """ipex.optimize(model) when NORI_IPEX=1, the device is XPU and IPEX imports; else model."""
    if os.environ.get("NORI_IPEX") != "1" or getattr(device, "type", "") != "xpu":
        return model
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore
    except Exception as exc:  # pragma: no cover - optional
        log.warning("NORI_IPEX=1 but IPEX failed to import (%s); continuing without it", exc)
        return model
    return ipex.optimize(model.eval(), dtype=dtype) if dtype else ipex.optimize(model.eval())


def onnx_providers() -> list[Any]:
    """ONNX Runtime providers in preference order: CUDA, OpenVINO GPU (Intel Arc), CPU."""
    import onnxruntime as ort  # type: ignore

    have = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in have:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "OpenVINOExecutionProvider" in have:
        return [("OpenVINOExecutionProvider", {"device_type": "GPU"}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def describe() -> dict[str, Any]:
    """What this machine offers, for `make gpu-info` / bug reports."""
    info: dict[str, Any] = {"NORI_DEVICE": os.environ.get("NORI_DEVICE"),
                            "MUJOCO_GL": os.environ.get("MUJOCO_GL")}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
        info["xpu"] = xpu
        if xpu:
            info["xpu_devices"] = [torch.xpu.get_device_name(i)
                                   for i in range(torch.xpu.device_count())]
        info["selected"] = str(torch_device())
    except ImportError:
        info["torch"] = None
    try:
        import onnxruntime as ort  # type: ignore

        info["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime_providers"] = None
    try:
        out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5).stdout
        info["gpus"] = [ln for ln in out.splitlines()
                        if "VGA" in ln or "Display" in ln or "3D controller" in ln]
    except Exception:
        info["gpus"] = None
    return info


if __name__ == "__main__":
    import json

    print(json.dumps(describe(), indent=2, default=str))
