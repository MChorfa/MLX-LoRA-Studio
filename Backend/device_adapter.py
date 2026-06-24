#!/usr/bin/env python3
"""Hexagonal device adapter for PyTorch references that hardcode their device.

Many `trust_remote_code` models (e.g. baidu/Unlimited-OCR) pin inference to CUDA:
`input_ids.cuda()`, bf16 casts, `device="cuda"`. That makes them unrunnable on
Apple Silicon even though the math is portable. This module is the ports-and-
adapters layer that decouples such code from a concrete accelerator:

    DevicePort  (the port)        -- device string, dtype, availability, install()
    CudaAdapter / MetalAdapter /  -- the adapters (one per backend)
    CpuAdapter / NpuAdapter

`adapter.install()` reroutes the hardcoded CUDA/bf16 calls onto the adapter's
backend, so the *same* reference runs on Metal (MPS), CUDA, CPU, or a torch-NPU
without touching the model code.

Honesty: CUDA, Metal (MPS) and CPU are first-class torch backends. NPU is real
only where torch exposes one (Ascend via `torch_npu`); Apple's Neural Engine is
reachable through CoreML, not torch, so that path is a documented stub, not a fake.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DevicePort(ABC):
    """The port: what any accelerator backend must provide."""

    name: str

    @property
    @abstractmethod
    def device(self) -> str:
        """The torch device string this adapter targets."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The dtype to run the model in on this backend."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend can actually run here and now."""

    def synchronize(self) -> None:  # pragma: no cover - backend hook
        pass

    def install(self) -> None:
        """Reroute hardcoded `.cuda()` / bf16 / autocast calls onto this backend."""
        _install_routing(self.device, self.dtype)


class CudaAdapter(DevicePort):
    name = "cuda"
    device = "cuda"
    dtype = torch.bfloat16

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def install(self) -> None:
        # Native path: the model's CUDA/bf16 assumptions already hold.
        return


class MetalAdapter(DevicePort):
    name = "metal"
    device = "mps"
    # MPS bf16 coverage (e.g. conv bias) is incomplete, so run float32 and unify
    # every bf16/half cast in the reference to it.
    dtype = torch.float32

    def is_available(self) -> bool:
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    def synchronize(self) -> None:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()


class CpuAdapter(DevicePort):
    name = "cpu"
    device = "cpu"
    dtype = torch.float32

    def is_available(self) -> bool:
        return True


class NpuAdapter(DevicePort):
    name = "npu"
    # Ascend torch-NPU exposes the "npu" device; the dtype mirrors CUDA's bf16.
    device = "npu"
    dtype = torch.bfloat16

    def is_available(self) -> bool:
        # Real only where a torch NPU backend is installed (e.g. Ascend torch_npu).
        npu = getattr(torch, "npu", None)
        return bool(npu) and getattr(npu, "is_available", lambda: False)()

    def install(self) -> None:
        if not self.is_available():
            raise RuntimeError(
                "No torch NPU backend found. Ascend NPUs need `torch_npu`; Apple's "
                "Neural Engine is reachable via CoreML, not torch (out of scope here)."
            )
        _install_routing(self.device, self.dtype)


_ADAPTERS = {a.name: a for a in (CudaAdapter(), MetalAdapter(), CpuAdapter(), NpuAdapter())}
_PREFERENCE = ("cuda", "npu", "metal", "cpu")


def select_adapter(preference: str | None = None) -> DevicePort:
    """Pick an adapter by name, else the best available in preference order."""
    if preference:
        adapter = _ADAPTERS.get(preference.lower())
        if adapter is None:
            raise ValueError(f"Unknown device adapter '{preference}'. Choose from {list(_ADAPTERS)}.")
        return adapter
    for name in _PREFERENCE:
        if _ADAPTERS[name].is_available():
            return _ADAPTERS[name]
    return _ADAPTERS["cpu"]


def _install_routing(device: str, dtype: torch.dtype) -> None:
    """Single source of truth: reroute device, dtype, and autocast for the backend.

    One `Tensor.to` override handles both `cuda`→target and bf16/half→dtype so the
    patches never clobber each other; `.cuda()`/`.bfloat16()` go through the same
    path; and `torch.autocast("cuda", ...)` becomes a no-op off-CUDA (it otherwise
    probes `torch.cuda.current_device()` and asserts).
    """
    target = torch.device(device)
    unify = dtype is not torch.bfloat16
    original_to = torch.Tensor.to

    def _remap(value):
        if _is_cuda_arg(value):
            return target
        if unify and value in (torch.bfloat16, torch.float16):
            return dtype
        return value

    def tensor_to(self, *args, **kwargs):
        args = tuple(_remap(a) for a in args)
        if _is_cuda_arg(kwargs.get("device")):
            kwargs["device"] = target
        if unify and kwargs.get("dtype") in (torch.bfloat16, torch.float16):
            kwargs["dtype"] = dtype
        return original_to(self, *args, **kwargs)

    torch.Tensor.to = tensor_to
    torch.Tensor.cuda = lambda self, *a, **k: tensor_to(self, target)
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(target)
    if unify:
        torch.Tensor.bfloat16 = lambda self: tensor_to(self, dtype)
        torch.Tensor.half = lambda self: tensor_to(self, dtype)
        torch.nn.Module.bfloat16 = lambda self: self.to(dtype)
        torch.nn.Module.half = lambda self: self.to(dtype)

    if target.type != "cuda":
        _patch_autocast_off_cuda()


def _patch_autocast_off_cuda() -> None:
    """Make `torch.autocast("cuda", ...)` a no-op when not running on CUDA."""
    original_autocast = torch.autocast

    class _NullAutocast:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def autocast(device_type="cuda", *args, **kwargs):
        if str(device_type).startswith("cuda"):
            return _NullAutocast()
        return original_autocast(device_type, *args, **kwargs)

    torch.autocast = autocast


def _is_cuda_arg(value) -> bool:
    if isinstance(value, str):
        return value.startswith("cuda")
    if isinstance(value, torch.device):
        return value.type == "cuda"
    return False
