"""Pick the device for local models.

GPUs only help when they are real accelerators: NVIDIA (CUDA) or Apple Silicon (MPS).
On Intel Macs PyTorch also exposes MPS through the discrete AMD/Intel GPU, but it is slower
than the CPU for these models and runs out of memory, so Intel Macs use the CPU.
"""

from __future__ import annotations

import platform


def pick_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"
