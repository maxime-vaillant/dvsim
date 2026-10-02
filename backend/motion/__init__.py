"""Learned human-like camera motion (see docs/superpowers/specs/2026-10-02-human-motion-design.md).

Importing this package never imports torch: only ``model``, ``train`` and ``style`` do, through ``require_torch``.
"""


def require_torch():
    """The torch module, or an ImportError that says how to get it."""
    try:
        import torch
    except ImportError:
        raise ImportError("learned human motion needs PyTorch: pip install -e .[torch]") from None
    return torch
