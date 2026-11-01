"""Safe loading for project checkpoints containing NumPy RNG state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> Any:
    """Load a project checkpoint with PyTorch's restricted unpickler.

    Project checkpoints store NumPy's RNG state as an ndarray. Allow only the
    NumPy reconstruction types required for that state, keeping all other
    checkpoint objects subject to ``weights_only=True`` restrictions.
    """
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:  # NumPy 1.x
        numpy_core = np.core

    safe_globals = [
        numpy_core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ]

    with torch.serialization.safe_globals(safe_globals):
        return torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
