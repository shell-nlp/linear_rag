from __future__ import annotations

from typing import Sequence

import numpy as np


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """将模型返回的向量转换为普通浮点列表。"""

    array = np.asarray(vector, dtype=float)
    if array.ndim > 1:
        array = array.reshape(-1)
    return array.tolist()
