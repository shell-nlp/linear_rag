from __future__ import annotations

from typing import Sequence

import numpy as np


def normalize_vector(vector: Sequence[float]) -> list[float]:
    """将模型返回的向量转换为普通浮点列表。"""

    array = np.asarray(vector, dtype=float)
    if array.ndim > 1:
        array = array.reshape(-1)
    return array.tolist()


def cosine_similarity(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float:
    """计算两个向量的余弦相似度，空向量返回零分。"""

    left_array = np.asarray(left, dtype=float).reshape(-1)
    right_array = np.asarray(right, dtype=float).reshape(-1)
    if left_array.size == 0 or right_array.size == 0:
        return 0.0
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)
