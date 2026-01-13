from abc import ABC, abstractmethod
from typing import Any, Dict


class GraphBase:
    """图数据库的基类"""

    @abstractmethod
    def add_node(self, node):
        raise NotImplementedError

    @abstractmethod
    def add_edge(self, from_node, to_node):
        raise NotImplementedError

    @abstractmethod
    def get_neighbors(self, node):
        raise NotImplementedError
