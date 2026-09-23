"""搜索数据库适配器统一入口。"""

from src.adapters.search.elasticsearch_store import ElasticsearchSearchStore

__all__ = ["ElasticsearchSearchStore"]
