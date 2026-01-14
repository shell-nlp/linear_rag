from elasticsearch import helpers
import re
import logging

logger = logging.getLogger(__name__)

class Customize_Elastic():
    def __init__(self, es_client):
        self.es = es_client

    def create_index_if_not_exists(self, vector_dim,index_name):
        """
        创建索引并指定 Mapping（防止自动创建导致向量字段类型错误）
        :param vector_dim: 向量维度，例如 OpenAI 是 1536，m3e 是 768
        """
        if self.es.indices.exists(index=index_name):
            return

        mapping = {
            "mappings": {
                "properties": {
                    "hash_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "type": {"type": "keyword"},
                    "vector": {
                        "type": "dense_vector",
                        "dims": vector_dim,  # 必须与你的 Embedding 模型维度一致
                        "index": True,       # 开启向量索引加速
                        "similarity": "cosine" # 或 l2_norm, dot_product
                    }
                }
            }
        }
        try:
            self.es.indices.create(index=index_name, body=mapping)
            logger.info(f"索引 {index_name} 创建成功")
        except Exception as e:
            logger.error(f"创建索引失败: {e}")

    def save_batch(self, hash_ids, doc_infos, embeddings, index_name):
        """
        :param hash_ids: ID 列表
        :param doc_infos: 包含 text 和 元数据 的字典列表 (来自 full_docs_map)
        :param embeddings: 向量列表
        """
        
        self.create_index_if_not_exists(vector_dim=len(embeddings[0]) if len(embeddings) > 0 else 1024, index_name=index_name)

        pattern = r"^([^-]+)"
        match = re.match(pattern, hash_ids[0])
        node_type = match.group(1) if match else "Unknown"
        
        actions = []
        
        # 遍历时，doc_info 是一个字典
        for h_id, doc_info, vector in zip(hash_ids, doc_infos, embeddings):
            
            # 基础 source 结构
            source_data = {
                "hash_id": h_id,
                "text": doc_info.get("text"), # 从字典里取 text
                "vector": vector,
                "type": node_type,
            }
            
            # 动态将所有元数据写入 source
            # 即使是 None 也会被写入为 null，方便后续知晓该字段存在
            meta_keys = ["file_name", "file_id", "pages_number", "segment_id", "ori_text", "content_table", "content_image"]
            for key in meta_keys:
                source_data[key] = doc_info.get(key)

            action = {
                "_index": index_name,
                "_source": source_data
            }
            actions.append(action)
        
        try:
            success, failed = helpers.bulk(self.es, actions, stats_only=True, refresh=True)
            print(f"ES批量插入完成: 成功 {success} 条, 失败 {failed} 条")
        except Exception as e:
            print(f"ES批量插入异常: {e}")
        
    def es_search(self, index_name, **kwargs):
        """
        支持普通查询和 KNN 查询的通用方法
        """
        if not self.es.indices.exists(index=index_name):
            logger.warning(f"索引 {index_name} 不存在")
            return {"hits": {"hits": []}}

        try:
            # 准备搜索参数
            search_params = {
                "index": index_name,
                # "size": kwargs.get("size", 10),
                "_source": kwargs.get("_source", True)
            }

            # 处理 KNN 参数 (ES 8.x 官方推荐写法)
            if "knn" in kwargs:
                search_params["knn"] = kwargs["knn"]
            
            # 处理普通 Query Body
            if "query_body" in kwargs:
                search_params["body"] = kwargs["query_body"]
            elif "query" in kwargs: # 兼容直接传 query 的情况
                search_params["body"] = {"query": kwargs["query"]}

            # 执行查询 (不使用 scroll，因为 KNN 不支持)
            response = self.es.search(**search_params)
            return response
        
        except Exception as e:
            print(f"ES search error: {e}")
            return {"hits": {"hits": []}}
        
    def _batch_fetch_passages(self, hash_ids, index_names):
        """
        批量获取 ES 数据。
        """
        query_body = {
            "query": {
                "terms": {
                    "hash_id": hash_ids 
                }
            },
            "size": len(hash_ids),
            "_source": ["hash_id","text", "file_name", "file_id", "pages_number", "segment_id", "ori_text", "content_table", "content_image"]
        }
        
        try:
            # 注意：这里直接使用原生的 self.es.search，绕过你之前写的那个带 scroll 的 es_search
            response = self.es.search(index=index_names, body=query_body)
            
            hits = response.get("hits", {}).get("hits", [])
            print(f"DEBUG: ES query returned {len(hits)} hits for {len(hash_ids)} IDs") # 打印命中了多少条
            
            res_map = {}
            for hit in hits:
                source = hit.get("_source", {})
                # 如果 source 里拿不到 hash_id，尝试拿 ES 的内置 _id
                h_id = source.get("hash_id") or hit.get("_id")
                if h_id:
                    res_map[h_id] = source
            
            return res_map
        except Exception as e:
            print(f"DEBUG: ES batch fetch error: {e}")
            return {}
        
    def es_delete(self, index_name):
        """删除es知识库中单条数据"""
        # 注意：__init__ 中没有定义 self.namespace，这里可能运行会报错
        # 假设你是想用 index_name 或者传入参数
        try:
            # 同样建议先检查索引是否存在
            if not self.es.indices.exists(index=index_name):
                return 0

            # 示例逻辑
            if hasattr(self, 'namespace') and self.namespace:
                response = self.es.delete_by_query(
                    index=index_name,
                    body={"query": {"term": {"type.keyword": self.namespace}}}, # 注意：delete_by_query 参数名通常是 body
                )
            else:
                 # 如果没有 namespace，这里原逻辑是抛错，这里保留原样
                raise ValueError("必须提供筛选条件")
                
            deleted_count = response.get('deleted', 0)
            logger.info(f"已删除 {deleted_count} 条文档")
            return deleted_count
        except Exception as e:
            logger.error(f"删除失败: {str(e)}")
            return 0