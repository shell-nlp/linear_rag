import re
import logging

from elasticsearch import helpers

logger = logging.getLogger(__name__)

class Customize_Elastic():
    def __init__(self, es_client, write_queue=None):
        self.es = es_client
        self.write_queue = write_queue

    def create_index_if_not_exists(self, vector_dim,index_name):
        """
        创建索引并指定 Mapping（防止自动创建导致向量字段类型错误）
        :param vector_dim: 向量维度，例如 OpenAI 是 1536，m3e 是 768
        """
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

        if self.write_queue is not None:
            return self.write_queue.submit(
                "create_index",
                {
                    "index_name": index_name,
                    "body": mapping,
                    "only_if_missing": True,
                },
            )

        return self._create_index_local(
            index_name=index_name,
            body=mapping,
            only_if_missing=True,
        )

    def _create_index_local(self, index_name, body, only_if_missing=True):
        """
        本地直接创建索引，不经过队列。
        """
        if self.es.indices.exists(index=index_name):
            if only_if_missing:
                return {"created": False, "exists": True, "index_name": index_name}
            raise ValueError(f"索引 {index_name} 已存在")
        try:
            self.es.indices.create(index=index_name, body=body)
            logger.info(f"索引 {index_name} 创建成功")
            return {"created": True, "index_name": index_name}
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            if only_if_missing and self.es.indices.exists(index=index_name):
                return {"created": False, "exists": True, "index_name": index_name}
            raise

    def delete_index(self, index_name, ignore_unavailable=True):
        if self.write_queue is not None:
            return self.write_queue.submit(
                "delete_index",
                {
                    "index_name": index_name,
                    "ignore_unavailable": ignore_unavailable,
                },
            )
        return self._delete_index_local(
            index_name=index_name,
            ignore_unavailable=ignore_unavailable,
        )

    def _delete_index_local(self, index_name, ignore_unavailable=True):
        response = self.es.indices.delete(
            index=index_name,
            ignore_unavailable=ignore_unavailable,
        )
        return {"acknowledged": response.get("acknowledged", True), "index_name": index_name}

    def save_batch(self, hash_ids, doc_infos, embeddings, index_name):
        """
        批量保存文档，只插入不存在的文档（相同ID会被跳过）
        
        :param hash_ids: ID 列表
        :param doc_infos: 包含 text 和 元数据 的字典列表
        :param embeddings: 向量列表
        """
        if self.write_queue is not None:
            normalized_embeddings = [
                vector.tolist() if hasattr(vector, "tolist") else vector
                for vector in embeddings
            ]
            return self.write_queue.submit(
                "save_batch",
                {
                    "hash_ids": hash_ids,
                    "doc_infos": doc_infos,
                    "embeddings": normalized_embeddings,
                    "index_name": index_name,
                },
            )

        return self._save_batch_local(hash_ids, doc_infos, embeddings, index_name)

    def _save_batch_local(self, hash_ids, doc_infos, embeddings, index_name):
        if not hash_ids:
            return {"success": 0, "failed": 0}

        vector_dim = len(embeddings[0]) if len(embeddings) > 0 else 1024
        self._create_index_local(
            index_name=index_name,
            body={
                "mappings": {
                    "properties": {
                        "hash_id": {"type": "keyword"},
                        "text": {"type": "text"},
                        "type": {"type": "keyword"},
                        "vector": {
                            "type": "dense_vector",
                            "dims": vector_dim,
                            "index": True,
                            "similarity": "cosine",
                        },
                    }
                }
            },
            only_if_missing=True,
        )

        pattern = r"^([^-]+)"
        match = re.match(pattern, hash_ids[0])
        node_type = match.group(1) if match else "Unknown"

        actions = []

        for h_id, doc_info, vector in zip(hash_ids, doc_infos, embeddings):
            vector = vector.tolist() if hasattr(vector, "tolist") else vector
            source_data = {
                "hash_id": h_id,
                "text": doc_info.get("text"),
                "vector": vector,
                "type": node_type,
            }

            meta_keys = ["file_name", "file_id", "pages_number", "segment_id", "ori_text", "content_table", "content_image", "file_path", "bucket_name"]

            metadata_obj = {
                "type": node_type
            }
            for key in meta_keys:
                value = doc_info.get(key)
                source_data[key] = value
                metadata_obj[key] = value

            source_data["metadata"] = metadata_obj

            action = {
                "_index": index_name,
                "_id": h_id,
                "_op_type": "create",
                "_source": source_data
            }
            actions.append(action)

        try:
            success, failed = helpers.bulk(
                self.es,
                actions,
                stats_only=True,
                refresh=True,
                raise_on_error=False,
            )
            print(f"ES批量插入完成: 成功 {success} 条, 跳过已存在文档 {failed} 条")
            return {"success": success, "failed": failed}
        except Exception as e:
            print(f"ES批量插入异常: {e}")
            raise

    def delete_by_query(self, index_name, body, refresh=True):
        if self.write_queue is not None:
            return self.write_queue.submit(
                "delete_by_query",
                {
                    "index_name": index_name,
                    "body": body,
                    "refresh": refresh,
                },
            )
        return self._delete_by_query_local(
            index_name=index_name,
            body=body,
            refresh=refresh,
        )

    def _delete_by_query_local(self, index_name, body, refresh=True):
        response = self.es.delete_by_query(
            index=index_name,
            body=body,
            refresh=refresh,
        )
        return {"deleted": response.get("deleted", 0), "index_name": index_name}
        
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
            "_source": ["hash_id","text", "file_name", "file_id", "pages_number", "segment_id", "ori_text", "content_table", "content_image","file_path","bucket_name"]
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
                response = self.delete_by_query(
                    index_name=index_name,
                    body={"query": {"term": {"type.keyword": self.namespace}}},
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
