from typing import Any, Dict, List, Optional
from src.infra.neo4j.base import GraphBase
import uuid
from neo4j import GraphDatabase
from loguru import logger


class Neo4jGraph(GraphBase):
    """Neo4j图数据库的实现类"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        初始化 Neo4j 连接

        Args:
            uri: Neo4j 数据库 URI (如: bolt://localhost:7687)
            user: 用户名
            password: 密码
            database: 数据库名称，默认为 neo4j
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self._connect()
        
    def get_session(self):
        """暴露底层的 session 对象供外部使用"""
        if not self.driver:
            self._connect()
        return self.driver.session(database=self.database)
    
    def add_node(
        self, label: str, properties: Dict[str, Any] = None, node_id: str = None
    ) -> str:
        """
        添加节点

        Args:
            label: 节点标签,即实体培训
            properties: 节点属性
            node_id: 节点唯一标识符（可选）

        Returns:
            节点 ID
        """
        # TODO 暂未实现节点去重逻辑
        if properties is None:
            properties = {}

        # 如果提供了 node_id，则使用它作为唯一标识
        if node_id:
            properties["id"] = node_id

        parameters = {
            "node_id": node_id or self._generate_node_id(label),
            "properties": properties,
            "label": label,
        }

        query = f"""
        MERGE (n:{label} {{id: $node_id}})
        SET n += $properties 
        RETURN id(n) as node_internal_id, n.id  as node_id
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                record = result.single()
                logger.info(f" 成功添加节点: {label} - {record['node_id']}")
                return record["node_id"]
        except Exception as e:
            logger.error(f" 添加节点失败: {e}")
            raise
        
    def execute_query(self, cypher_query, parameters=None):
        """
        执行原生Cypher查询的通用方法。
        假设你的类中初始化了 self.driver (neo4j.GraphDatabase.driver)
        """
        if parameters is None:
            parameters = {}
        
        # 使用会话执行
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher_query, parameters)
            # 如果需要返回结果，可以在这里处理，写入操作通常不需要返回具体数据
            return result.consume()

    def delete_node(self, node_id: str) -> bool:
        """
        删除节点及其所有关系

        Args:
            node_id: 节点ID

        Returns:
            是否成功删除
        """
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        RETURN count(n) > 0 as deleted 
        """

        parameters = {"node_id": node_id}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                record = result.single()
                deleted = record["deleted"]
                if deleted:
                    logger.info(f" 成功删除节点: {node_id}")
                else:
                    logger.warning(f" 删除节点失败: 节点不存在")
                return deleted
        except Exception as e:
            logger.error(f" 删除节点失败: {e}")
            raise

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        根据节点ID获取节点信息

        Args:
            node_id: 节点ID

        Returns:
            节点信息字典或None
        """
        query = """
        MATCH (n {id: $node_id})
        RETURN n 
        """

        parameters = {"node_id": node_id}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                record = result.single()
                if record:
                    node = record["n"]
                    return {
                        "id": node.get("id"),
                        "labels": list(node.labels),
                        "properties": dict(node),
                    }
                return None
        except Exception as e:
            logger.error(f" 查询节点失败: {e}")
            raise

    def get_nodes_by_label(self, label: str) -> List[Dict[str, Any]]:
        """
        根据标签获取所有节点

        Args:
            label: 节点标签

        Returns:
            节点列表
        """
        query = f"""
        MATCH (n:{label})
        RETURN n 
        """

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query)
                nodes = []
                for record in result:
                    node = record["n"]
                    nodes.append(
                        {
                            "id": node.get("id"),
                            "labels": list(node.labels),
                            "properties": dict(node),
                        }
                    )
                return nodes
        except Exception as e:
            logger.error(f" 查询节点失败: {e}")
            raise

    def add_edge(
        self,
        from_node_id: str,
        to_node_id: str,
        relationship_type: str = "LINK",
        properties: Dict[str, Any] = None,
    ) -> bool:
        """
        添加边（关系）

        Args:
            from_node_id: 起始节点ID
            to_node_id: 目标节点ID
            relationship_type: 关系类型
            properties: 关系属性

        Returns:
            是否成功添加
        """
        if properties is None:
            properties = {}

        query = f"""
        MATCH (a {{id: $from_node_id}}), (b {{id: $to_node_id}})
        MERGE (a)-[r:{relationship_type}]->(b)
        SET r += $properties 
        RETURN count(r) > 0 as created
        """

        parameters = {
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "properties": properties,
        }

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                record = result.single()
                success = record["created"]
                if success:
                    logger.info(
                        f" 成功添加关系: {from_node_id} -[{relationship_type}]-> {to_node_id}"
                    )
                else:
                    logger.warning(f" 添加关系失败: 节点未找到")
                return success
        except Exception as e:
            logger.error(f" 添加关系失败: {e}")
            raise

    def delete_edge(
        self, from_node_id: str, to_node_id: str, relationship_type: str
    ) -> bool:
        """
        删除指定的关系

        Args:
            from_node_id: 起始节点ID
            to_node_id: 目标节点ID
            relationship_type: 关系类型

        Returns:
            是否成功删除
        """
        query = f"""
        MATCH (a {{id: $from_node_id}})-[r:{relationship_type}]->(b {{id: $to_node_id}})
        DELETE r
        RETURN count(r) > 0 as deleted
        """

        parameters = {"from_node_id": from_node_id, "to_node_id": to_node_id}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                record = result.single()
                deleted = record["deleted"]
                if deleted:
                    logger.info(
                        f" 成功删除关系: {from_node_id} -[{relationship_type}]-> {to_node_id}"
                    )
                else:
                    logger.warning(f" 删除关系失败: 关系不存在")
                return deleted
        except Exception as e:
            logger.error(f" 删除关系失败: {e}")
            raise

    def get_neighbors(
        self, node_id: str, relationship_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        获取节点的邻居节点

        Args:
            node_id: 节点ID
            relationship_type: 关系类型（可选）

        Returns:
            邻居节点列表
        """
        if relationship_type:
            query = f"""
            MATCH (n {{id: $node_id}})-[r:{relationship_type}]->(neighbor)
            RETURN neighbor, type(r) as relationship_type
            """
        else:
            query = """
            MATCH (n {id: $node_id})-[r]->(neighbor)
            RETURN neighbor, type(r) as relationship_type
            """

        parameters = {"node_id": node_id}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                neighbors = []
                for record in result:
                    neighbor = record["neighbor"]
                    neighbors.append(
                        {
                            "node": {
                                "id": neighbor.get("id"),
                                "labels": list(neighbor.labels),
                                "properties": dict(neighbor),
                            },
                            "relationship_type": record["relationship_type"],
                        }
                    )
                return neighbors
        except Exception as e:
            logger.error(f" 查询邻居节点失败: {e}")
            raise

    def _connect(self):
        """建立数据库连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            # 验证连接
            with self.driver.session(database=self.database) as session:
                session.run("RETURN  1")
            logger.info(f" 成功连接到 Neo4j 数据库: {self.uri}")
        except Exception as e:
            logger.error(f" 连接 Neo4j 失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info(" 已关闭 Neo4j 连接")

    def _generate_node_id(self, label: str) -> str:
        """生成节点ID"""

        return f"{label}_{str(uuid.uuid4())[:8]}"

    def run_cypher_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        执行自定义 Cypher 查询

        Args:
            query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        if parameters is None:
            parameters = {}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, parameters)
                records = []
                for record in result:
                    records.append(dict(record))
                return records
        except Exception as e:
            logger.error(f"Cypher  查询失败: {e}")
            raise
        
    def clear_database(self):
        """清空数据库（谨慎使用）"""
        query = "MATCH (n) DETACH DELETE n"
        try:
            with self.driver.session(database=self.database) as session:
                session.run(query)
            logger.info(" 数据库已清空")
        except Exception as e:
            logger.error(f" 清空数据库失败: {e}")
            raise



if __name__ == "__main__":
    graph = Neo4jGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="neo4j@2025",
        database="neo4j",
    )
    graph.clear_database()
    # try:
    #     # 添加节点
    #     person1_id = graph.add_node(
    #         "职工信息",
    #         {
    #             "name": "黄柄涵",
    #             "type":"entity",
    #             "file_id": "123",
    #             "file_name": "大模型研究院职工",
    #         },
    #         "entity-05f2fd8c6e36ccb78f4ba294f1c8121b",
    #     )

    #     person2_id = graph.add_node(
    #         "职工信息",
    #         {
    #             "name": "刘宇",
    #             "type":"entity",
    #             "file_id": "123",
    #             "file_name": "大模型研究院职工",
    #         },
    #         "entity-f6ae769f3937cdd44114ba1ca27c9e28",
    #     )
    #     person3_id = graph.add_node(
    #         "职工信息",
    #         {
    #             "name": "杨佳雯",
    #             "type":"entity",
    #             "file_id": "123",
    #             "file_name": "大模型研究院职工",
    #         },
    #         "entity-51eb87260674fe8fd0355fc9d7035592",
    #     )

    #     company_id = graph.add_node(
    #         "职工信息", 
    #         {
    #             "name": "大模型研究院职工包括：黄柄涵、刘宇、杨佳雯",
    #             "type":"entity",
    #             "file_id": "123",
    #             "file_name": "大模型研究院职工",
    #         }, 
    #         "passage-48c6c1876a32b65fa25c317be59a6502"
    #     )

    #     # 添加关系
    #     graph.add_edge(
    #         person1_id, company_id, "LINK", {"e_weight": "0.84"}
    #     )
    #     graph.add_edge(
    #         person2_id, company_id, "LINK", {"e_weight": "0.89"}
    #     )
    #     graph.add_edge(
    #         person3_id, company_id, "LINK", {"e_weight": "0.85"}
    #     )
        # 查询节点
        # print("=== 查询节点 ===")
        # node = graph.get_node(person1_id)
        # print(f"Alice节点: {node}")

        # # 查询标签下的所有节点
        # print("\n=== 查询所有人员 ===")
        # persons = graph.get_nodes_by_label("员工")
        # for person in persons:
        #     print(f"员工: {person}")

        # # 查询邻居节点
        # print("\n=== 查询Alice的邻居 ===")
        # neighbors = graph.get_neighbors(person1_id)
        # for neighbor in neighbors:
        #     print(
        #         f"邻居: {neighbor['node']['properties']['name']}, 关系: {neighbor['relationship_type']}"
        #     )

        # # 自定义查询
        # print("\n=== 自定义查询：查找年龄大于25的人 ===")
        # custom_query = """
        # MATCH (p:员工)
        # WHERE p.age  > $min_age 
        # RETURN p.name  as name, p.age  as age
        # ORDER BY p.age  DESC 
        # """
        # results = graph.run_cypher_query(custom_query, {"min_age": 25})
        # for result in results:
        #     print(f"Name: {result['name']}, Age: {result['age']}")

    # except Exception as e:
    #     print(f"操作失败: {e}")
    # finally:
    #     # 关闭连接
    #     graph.close()
