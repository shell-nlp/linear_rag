import time

from neo4j.exceptions import ServiceUnavailable, TransientError


def save_graph_batches(
    neo4j_driver,
    kb_name,
    node_batch,
    edge_batch,
    anchor_label="BaseNode",
    edge_label="LINK",
):
    try:
        cypher_kb = (
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{kb_name}`) "
            "REQUIRE n.orig_id IS UNIQUE"
        )
        neo4j_driver.execute_query(cypher_kb)
        cypher_base = (
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{anchor_label}`) "
            "REQUIRE n.orig_id IS UNIQUE"
        )
        neo4j_driver.execute_query(cypher_base)
    except Exception:
        pass

    create_nodes_cypher = f"""
    UNWIND $batch_data AS row
    MERGE (n:`{anchor_label}` {{orig_id: row.orig_id}})
    SET n:`{kb_name}`, n.name = row.name, n.type = row.type
    WITH n, row
    WHERE row.new_file_ids IS NOT NULL
    SET n.file_id = REDUCE(s = coalesce(n.file_id, []), fid IN row.new_file_ids |
        CASE WHEN fid IN s THEN s ELSE s + fid END
    )
    """

    node_batch_size = 2000
    for i in range(0, len(node_batch), node_batch_size):
        batch = node_batch[i : i + node_batch_size]
        if not batch:
            continue
        for attempt in range(3):
            try:
                neo4j_driver.execute_query(
                    create_nodes_cypher, {"batch_data": batch}
                )
                break
            except (TransientError, ServiceUnavailable):
                time.sleep(0.2 * (attempt + 1))
                if attempt == 2:
                    raise

    create_edges_cypher = f"""
    UNWIND $batch_data AS row
    MATCH (s:`{anchor_label}` {{orig_id: row.source}})
    MATCH (t:`{anchor_label}` {{orig_id: row.target}})
    WITH s, t, row
    ORDER BY id(s), id(t)
    MERGE (s)-[r:`{edge_label}`]->(t)
    SET r.weight = row.weight
    """

    edge_batch_size = 1000
    for i in range(0, len(edge_batch), edge_batch_size):
        batch = edge_batch[i : i + edge_batch_size]
        if not batch:
            continue
        for attempt in range(5):
            try:
                neo4j_driver.execute_query(
                    create_edges_cypher, {"batch_data": batch}
                )
                break
            except (TransientError, ServiceUnavailable):
                if attempt == 4:
                    raise
                time.sleep(0.5 * (2**attempt))

    return {
        "kb_name": kb_name,
        "node_count": len(node_batch),
        "edge_count": len(edge_batch),
    }


def delete_file_nodes(neo4j_driver, index_name, file_ids):
    if not isinstance(file_ids, list):
        file_ids = [file_ids]

    find_ids_query = f"""
        MATCH (n:`{index_name}`)
        WHERE any(fid IN $file_ids WHERE fid IN n.file_id)
        RETURN id(n) as node_id
        ORDER BY node_id ASC
        """

    update_query = f"""
        MATCH (n)
        WHERE id(n) IN $batch_node_ids
        REMOVE n:`{index_name}`
        SET n.file_id = [x IN n.file_id WHERE NOT x IN $file_ids]
        WITH n
        WHERE size(n.file_id) = 0 OR size(labels(n)) <= 1
        DETACH DELETE n
        """

    with neo4j_driver.get_session() as session:
        result = session.run(find_ids_query, file_ids=file_ids)
        sorted_node_ids = [record["node_id"] for record in result]

        total_nodes = len(sorted_node_ids)
        if total_nodes == 0:
            return {
                "index_name": index_name,
                "file_ids": file_ids,
                "total_nodes": 0,
                "deleted_nodes": 0,
            }

        batch_size = 1000
        deleted_count = 0
        for i in range(0, total_nodes, batch_size):
            batch_ids = sorted_node_ids[i : i + batch_size]
            update_result = session.run(
                update_query, batch_node_ids=batch_ids, file_ids=file_ids
            )
            summary = update_result.consume()
            deleted_count += summary.counters.nodes_deleted

    return {
        "index_name": index_name,
        "file_ids": file_ids,
        "total_nodes": total_nodes,
        "deleted_nodes": deleted_count,
    }
