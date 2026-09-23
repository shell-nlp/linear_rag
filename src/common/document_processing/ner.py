from collections import defaultdict
import os

import spacy

class SpacyNER:
    """基于 spaCy 的实体识别实现。"""

    def __init__(self,spacy_model):
        """加载指定名称的 spaCy 模型。"""

        self.spacy_model = spacy.load(spacy_model)

    def batch_ner(self, hash_id_to_passage, max_workers):
        """批量提取段落实体，兼容旧的调用名称。"""

        all_keys  = list(hash_id_to_passage.keys())
        passage_texts = list(hash_id_to_passage.values())
        requested_workers = int(os.getenv("SPACY_N_PROCESS", "1"))
        worker_count = max(1, min(max_workers, requested_workers, len(passage_texts)))
        batch_size = max(1, len(passage_texts) // worker_count)
        # 2. 对文本进行处理
        docs_list = self.spacy_model.pipe(
            passage_texts,
            batch_size=batch_size,
            n_process=worker_count,
        )
        passage_hash_id_to_entities = {}
        for idx,doc in enumerate(docs_list):
            passage_hash_id = all_keys[idx]
            single_passage_hash_id_to_entities,single_sentence_to_entities = self.extract_entities_sentences(doc,passage_hash_id)
            passage_hash_id_to_entities.update(single_passage_hash_id_to_entities)
        return passage_hash_id_to_entities
            
    def extract_entities_sentences(self, doc,passage_hash_id):
        """提取整篇段落实体以及句子级实体。"""

        sentence_to_entities = defaultdict(list)
        passage_entities = []
        passage_hash_id_to_entities = {}
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            sent_text = ent.sent.text
            ent_text = ent.text
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            # 保留重复实体，索引服务需要据此计算段落内出现次数。
            passage_entities.append(ent_text)
        passage_hash_id_to_entities[passage_hash_id] = passage_entities
        return passage_hash_id_to_entities,sentence_to_entities

    def extract_passage_entities(
        self,
        hash_id_to_passage,
        max_workers: int,
    ) -> dict[str, list[str]]:
        """实现 EntityExtractor 端口的段落实体提取方法。"""

        return self.batch_ner(hash_id_to_passage, max_workers)
