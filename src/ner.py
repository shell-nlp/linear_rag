import spacy
from collections import defaultdict

class SpacyNER:
    def __init__(self,spacy_model):
        self.spacy_model = spacy.load(spacy_model)

    def batch_ner(self, hash_id_to_passage, max_workers):
        all_keys  = list(hash_id_to_passage.keys())
        passage_texts = list(hash_id_to_passage.values())
        batch_size = max(1, len(passage_texts) // max_workers)       
        # 2. 对文本进行处理
        docs_list = self.spacy_model.pipe(passage_texts, batch_size=batch_size, n_process=max_workers) 
        passage_hash_id_to_entities = {}
        for idx,doc in enumerate(docs_list):
            passage_hash_id = all_keys[idx]
            single_passage_hash_id_to_entities,single_sentence_to_entities = self.extract_entities_sentences(doc,passage_hash_id)
            passage_hash_id_to_entities.update(single_passage_hash_id_to_entities)
        return passage_hash_id_to_entities
            
    def extract_entities_sentences(self, doc,passage_hash_id):
        sentence_to_entities = defaultdict(list)
        unique_entities = set()
        passage_hash_id_to_entities = {}
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            sent_text = ent.sent.text
            ent_text = ent.text
            if ent_text not in sentence_to_entities[sent_text]:
                sentence_to_entities[sent_text].append(ent_text)
            unique_entities.add(ent_text)
        passage_hash_id_to_entities[passage_hash_id] = list(unique_entities)
        return passage_hash_id_to_entities,sentence_to_entities

    def question_ner(self, question: str):
        doc = self.spacy_model(question)
        question_entities = set()
        for ent in doc.ents:
            if ent.label_ == "ORDINAL" or ent.label_ == "CARDINAL":
                continue
            question_entities.add(ent.text.lower())
        return question_entities