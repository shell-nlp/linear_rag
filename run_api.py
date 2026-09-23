# run_local.py

import os
import warnings

from src.common.model_providers import create_embedding_provider
from src.settings import get_settings
from src.utils import get_es_client, setup_logging
from src.common.document_processing.ner import SpacyNER
from src.common.search_store import ElasticsearchSearchStore
from src.indexing.service import IndexingService

warnings.filterwarnings("ignore")


DATASET_NAME = "miyun_test"  # 要使用的数据集名称S
settings = get_settings()


def setup_environment():
    """设置OpenAI库所需的环境变量，使其指向您的本地LLM服务。"""
    os.environ["OPENAI_API_KEY"] = settings.llm_api_key
    os.environ["OPENAI_BASE_URL"] = settings.llm_base_url
    print(f"环境变量已设置: OPENAI_BASE_URL -> {settings.llm_base_url}")


def load_local_embedding_model(api_url, model_name):
    """实例化本地Embedding模型的API客户端。"""
    print(f"连接到本地Embedding服务: {api_url} (模型: {model_name})")
    return create_embedding_provider(
        api_url=api_url,
        model_name=model_name,
        api_key=settings.llm_api_key,
    )


def main():
    """主执行函数"""
    setup_environment()
    embedding_model = load_local_embedding_model(
        settings.embedding_api_url,
        settings.embedding_model_name,
    )

    log_dir = f"results/{DATASET_NAME}"
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(os.path.join(log_dir, "log.txt"))

    print(f"初始化LLM客户端 (模型: {settings.llm_model_name})")

    es_client = get_es_client()
    search_store = ElasticsearchSearchStore(es_client)

    print("开始创建索引...")
    config = settings.runtime_config()

    document_list = [
        "密云水库坐落在燕山南麓密云区境内，距北京市中心约90km，总库容 43.75 亿 m3，为华北地区最大的水库。工程于 1958 年 9 月动工兴建，1959 年汛期拦洪，1960 年 9 月基本建成，是一座具有防洪、供水等多种功能综合利用、多年调节的大型水利枢纽，目前是首都北京最重要的地表饮用水源地。水库工程按千年一遇洪水设计，万年一遇洪水校核，坝顶高程 160.00m，校核水位 158.50m，设计水位 157.50m，汛期限制水位 152.00m，死水位 126.00m，调洪库容 11.08亿 m3，防洪库容 9.27 亿 m3，兴利库容 35.45 亿 m3，死库容4.19 亿 m3。密云水库水工建筑物及附属设施众多，主要包括 7 座主副坝、3 座溢洪道、7 条输泄水隧洞、1 座调节池、41 扇闸门、43 台启闭机、43.55km 高低压线路、36 台变压器、16台发电机等。",
        "密云水库大坝和密云水库开展的工情监测项目有渗流、变形、混凝土应力应变、温度和裂缝监测等，监测设施包括 256 个测压管、12 个量水堰、203 个变形标点、14 个三向测缝仪、37 个钢2筋计、12 个测缝计、12 个应变计、4 个电阻温度计等。",
    ]
    passages = {
        "text": document_list,
        "pages_number": [1, 1],
        "content_table": [],
        "content_image": [],
        "ori_text": document_list,
        "file_name": ["密云水库项目.pdf", "密云水库项目2.pdf"],
        "file_id": ["123", "1234"],
        "segment_id": [1, 1],
        "file_path": ["密云水库项目.pdf", "密云水库项目2.pdf"],
        "bucket_name": ["111", "111"],
    }

    indexing_service = IndexingService(
        config=config,
        search_store=search_store,
        embedding_provider=embedding_model,
        entity_extractor=SpacyNER(settings.spacy_model),
    )
    indexing_service.index(passages, kb_name="hh_test")

    # print("开始进行检索...")
    # questions = [
    #     {"question": "密云水库开展的工情监测项目有什么?", "answer": "..."}
    # ]
    # results = retrieval_service.retrieve(
    #     questions, index_names=["lin_test"], top_k=3
    # )

    # print("检索结果:")
    # for idx, res in enumerate(results):
    #     print(res)
    # print(results)

    # 删除功能
    # indexing_service.delete_files(index_name="lin_test11", file_ids=["456"])


if __name__ == "__main__":
    main()
