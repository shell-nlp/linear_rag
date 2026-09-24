from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.settings import ENV_FILE, Settings


class SettingsTests(unittest.TestCase):
    """验证 pydantic-settings 能从环境变量读取配置。"""

    def test_env_file_points_to_project_root(self):
        """配置文件必须读取项目根目录下的 .env，而不是父目录。"""

        self.assertEqual(
            Path(ENV_FILE).resolve(),
            Path(__file__).resolve().parents[1] / ".env",
        )

    def test_environment_aliases_are_loaded(self):
        """兼容旧变量名，同时统一为 Settings 字段。"""

        with patch.dict(
            os.environ,
            {
                "EMBEDDING_DIM": "512",
                "es_url": "http://localhost:9200",
                "ENTITY_EXPANSION_TOP_K": "30",
            },
            clear=False,
        ):
            settings = Settings(_env_file=None)

        self.assertEqual(settings.embedding_dim, 512)
        self.assertEqual(settings.es_url, "http://localhost:9200")
        self.assertEqual(settings.entity_expansion_top_k, 30)

    def test_runtime_config_uses_settings_values(self):
        """运行时配置应由 Settings 统一构造。"""

        settings = Settings(
            _env_file=None,
            max_workers=7,
            spacy_model="test_model",
            working_dir="/tmp/import",
        )
        config = settings.runtime_config()

        self.assertEqual(config.max_workers, 7)
        self.assertEqual(config.spacy_model, "test_model")
        self.assertEqual(config.working_dir, "/tmp/import")


if __name__ == "__main__":
    unittest.main()
