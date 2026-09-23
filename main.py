from __future__ import annotations

import uvicorn

from src.api.app import create_app
from src.settings import get_settings

app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)
