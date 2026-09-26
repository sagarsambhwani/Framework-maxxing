"""Server CLI Command Handler."""

from src.common.config import settings
from src.common.logging import print_banner


def run_server():
    """Starts the FastAPI Web Application & ChatGPT dark-themed interface."""
    import uvicorn
    from src.server.app import create_app

    app = create_app()
    print_banner(
        "CHATGPT PRO FASTAPI + JAVASCRIPT SERVER",
        f"Web UI: http://localhost:{settings.SERVER_PORT} | Terminal Logs: ACTIVE | Debug Mode: {'ON' if settings.DEBUG_MODE else 'OFF'}"
    )
    uvicorn.run(app, host="127.0.0.1", port=settings.SERVER_PORT, log_level="warning")
