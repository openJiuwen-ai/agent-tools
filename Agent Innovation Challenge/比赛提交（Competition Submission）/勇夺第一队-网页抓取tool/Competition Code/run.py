import os
import sys
from pathlib import Path

import uvicorn


def main() -> None:
    # Make sure local package is importable even if editable install did not take effect.
    root = Path(__file__).resolve().parent
    src_dir = root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    # Some Windows environments reserve/deny binding to certain ports.
    # Default to a higher port; you can also set WEB_SCRAPER_PORT=0 to let OS pick a free port.
    host = os.environ.get("WEB_SCRAPER_HOST", "0.0.0.0")
    port = int(os.environ.get("WEB_SCRAPER_PORT", "18086"))
    uvicorn.run(
        "webscraper_tool.app:app",
        host=host,
        port=port,
        reload=False,
        http="h11",
    )


if __name__ == "__main__":
    main()
