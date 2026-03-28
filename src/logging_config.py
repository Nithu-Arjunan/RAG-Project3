import logging
import os


# configure_logging.
def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
