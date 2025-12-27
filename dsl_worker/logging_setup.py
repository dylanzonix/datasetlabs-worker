import logging
import os

def setup_logging(level: str | None = None) -> None:
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return  # already configured, don't touch

    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    level_value = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level_value,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
