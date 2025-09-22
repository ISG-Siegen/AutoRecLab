import logging

from rich.logging import RichHandler

_ROOT_LOGGER = logging.getLogger("isgsa")

logging.basicConfig(
    format="%(message)s", datefmt="[%Y/%m/%d %H:%M:%S]", handlers=[RichHandler()]
)


def set_log_level(level: str):
    level = level.upper()
    if level in logging._nameToLevel:
        _ROOT_LOGGER.setLevel(level)
        _ROOT_LOGGER.debug("LOG LEVEL IS SET TO DEBUG")
    else:
        raise ValueError(f"Unknown log level: {level}")
