"""日志配置：初始化根日志的统一输出格式与级别。"""

import logging


def configure_logging(level: str = "INFO") -> None:
    """按字符串级别初始化根日志，非法级别回退为 INFO。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
