import sys
from loguru import logger

# 移除默认 handler
logger.remove()

# 控制台输出：彩色、带时间
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True
)

# 文件输出：按天切割，保留 7 天
logger.add(
    "logs/app_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    level="INFO",
    encoding="utf-8"
)

__all__ = ["logger"]