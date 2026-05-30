# src/utils/logger.py
"""日志设置"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_file: Path = None, level=logging.INFO):
    """设置日志"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除已有handler
    logger.handlers = []

    # 格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger