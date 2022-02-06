import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger()

CRITICAL = logging.CRITICAL
ERROR = logging.ERROR
WARNING = logging.WARNING
WARN = WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG

class context_disabled():
    def __enter__(self):
        disabled()
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        enabled()

def init_logger(log_file=None,
                log_file_level=logging.NOTSET,
                rotate=False,
                log_level=logging.INFO,):
    """
    Adopted from OpenNMT-py:
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/utils/logging.py
    """
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if rotate:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10)
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

def disabled():
    return logging.disable(logging.CRITICAL)

def enabled():
    return logging.disable(logging.NOTSET)
