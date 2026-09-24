import functools
import logging
import time

from prodml.logging_conf import setup_logging

setup_logging()
logger = logging.getLoger(__name__)


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} a pris {end - start:.4f} secondes")
        return result

    return wrapper
