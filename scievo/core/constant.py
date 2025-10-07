import os

import global_state
from dotenv import load_dotenv

load_dotenv()  # 加载.env文件


def str_to_bool(value) -> bool:
    """convert string to bool"""
    true_values = {"true", "yes", "1", "on", "t", "y"}
    false_values = {"false", "no", "0", "off", "f", "n"}

    if isinstance(value, bool):
        return value

    if not value:
        return False

    value = str(value).lower().strip()
    if value in true_values:
        return True
    if value in false_values:
        return False
    return True  # default return True


DEBUG = str_to_bool(os.getenv("DEBUG", True))

DEFAULT_LOG = str_to_bool(os.getenv("DEFAULT_LOG", True))
LOG_PATH = os.getenv("LOG_PATH", global_state.LOG_PATH)
EVAL_MODE = str_to_bool(os.getenv("EVAL_MODE", False))

COMPLETION_MODEL = os.getenv("COMPLETION_MODEL", "gpt-4o-2024-08-06")  # gpt-4o-2024-08-06
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

API_KEY = os.getenv("API_KEY", None)
FN_CALL = str_to_bool(os.getenv("FN_CALL", True))
API_BASE_URL = os.getenv("API_BASE_URL", None)

if EVAL_MODE:
    DEFAULT_LOG = False
