"""SciDER — multi-agent system for automating research processes.

Side effect on import: configure loguru's default sink to honour the
``LOGURU_LEVEL`` env var (default ``INFO``). loguru does NOT auto-read this
env var on its own — without this, every CLI / script inherits loguru's
hard-coded DEBUG default and floods stdout with tool-registration noise.

If you want a different sink (e.g. file output, structured json) configure
it after import and the env var simply controls THIS default sink.
"""

import os
import sys

from loguru import logger

_LOGURU_LEVEL = os.getenv("LOGURU_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=_LOGURU_LEVEL)
