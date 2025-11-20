import os
import logging

logger = logging.getLogger("NTOPS")

NT_DEBUG = os.getenv("NT_DEBUG", 'false').lower()

if NT_DEBUG in ["true", "1"]:
    NT_DEBUG = True
else:
    NT_DEBUG = False

logger.info_once(f"\033[31mNT DEBUG: {NT_DEBUG}\033[0m")

def debug_log(info: str):
    if NT_DEBUG:
        logger.info(f"\033[31m{info}\033[0m")