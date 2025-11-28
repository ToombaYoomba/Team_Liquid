from dataclasses import dataclass
from logging import Logger

from src.services.base import OSConsoleServiceBase


@dataclass
class Container:
    console_service: OSConsoleServiceBase
