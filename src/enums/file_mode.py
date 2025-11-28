from enum import Enum


class FileReadMode(str, Enum):
    string = ("string",)
    bytes = ("bytes",)
