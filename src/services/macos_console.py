import logging
from logging import Logger
from os import PathLike
from pathlib import Path
from typing import Literal

from src.enums.file_mode import FileReadMode
from src.services.base import OSConsoleServiceBase


class MacOSConsoleService(OSConsoleServiceBase):
    def __init__(self, logger: Logger):
        self._logger = logger

    def ls(self, path: PathLike[str] | str) -> list[str]:
        path = Path(path)
        if not path.exists():
            self._logger.error(f"Folder not found: {path}")
            raise FileNotFoundError(path)
        if not path.is_dir():
            self._logger.error(f"You entered {path} is not a directory")
            raise NotADirectoryError(path)
        self._logger.info(f"Listing {path}")
        return [entry.name + "\n" for entry in path.iterdir()]

    def cat(
        self,
        filename: PathLike[str] | str,
        mode: Literal[FileReadMode.string, FileReadMode.bytes] = FileReadMode.string,
    ) -> str | bytes:
        path = Path(filename)
        if not path.exists(follow_symlinks=True):
            self._logger.error(f"File not found: {filename}")
            raise FileNotFoundError(filename)
        if path.is_dir(follow_symlinks=True):
            self._logger.error(f"You entered {filename} is not a file")
            raise IsADirectoryError(f"You entered {filename} is not a file")
        try:
            self._logger.info(f"Reading file {filename} in mode {mode}")
            match mode:
                case FileReadMode.string:
                    return path.read_text(encoding="utf-8")
                case FileReadMode.bytes:
                    return path.read_bytes()
        except OSError as e:
            self._logger.exception(f"Error reading {filename}: {e}")
            raise
