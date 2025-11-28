from src.common.config import LOGGING_CONFIG

import logging
import sys
from pathlib import Path

import typer
from typer import Typer, Context

from src.dependencies.container import Container
from src.enums.file_mode import FileReadMode
from src.services.macos_console import MacOSConsoleService

app = Typer()


def get_container(ctx: Context) -> Container:
    container = ctx.obj
    if not isinstance(container, Container):
        raise RuntimeError("DI container is not initialized")
    return container


@app.callback()
def main(ctx: Context):
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    ctx.obj = Container(
        console_service=MacOSConsoleService(logger=logger),
    )


@app.command()
def ls(
    ctx: Context,
    path: Path = typer.Argument(
        ..., exists=False, readable=False, help="File to print"
    ),
) -> None:
    """
    List all files in a directory.
    :param ctx:   typer context object for imitating di container
    :param path:  path to directory to list
    :return: content of directory
    """
    try:
        container: Container = get_container(ctx)
        content = container.console_service.ls(path)
        sys.stdout.writelines(content)
    except OSError as e:
        typer.echo(e)


@app.command()
def cat(
    ctx: Context,
    filename: Path = typer.Argument(
        ..., exists=False, readable=False, help="File to print"
    ),
    mode: bool = typer.Option(False, "--bytes", "-b", help="Read as bytes"),
):
    """
    Cat a file
    :param ctx: typer context object for imitating di container
    :param filename: Filename to cat
    :param mode: Mode to read the file in
    :return:
    """
    try:
        container: Container = get_container(ctx)
        mode = FileReadMode.bytes if mode else FileReadMode.string
        data = container.console_service.cat(
            filename,
            mode=mode,
        )
        if isinstance(data, bytes):
            sys.stdout.buffer.write(data)
        else:
            sys.stdout.write(data)
    except OSError as e:
        typer.echo(e)


if __name__ == "__main__":
    app()
