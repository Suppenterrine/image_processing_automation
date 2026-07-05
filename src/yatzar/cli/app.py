import sys

import typer

from yatzar import __version__
from yatzar.cli.apply_cmd import apply_command
from yatzar.cli.gif_cmd import gif_command
from yatzar.cli.looks_cmd import looks_app

app = typer.Typer(
    name="yatzar",
    help="Wendet konfigurierbare, ästhetische Bild-Looks an — mit automatischem parallelem Batching.",
    no_args_is_help=True,
)

app.command("apply")(apply_command)
app.command("gif")(gif_command)
app.add_typer(looks_app, name="looks")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"yatzar {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(False, "--version", callback=_version_callback, is_eager=True, help="Version anzeigen und beenden"),
) -> None:
    pass


def _force_utf8_streams() -> None:
    # Auf Windows fällt die Stream-Kodierung bei nicht-interaktivem/umgeleitetem
    # Output auf die Locale-Kodierung (z.B. cp1252) zurück, die rich's Unicode-
    # Spinner-/Symbolzeichen nicht darstellen kann und mit UnicodeEncodeError abbricht.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _force_utf8_streams()
    try:
        app()
    except KeyboardInterrupt:
        typer.echo("\nAbgebrochen (Strg+C).")
        sys.exit(130)
