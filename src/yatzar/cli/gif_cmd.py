from pathlib import Path

import typer

from yatzar.gif import create_gif_from_outputs


def gif_command(
    input_dir: str = typer.Argument(..., help="Ordner mit Bildern für das GIF"),
    pattern: str = typer.Option("*.jpg", "--pattern", help="Glob-Muster für die einzubindenden Bilder"),
    name: str = typer.Option("animation.gif", "--name", help="Dateiname des erzeugten GIFs"),
    delay: int = typer.Option(15, "--delay", help="Verzögerung zwischen Frames (in 1/100s, ImageMagick-Konvention)"),
) -> None:
    """Baut ein GIF aus bereits vorhandenen Bildern in INPUT_DIR (benötigt ImageMagick's `magick`)."""
    create_gif_from_outputs(Path(input_dir), pattern=pattern, gif_name=name, delay=delay)
