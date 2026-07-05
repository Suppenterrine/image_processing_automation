from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from yatzar import gif as gif_module
from yatzar import io_utils, looks
from yatzar.config import default_looks_dir, load_look_configs
from yatzar.progress import run_with_progress

console = Console()


def apply_command(
    look: str = typer.Argument(..., help="Name des Look-Presets (siehe `yatzar looks list`)"),
    input_dir: str = typer.Argument("img", help="Ordner mit Eingabebildern"),
    output: str = typer.Option("output", "--output", "-o", help="Ausgabe-Ordner"),
    looks_dir: Optional[Path] = typer.Option(None, "--looks-dir", help="Eigenes Verzeichnis mit Look-YAMLs statt der gebündelten Defaults"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed für reproduzierbare Zufalls-Effekte (z.B. Motion Blur)"),
    make_gif: bool = typer.Option(False, "--gif/--no-gif", help="Zusätzlich ein GIF aus den Ausgabebildern erzeugen"),
    batch_size: int = typer.Option(25, "--batch-size", help="Maximale Bilder pro Parallel-Batch"),
    workers: Optional[int] = typer.Option(None, "--workers", help="Anzahl paralleler Worker (Default: CPU-Kerne, gedeckelt durch Batch-Anzahl)"),
    no_parallel: bool = typer.Option(False, "--no-parallel", help="Erzwingt sequentielle Verarbeitung ohne Batching"),
    ext: str = typer.Option(".jpg", "--ext", help="Dateiendung der Ausgabebilder"),
) -> None:
    """Wendet einen Look auf alle Bilder in INPUT_DIR an."""
    configs = load_look_configs(looks_dir or default_looks_dir())
    if look not in configs:
        available = ", ".join(sorted(configs)) or "(keine gefunden)"
        console.print(f"[red]Look '{look}' nicht gefunden.[/red] Verfügbar: {available}")
        raise typer.Exit(code=1)

    look_cfg = configs[look]
    looks.get(look_cfg["type"])  # früh validieren, dass der Typ registriert ist

    input_files = io_utils.collect_input_files(input_dir)
    if not input_files:
        console.print(f"Keine Bilder in '{input_dir}' gefunden.")
        return

    console.print(f"Verarbeite {len(input_files)} Bilder mit Look: [bold]{look}[/bold]")

    output_dir = Path(output)
    saved_count, errors = run_with_progress(
        files=input_files,
        look_name=look,
        look_cfg=look_cfg,
        output_dir=output_dir,
        seed=seed,
        ext=ext,
        batch_size=batch_size,
        workers=workers,
        no_parallel=no_parallel,
    )

    console.print(f"[green]{saved_count}[/green] Bilder gespeichert in '{output_dir}'.")
    if errors:
        raise typer.Exit(code=1)

    if make_gif:
        gif_module.create_gif_from_outputs(output_dir, pattern=f"{look}_*{ext}")
