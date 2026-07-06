"""Live-Progress-UI (mehrere gleichzeitig aktualisierte Zeilen, wie `docker compose up`).

Kleine Ordner laufen synchron im Hauptprozess (Fast-Path, keine Prozess-/Manager-
Overhead). Größere Ordner werden gemäß `batching.compute_batches` aufgeteilt und
über einen ProcessPoolExecutor parallel verarbeitet; die Worker melden ihren
Fortschritt über ein `multiprocessing.Manager().dict()`, das der Hauptprozess in
einer Polling-Schleife an die rich-Progress-Zeilen weitergibt.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from yatzar.batching import compute_batches, decide_workers
from yatzar.processing import BatchResult, process_batch

_POLL_INTERVAL_SECONDS = 0.1

_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[bold]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
)


def run_with_progress(
    files: list[Path],
    look_name: str,
    look_cfg: dict,
    output_dir: Path,
    seed: int | None,
    ext: str,
    batch_size: int,
    workers: int | None,
    no_parallel: bool,
    overwrite: bool = False,
) -> tuple[int, list[tuple[Path, str]]]:
    """Verarbeitet alle Dateien mit Live-Fortschrittsanzeige. Gibt (Anzahl gespeichert, Fehler) zurück."""
    effective_batch_size = len(files) if no_parallel else batch_size
    bounds = compute_batches(len(files), effective_batch_size)
    batches = [files[start:end] for start, end in bounds]

    console = Console()
    if len(batches) <= 1:
        return _run_sequential(batches, look_name, look_cfg, output_dir, seed, ext, console, overwrite=overwrite)
    return _run_parallel(batches, look_name, look_cfg, output_dir, seed, ext, workers, console, overwrite=overwrite)


def _run_sequential(batches, look_name, look_cfg, output_dir, seed, ext, console: Console, overwrite: bool = False) -> tuple[int, list]:
    if not batches:
        return 0, []

    files = batches[0]
    state: dict = {}
    with Progress(*_COLUMNS, console=console) as progress:
        task_id = progress.add_task(look_name, total=len(files))
        result = process_batch(files, look_name, look_cfg, output_dir, seed, 0, ext, state, overwrite=overwrite)
        progress.update(task_id, completed=state[0]["completed"], total=state[0]["total"])

    _print_errors(console, result.errors)
    return len(result.saved), result.errors


def _run_parallel(batches, look_name, look_cfg, output_dir, seed, ext, workers, console: Console, overwrite: bool = False) -> tuple[int, list]:
    num_workers = decide_workers(len(batches), workers)

    with Manager() as manager:
        shared = manager.dict()
        with Progress(*_COLUMNS, console=console) as progress:
            task_ids = {
                batch_id: progress.add_task(f"{look_name} · Batch {batch_id + 1}", total=len(batch_files))
                for batch_id, batch_files in enumerate(batches)
            }

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(
                        process_batch, batch_files, look_name, look_cfg, output_dir, seed, batch_id, ext, shared, overwrite
                    )
                    for batch_id, batch_files in enumerate(batches)
                ]

                while not all(f.done() for f in futures):
                    _sync_progress(progress, task_ids, shared)
                    time.sleep(_POLL_INTERVAL_SECONDS)
                _sync_progress(progress, task_ids, shared)

                results: list[BatchResult] = [f.result() for f in futures]

    saved_count = sum(len(r.saved) for r in results)
    all_errors = [error for r in results for error in r.errors]
    _print_errors(console, all_errors)
    return saved_count, all_errors


def _sync_progress(progress: Progress, task_ids: dict[int, int], shared) -> None:
    for batch_id, state in shared.items():
        progress.update(task_ids[batch_id], completed=state["completed"], total=state["total"])


def _print_errors(console: Console, errors: list[tuple[Path, str]]) -> None:
    if not errors:
        return
    console.print(f"\n[bold red]{len(errors)} Fehler:[/bold red]")
    for path, message in errors:
        console.print(f"  [red]x[/red] {path}: {message}")
