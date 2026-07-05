from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from yatzar.config import default_looks_dir, load_look_configs

console = Console()
looks_app = typer.Typer(help="Verfügbare Look-Presets auflisten und inspizieren.")


@looks_app.command("list")
def list_looks(
    looks_dir: Optional[Path] = typer.Option(None, "--looks-dir", help="Eigenes Verzeichnis mit Look-YAMLs statt der gebündelten Defaults"),
) -> None:
    """Listet alle verfügbaren Look-Presets mit ihrem Typ."""
    configs = load_look_configs(looks_dir or default_looks_dir())
    table = Table()
    table.add_column("Name")
    table.add_column("Type")
    for name, cfg in sorted(configs.items()):
        table.add_row(name, cfg.get("type", "?"))
    console.print(table)


@looks_app.command("show")
def show_look(
    name: str = typer.Argument(..., help="Name des Look-Presets"),
    looks_dir: Optional[Path] = typer.Option(None, "--looks-dir", help="Eigenes Verzeichnis mit Look-YAMLs statt der gebündelten Defaults"),
) -> None:
    """Zeigt die effektive Konfiguration eines Look-Presets."""
    configs = load_look_configs(looks_dir or default_looks_dir())
    if name not in configs:
        available = ", ".join(sorted(configs)) or "(keine gefunden)"
        console.print(f"[red]Look '{name}' nicht gefunden.[/red] Verfügbar: {available}")
        raise typer.Exit(code=1)
    console.print(yaml.dump(configs[name], sort_keys=False, allow_unicode=True))
