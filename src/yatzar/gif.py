"""GIF-Erzeugung aus einer Bildsequenz via externes ImageMagick (`magick`)."""

import subprocess
from pathlib import Path


def create_gif_from_outputs(output_dir: Path, pattern: str = "*.jpg", gif_name: str = "animation.gif", delay: int = 15) -> None:
    files = sorted(output_dir.glob(pattern))
    if not files:
        print("Keine Dateien für GIF gefunden.")
        return

    cmd = ["magick", "-delay", str(delay), "-loop", "0"]
    cmd.extend([f.name for f in files])
    cmd.append(gif_name)

    result = subprocess.run(
        cmd,
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        shell=False
    )

    if result.returncode != 0:
        print("ImageMagick Fehler:")
        print(result.stderr.strip())
    else:
        print(f"GIF erstellt: {output_dir / gif_name}")
