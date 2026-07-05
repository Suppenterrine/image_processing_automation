# Yatzar

> יָצַר (Yatzar) — Verb. Formen, gestalten, modellieren. Nicht erschaffen. Sondern wie ein Töpfer.
> Daher kommt יוצר (Yotzer) = Former, Gestalter. In der Bibel ist Gott häufig *Yotzer Or* — der Former des Lichts.

A CLI tool for applying cinematic and analog-inspired looks to batches of images — with automatic parallel batching and a live, per-batch progress display for large folders.

Built with OpenCV, Typer and Rich; looks are defined as plain YAML presets.

---

## Features

* Docker-style CLI with subcommands (`apply`, `looks`, `gif`)
* Registry-based look engines behind a single, uniform interface
* One YAML file per look preset, easy to add your own
* Automatic parallel batching for large input folders, with a live multi-row progress display (spinner, bar, percentage, elapsed time — one row per batch, updated independently, like `docker compose up`)
* Per-image error handling — a single broken file doesn't abort the whole run
* Film-inspired presets: `film_soft`, `film_heavy`, `tri_x_pushed`, `gazette`, `gazette_halftone`, `engraving`
* Optional motion blur with randomness control
* Automatic GIF generation via ImageMagick
* Reproducible randomness via `--seed` (see [Batching & seed behavior](#batching--parallel-processing))

---

## Installation

### Requirements

* Python 3.11+
* [ImageMagick](https://imagemagick.org/) (optional, only needed for `yatzar gif`)

### Setup

```bash
git clone <your-repo-url>
cd yatzar

uv sync            # installs into a local .venv via uv
# or
pip install .
```

This installs the `yatzar` command. If you want GIF support:

```bash
# Linux
sudo apt install imagemagick

# macOS
brew install imagemagick
```

---

## Quickstart

```bash
yatzar apply film_soft img/
```

Applies the `film_soft` look to every image in `img/` and writes the results to `output/`.

---

## CLI reference

### `yatzar apply <look> [input_dir]`

Applies a look to every image in `input_dir` (default: `img`).

| Option | Description |
| --- | --- |
| `-o, --output TEXT` | Output folder (default: `output`) |
| `--looks-dir PATH` | Use a custom directory of look YAMLs instead of the bundled defaults |
| `--seed INTEGER` | Seed for reproducible randomness |
| `--gif / --no-gif` | Also build a GIF from the output images (default: off) |
| `--batch-size INTEGER` | Max images per parallel batch (default: `25`) |
| `--workers INTEGER` | Number of parallel workers (default: CPU cores, capped by batch count) |
| `--no-parallel` | Force sequential processing, no batching |
| `--ext TEXT` | Output file extension (default: `.jpg`) |

### `yatzar looks list`

Lists all available look presets with their type.

### `yatzar looks show <name>`

Dumps the effective configuration of a look preset.

### `yatzar gif <input_dir>`

Builds a GIF from already-processed images (requires ImageMagick's `magick`).

| Option | Description |
| --- | --- |
| `--pattern TEXT` | Glob pattern for included images (default: `*.jpg`) |
| `--name TEXT` | Output filename (default: `animation.gif`) |
| `--delay INTEGER` | Frame delay in 1/100s, ImageMagick convention (default: `15`) |

---

## Configuration

Looks live as individual YAML files under `src/yatzar/data/looks/`, one file per preset (filename = preset name). Point `--looks-dir` at your own folder to use custom presets without touching the package.

Schema convention, consistent across all presets:
* Optional effect blocks always look like `{enabled: bool, ...params}` — e.g. `barrel_distortion`, `motion_blur`, `grain`, `vignette`, `paper_tone`.
* Parameter groups that are always active (no on/off switch) are still a named block, just without `enabled` — e.g. `contrast`, `sharpen`.

Example (`gazette.yaml`):

```yaml
type: gazette

contrast:
  alpha: 1.25
  beta: -10

sharpen:
  amount: 0.3

grain:
  amount: 27
  opacity: 0.24
  scale: 5

vignette:
  strength: 0.22

paper_tone:
  enabled: true
  sepia_amount: 0.1

motion_blur:
  enabled: false
```

### Built-in looks

| Name | Type | Description |
| --- | --- | --- |
| `film_soft` | `standard` | Gentle analog look: barrel distortion, soft blur, light grain, subtle vignette |
| `film_heavy` | `standard` | Same pipeline as `film_soft`, pushed harder — more distortion, blur, grain, vignette |
| `tri_x_pushed` | `tri_x_pushed` | High-contrast black-and-white, approximating pushed Kodak Tri-X: strong contrast curve, local sharpening, monochrome grain, vignette |
| `gazette` | `gazette` | Old newspaper photo look (19th/early 20th century): soft sharpening, coarse clumpy grain instead of fine noise, optional sepia paper tone |
| `gazette_halftone` | `gazette_halftone` | Variant of `gazette` with a real rotated halftone dot screen, like classic newspaper print |
| `engraving` | `engraving` | Steel/wood engraving approximation: edge-tangent-flow-following hatching with staggered cross-hatch layers, pure black/white |

Run `yatzar looks show <name>` to see the full parameter set of any preset.

---

## Batching & parallel processing

Large input folders are automatically split into balanced batches and processed in parallel, each shown as its own live-updating row (spinner, bar, percentage, elapsed time) — independent of the others, similar to `docker compose up`.

* **Batch size**: controlled by `--batch-size` (default `25`). Batches are balanced, not just chunked — e.g. 100 images → 4×25, 110 images → 5×22 (not 4×25 + 1×10).
* **Workers**: `--workers` overrides the default, which is the number of CPU cores capped by the number of batches.
* **No overhead for small folders**: if the folder fits in a single batch (`n <= batch-size`), processing runs synchronously in the main process — no worker pool, no extra machinery.
* **`--no-parallel`**: forces the single-batch, sequential path regardless of folder size.
* **Per-image error handling**: a broken/unreadable file is recorded as an error and skipped; the rest of the batch and run continue. Errors are printed together after the run, and the command exits with a non-zero status if any occurred.
* **Seed behavior with parallel batches**: each batch worker gets its own `random.Random(seed + batch_index)` (or an unseeded one if `--seed` is omitted). This makes runs reproducible for a given seed *and* batch layout, but is not equivalent to a single running random sequence over all files — a different `--batch-size` changes the batch layout and therefore the per-file randomness, even with the same seed. Note also that film grain uses NumPy's global RNG, which `--seed` does not control — grain noise differs between runs regardless of seed.

---

## Output

Processed images are saved as:

```
<look_name>_<original_name>.jpg
```

Optional GIF output:

```
animation.gif
```

---

## Project structure

```
.
├── pyproject.toml
├── README.md
├── src/yatzar/
│   ├── __main__.py          # python -m yatzar
│   ├── ops.py                # stateless, reusable image operations
│   ├── config.py              # look preset loader
│   ├── io_utils.py            # file discovery & saving
│   ├── gif.py                  # GIF export via ImageMagick
│   ├── batching.py             # batch-size/worker arithmetic (pure functions)
│   ├── processing.py           # per-batch worker function
│   ├── progress.py             # rich.Progress + ProcessPoolExecutor orchestration
│   ├── looks/                   # one module per look engine + registry
│   ├── data/looks/               # bundled default look presets (YAML)
│   └── cli/                      # Typer app and subcommands
├── tests/
├── img/            # your input images
└── output/         # generated output
```

---

## Development

```bash
uv sync --extra dev   # installs pytest alongside runtime dependencies
uv run pytest tests/
```

### Adding a new look

1. Write a function `(image, config, rng) -> image` in a new module under `src/yatzar/looks/`.
2. Register it with `@register("your_type_name")` and import the module from `src/yatzar/looks/__init__.py`.
3. Add a YAML preset under `src/yatzar/data/looks/` (or your own `--looks-dir`) with `type: your_type_name`.

---

## Notes

* Motion blur is applied probabilistically if enabled
* Grain is applied in float space to avoid uint8 artifacts
* Kernel sizes are automatically adjusted to odd values where required

---

## License

MIT

---

## Contributing

Pull requests are welcome. Keep changes minimal and focused.
