# Image Look Pipeline

A lightweight, configurable image processing pipeline for applying cinematic and analog-inspired looks to batches of images.

Built with OpenCV and YAML-based configuration for flexibility and reproducibility.

---

## Features

* Modular image processing pipeline
* YAML-driven configuration for custom looks
* Batch processing of images
* Optional motion blur with randomness control
* Film-inspired presets (e.g. Tri-X pushed)
* Automatic GIF generation via ImageMagick
* Reproducible randomness via seed

---

## Installation

### Requirements

* Python 3.9+
* OpenCV (`cv2`)
* NumPy
* PyYAML
* ImageMagick (optional, for GIF export)

### Setup

```bash
git clone <your-repo-url>
cd <repo-name>

pip install -r requirements.txt
```

If you want GIF support:

```bash
# Linux
sudo apt install imagemagick

# macOS
brew install imagemagick
```

---

## Usage

Basic command:

```bash
python main.py --look <look_name>
```

### Arguments

| Argument   | Description                                 |
| ---------- | ------------------------------------------- |
| `--config` | Path to YAML config (default: `looks.yaml`) |
| `--look`   | Name of the look (required)                 |
| `--input`  | Input folder (default: `img`)               |
| `--output` | Output folder (default: `output`)           |
| `--gif`    | Generate GIF from outputs                   |
| `--seed`   | Set seed for reproducible randomness        |

### Example

```bash
python main.py --look tri_x --input photos --output results --gif --seed 42
```

---

## Project Structure

```
.
├── main.py
├── looks.yaml
├── img/
├── output/
└── README.md
```

---

## Configuration

All looks are defined in `looks.yaml`.

Example:

```yaml
looks:
  tri_x:
    type: tri_x_pushed
    contrast_alpha: 1.22
    contrast_beta: -8
    sharpen_amount: 1.2
    grain_amount: 22
    grain_opacity: 0.18
    vignette_strength: 0.18
    motion_blur:
      enabled: true
      chance: 0.3
      length_min: 5
      length_max: 20
      angle_min: 0
      angle_max: 180
```

### Available Pipeline Blocks

* `barrel_distortion`
* `gaussian_blur`
* `motion_blur`
* `grain`
* `color` (HSV adjustments)
* `tone` (contrast/brightness)
* `vignette`

Each block can be enabled/disabled and tuned individually.

---

## Built-in Looks

### Tri-X Pushed

A high-contrast black-and-white look inspired by pushed Kodak Tri-X film.

Includes:

* Strong contrast curve
* Local sharpening
* Monochrome grain
* Subtle vignette

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
