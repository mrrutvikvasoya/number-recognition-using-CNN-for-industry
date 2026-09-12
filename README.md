# Vision Crafters Seal OCR

An offline OCR system for reading the seven-digit numeric code stamped on an industrial seal. It uses OpenCV to locate the digit row and a compact convolutional neural network (CNN) to recognize the seven digits while avoiding distractor text such as `TESCO`.

![Detected seal code with prediction](assets/example_prediction.jpg)

## Pipeline

1. Read one PNG image in grayscale.
2. Find seven aligned digit-shaped components using thresholding and geometric checks.
3. Deskew, normalize, and resize each digit crop to 64 × 64 pixels.
4. Predict the seven crops together with the CNN.
5. Save the code and continue with the next image.

The model is loaded once when the program starts. Full seal images are processed one at a time in sorted filename order.

## Local results

| Evaluation split | Exact codes | Accuracy |
| --- | ---: | ---: |
| Validation | 1,414 / 1,414 | 100.0000% |
| Labelled local test | 1,413 / 1,414 | 99.9293% |

The complete test run produced 1,414 unique, correctly formatted rows with no missing images. These measurements apply to the supplied local dataset and do not guarantee performance on different or hidden images.

## Requirements

- Python 3.12
- NumPy 1.26.4
- OpenCV 4.11.0
- PyTorch 2.10.0

Install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

For GPU inference, install a PyTorch build compatible with the computer's NVIDIA driver. The program automatically falls back to CPU when CUDA is unavailable.

## Run

Open PowerShell in the repository directory and run:

```powershell
python -B main.py --input-dir PATH_TO_PNG_IMAGES --output-dir results --device auto
```

The program creates `results/vision_crafters.csv`:

```text
filename;number
00000.png;1938192
00001.png;1581456
```

Use a new output directory for each run. The program refuses to overwrite an existing result CSV.

## Failure handling

If normal localization cannot find seven digits, the program tries a fixed seven-cell layout with both crop polarities and keeps the more confident CNN result. If an image cannot be decoded or fallback inference also fails, it logs the failure, writes `0000000`, and continues. This guarantees one output row per uniquely named PNG, although the fallback value cannot be considered an accurate reading of an unreadable image.

## Repository contents

| Path | Purpose |
| --- | --- |
| `main.py` | Command-line entry point |
| `common/` | OpenCV localization and digit preprocessing |
| `cnn/` | CNN architecture and inference code |
| `weights/best.pt` | Trained baseline checkpoint |
| `assets/` | README example image |
| `requirements.txt` | Pinned runtime dependencies |
| `RUN.md` | Short competition run guide |

No training dataset, labels, test-specific correction, network API, OCR service, or language model is required at runtime.

## License

Provided for the BTHA Summer School 2026 competition.
