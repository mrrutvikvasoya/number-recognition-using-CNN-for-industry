# Vision Crafters Seal OCR

This project reads a seven-digit code stamped on an industrial seal while ignoring nearby text and hardware. It combines OpenCV localization with a compact convolutional neural network (CNN) digit classifier.

![Example seal prediction](assets/example_prediction.jpg)

## How it works

1. Read one PNG seal image in grayscale.
2. Locate a row of seven digit-shaped components with thresholding and geometric checks.
3. Deskew, normalize, and resize each digit to 64 × 64 pixels.
4. Classify all seven crops together with the CNN.
5. Write `filename;number` to the result CSV, then process the next image.

The trained model is loaded once. Seal images are processed sequentially, while the seven digits from the current image are inferred together. Distractor text such as `TESCO` is rejected by the row geometry used during localization.

## Results

| Evaluation | Exact codes | Result |
| --- | ---: | ---: |
| Validation images | 1,414 / 1,414 | 100.0000% |
| Local labelled test images | 1,413 / 1,414 | 99.9293% |

The test run produced 1,414 unique, correctly formatted rows with no missing images. These results describe the supplied local splits and do not guarantee the score on unseen competition data.

## Run the verified submission

Python 3.12 and the packages in `backups/submission/requirements.txt` are required:

```powershell
python -m pip install -r backups/submission/requirements.txt
python -B backups/submission/main.py --input-dir PATH_TO_PNG_IMAGES --output-dir results --device auto
```

The command creates `results/vision_crafters.csv`:

```text
filename;number
00000.png;1938192
00001.png;1581456
```

Use a new output directory for every run because the program refuses to overwrite an existing result. `--device auto` uses CUDA when it is available and otherwise uses the CPU.

## Failure handling

If normal localization cannot find seven digits, the program tries a fixed seven-cell layout with both crop polarities and keeps the more confident CNN result. If decoding or fallback inference also fails, it writes the deterministic value `0000000`, logs the error, and continues. This guarantees one CSV row per uniquely named PNG; the fallback value is not claimed to be an accurate reading of an unreadable image.

## Repository structure

| Path | Purpose |
| --- | --- |
| `backups/submission/` | Verified, self-contained baseline submission |
| `submission/` | Current submission candidate with optional residual-ensemble support |
| `common/` | Localization, preprocessing, augmentation, and file handling |
| `cnn/` | CNN architectures, training, inference, metrics, and checkpoints |
| `audit/` | Dataset preparation, evaluation, review galleries, and tests |
| `TRAINING.md` | Training, validation, ensemble, and promotion commands |
| `reports/` | Dataset and localization audit evidence |

The current submission can automatically ensemble the baseline with a trained residual checkpoint named `submission/weights/residual.pt`. That residual checkpoint has not been fully trained or promoted. The verified backup therefore remains the recommended competition submission.

## Verification

The backup submission completed all 1,414 local test images. Separate runtime checks covered a normal image, a valid blank image that forced localization fallback, and an unreadable PNG; all produced rows and the process exited successfully. The project test suite contains 35 passing tests. Near-duplicate seals and hidden-test performance remain unproven.

## Data and training

Training uses 9,901 leakage-filtered seal records, producing 69,307 digit crops. Validation contains 1,414 seals and remains unaugmented. Training augmentation includes clean samples, small affine changes, gamma, contrast, noise, defocus, motion blur, mild morphology, uneven illumination, and fading. Labels come from the supplied seal CSV and are assigned to crops from left to right.

The dataset is not redistributed by this README. Place it according to the paths described in `TRAINING.md` before running development or training commands.
