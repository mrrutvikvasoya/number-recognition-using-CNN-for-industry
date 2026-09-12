import argparse
import csv
import logging
import sys
from pathlib import Path
from time import perf_counter

import cv2

# Prefer the modules bundled beside this file, regardless of the launch directory.
SUBMISSION_ROOT = Path(__file__).resolve().parent
SUBMISSION_PATH = str(SUBMISSION_ROOT)
if SUBMISSION_PATH in sys.path:
    sys.path.remove(SUBMISSION_PATH)
sys.path.insert(0, SUBMISSION_PATH)

from cnn.predict import load_models, predict_crops
from cnn.runtime import select_device
from common.config import DIGIT_COUNT
from common.crops import prepare_crops
from common.files import read_gray
from common.localize import Localization, find_digits

LOGGER = logging.getLogger(__name__)
# The evaluator accepts one semicolon-separated file with these exact columns.
CSV_COLUMNS = ('filename', 'number')
# A neutral filename works until the organizer requires the registered team name.
DEFAULT_CSV_NAME = 'vision_crafters.csv'
# A promoted residual checkpoint is automatically ensembled with the proven baseline.
RESIDUAL_CHECKPOINT = SUBMISSION_ROOT / 'weights' / 'residual.pt'
# Training seals place digit centers at regular intervals across this horizontal band.
FALLBACK_FIRST_CENTER = 0.276
FALLBACK_CENTER_STEP = 0.065
FALLBACK_BOX_WIDTH = 0.055
FALLBACK_BOX_TOP = 0.64
FALLBACK_BOX_HEIGHT = 0.18
FALLBACK_NUMBER = (0,) * DIGIT_COUNT


def parse_arguments():
    """Read the competition directories and optional runtime settings."""
    parser = argparse.ArgumentParser(description='Read seven-digit codes from seal PNG images.')
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--csv-name', default=DEFAULT_CSV_NAME)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    arguments = parser.parse_args()
    if Path(arguments.csv_name).name != arguments.csv_name or not arguments.csv_name.lower().endswith('.csv'):
        parser.error('--csv-name must be a filename ending in .csv')
    arguments.input_dir = arguments.input_dir.resolve()
    arguments.output_dir = arguments.output_dir.resolve()
    arguments.checkpoint = (arguments.checkpoint or SUBMISSION_ROOT / 'weights' / 'best.pt').resolve()
    return arguments


def checkpoint_paths(arguments):
    """Use the promoted residual model when it is bundled beside the baseline."""
    paths = [arguments.checkpoint]
    if arguments.checkpoint == (SUBMISSION_ROOT / 'weights' / 'best.pt').resolve():
        if RESIDUAL_CHECKPOINT.is_file():
            paths.append(RESIDUAL_CHECKPOINT)
    return paths


def list_images(input_directory):
    """Return every input PNG once in stable filename order."""
    if not input_directory.is_dir():
        raise ValueError(f'Input directory does not exist: {input_directory}')
    images = sorted((path for path in input_directory.iterdir()
                     if path.is_file() and path.suffix.lower() == '.png'),
                    key=lambda path: path.name.casefold())
    names = [path.name.casefold() for path in images]
    if not images or len(names) != len(set(names)):
        raise ValueError('Input must contain uniquely named PNG images')
    return images


def read_image(path):
    """Decode one full seal image."""
    gray = read_gray(path)
    if gray is None:
        raise ValueError(f'Cannot decode {path.name}')
    return gray


def prepare_image(gray, path):
    """Locate and convert one seal into seven ordered model crops."""
    localization = find_digits(gray)
    if localization is None:
        raise ValueError(f'Cannot locate seven digits in {path.name}')
    return prepare_crops(gray, localization)


def fallback_localization(gray, polarity):
    """Describe the stable seven-cell seal band when component localization fails."""
    height, width = gray.shape
    box_width = max(1, round(width * FALLBACK_BOX_WIDTH))
    box_height = max(1, round(height * FALLBACK_BOX_HEIGHT))
    top = min(height - box_height, round(height * FALLBACK_BOX_TOP))
    boxes = []
    for index in range(DIGIT_COUNT):
        center = width * (FALLBACK_FIRST_CENTER + index * FALLBACK_CENTER_STEP)
        left = min(width - box_width, max(0, round(center - box_width / 2)))
        boxes.append((left, top, box_width, box_height))
    return Localization(tuple(boxes), polarity, 0.0, float('inf'), 'fixed_layout')


def predict_fallback(gray, model, device):
    """Try both crop polarities and keep the more confident fixed-layout code."""
    candidates = []
    for polarity in ('dark', 'light'):
        location = fallback_localization(gray, polarity)
        crops = prepare_crops(gray, location)[None, ...]
        digits, confidence = predict_crops(model, crops, device, batch_size=1)
        candidates.append((float(confidence[0]), digits[0]))
    return max(candidates, key=lambda candidate: candidate[0])[1]


def predict_image(path, model, device):
    """Predict one seal, using the fixed layout only if localization fails."""
    gray = read_image(path)
    try:
        crops = prepare_image(gray, path)[None, ...]
        digits, _ = predict_crops(model, crops, device, batch_size=1)
        return digits[0]
    except ValueError as error:
        LOGGER.warning('%s; using fixed-layout fallback', error)
        return predict_fallback(gray, model, device)


def predict_images(paths, model, device):
    """Fully process one seal before reading the next seal."""
    predictions = []
    for index, path in enumerate(paths, start=1):
        try:
            predictions.append(predict_image(path, model, device))
        except Exception as error:
            LOGGER.error('%s failed (%s); writing deterministic fallback', path.name, error)
            predictions.append(FALLBACK_NUMBER)
        if index % 500 == 0:
            LOGGER.info('Processed %d/%d images', index, len(paths))
    return predictions


def write_predictions(output_path, paths, predictions):
    """Write the required CSV atomically after every image has a seven-digit result."""
    if output_path.exists():
        raise ValueError(f'Refusing to overwrite existing result: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, delimiter=';')
            writer.writerow(CSV_COLUMNS)
            for path, digits in zip(paths, predictions, strict=True):
                number = ''.join(str(int(digit)) for digit in digits)
                writer.writerow((path.name, number))
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run(arguments):
    """Run localization and recognition, then create exactly one result CSV."""
    start = perf_counter()
    output_path = arguments.output_dir / arguments.csv_name
    if output_path.exists():
        raise ValueError(f'Refusing to overwrite existing result: {output_path}')
    device = select_device(arguments.device)
    paths = list_images(arguments.input_dir)
    models = load_models(checkpoint_paths(arguments), device)
    LOGGER.info('Loaded %d recognition model(s) on %s', len(models), device)
    predictions = predict_images(paths, models, device)
    if device.type == 'cuda':
        import torch
        torch.cuda.synchronize(device)
    write_predictions(output_path, paths, predictions)
    elapsed = perf_counter() - start
    LOGGER.info('Wrote %d predictions to %s in %.3f seconds', len(paths), output_path, elapsed)


def main():
    """Run the CLI and return a nonzero status without inventing failed predictions."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    cv2.setNumThreads(1)
    try:
        run(parse_arguments())
        return 0
    except Exception:
        LOGGER.exception('Submission failed; no incomplete prediction CSV was accepted')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
