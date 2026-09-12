import csv
import logging
from pathlib import Path

import cv2
import numpy as np

from common.config import DIGIT_COUNT

LOGGER = logging.getLogger(__name__)


def read_manifest(path: Path, image_directory: Path) -> tuple[list[tuple[Path, str]], int]:
    records, seen, skipped = [], set(), 0
    try:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream, delimiter=';', strict=True)
            if sorted(reader.fieldnames or []) != ['filename', 'number']:
                raise ValueError('Manifest must have filename;number columns')
            for row in reader:
                filename, number = row.get('filename') or '', row.get('number') or ''
                valid_name = Path(filename).name == filename and filename.lower().endswith('.png')
                valid_number = len(number) == DIGIT_COUNT and number.isascii() and number.isdigit()
                if not valid_name or not valid_number or None in row:
                    skipped += 1
                    LOGGER.warning('Skipping malformed manifest row %d: %r', reader.line_num, filename)
                    continue
                if filename.casefold() in seen:
                    raise ValueError(f'Duplicate manifest filename: {filename}')
                seen.add(filename.casefold())
                records.append((image_directory / filename, number))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f'Cannot read manifest {path}: {error}') from error
    return sorted(records, key=lambda record: record[0].name), skipped


def read_gray(path: Path) -> np.ndarray | None:
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError('Decoder returned no image')
        return image
    except (OSError, ValueError, cv2.error) as error:
        LOGGER.error('Cannot decode %s: %s', path, error)
        return None


def save_jpeg(path: Path, image: np.ndarray) -> bool:
    return save_image(path, image, '.jpg')


def save_image(path: Path, image: np.ndarray, extension: str) -> bool:
    try:
        success, encoded = cv2.imencode(extension, image)
        if not success:
            raise ValueError(f'{extension} encoding failed')
        encoded.tofile(path)
        return True
    except (OSError, ValueError, cv2.error) as error:
        LOGGER.error('Cannot save %s: %s', path, error)
        return False
