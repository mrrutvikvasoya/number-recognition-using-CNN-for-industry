from torch import nn

from cnn.residual_model import build_residual_model
from common.crops import CROP_SIDE

# Decimal digits are the only recognition classes; distractors are handled by localization.
CLASS_COUNT = 10
# Three pooling stages reduce 64-pixel crops to an 8-pixel feature grid.
CHANNELS = (16, 32, 64)
# A modest hidden layer limits the baseline's parameter count.
HIDDEN_FEATURES = 128
# Regularize the classifier without dropping pixels from the input strokes.
DROPOUT_RATE = 0.20
# Checkpoint names prevent weights from being loaded into the wrong architecture.
BASELINE_ARCHITECTURE = 'digit_cnn_v1'
RESIDUAL_ARCHITECTURE = 'residual_cnn_v1'
ARCHITECTURES = (BASELINE_ARCHITECTURE, RESIDUAL_ARCHITECTURE)


def build_baseline_model():
    """Build the original compact classifier retained for ensemble inference."""
    layers = []
    input_channels = 1
    for output_channels in CHANNELS:
        layers.extend((nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                       nn.ReLU(), nn.MaxPool2d(2)))
        input_channels = output_channels
    feature_side = CROP_SIDE // (2 ** len(CHANNELS))
    layers.extend((nn.Flatten(), nn.Linear(CHANNELS[-1] * feature_side ** 2, HIDDEN_FEATURES),
                   nn.ReLU(), nn.Dropout(DROPOUT_RATE), nn.Linear(HIDDEN_FEATURES, CLASS_COUNT)))
    return nn.Sequential(*layers)


def build_model(architecture=BASELINE_ARCHITECTURE):
    """Construct the architecture recorded in a training checkpoint."""
    if architecture == BASELINE_ARCHITECTURE:
        return build_baseline_model()
    if architecture == RESIDUAL_ARCHITECTURE:
        return build_residual_model(CLASS_COUNT)
    raise ValueError(f'Unknown model architecture: {architecture}')
