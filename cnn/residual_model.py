from torch import nn

# Wider late stages retain capacity for worn and low-contrast digit shapes.
RESIDUAL_CHANNELS = (32, 64, 128, 256)
# Mild classifier dropout regularizes without deleting input strokes.
RESIDUAL_DROPOUT = 0.10


def normalization(channels):
    """Use group normalization so training remains stable with small GPU batches."""
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class ResidualBlock(nn.Module):
    """Learn two convolutions while retaining a direct feature path."""

    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            normalization(output_channels), nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            normalization(output_channels))
        self.shortcut = nn.Identity()
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                normalization(output_channels))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images):
        """Combine learned and direct paths before the next stage."""
        return self.activation(self.features(images) + self.shortcut(images))


class ResidualDigitCNN(nn.Module):
    """Classify a grayscale digit using four compact residual stages."""

    def __init__(self, class_count):
        super().__init__()
        channels = RESIDUAL_CHANNELS
        self.stem = nn.Sequential(nn.Conv2d(1, channels[0], 3, padding=1, bias=False),
                                  normalization(channels[0]), nn.ReLU(inplace=True))
        blocks = [ResidualBlock(channels[0], channels[0])]
        for previous, current in zip(channels[:-1], channels[1:], strict=True):
            blocks.extend((ResidualBlock(previous, current, stride=2), ResidualBlock(current, current)))
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                        nn.Dropout(RESIDUAL_DROPOUT), nn.Linear(channels[-1], class_count))

    def forward(self, images):
        """Return unnormalized scores for the ten digit classes."""
        return self.classifier(self.features(self.stem(images)))


def build_residual_model(class_count):
    """Construct the residual classifier through the shared model factory."""
    return ResidualDigitCNN(class_count)
