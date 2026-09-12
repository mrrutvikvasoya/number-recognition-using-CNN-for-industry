import numpy as np
import torch

from cnn.model import build_model
from common.config import DIGIT_COUNT
from common.crops import CROP_SIDE


def load_model(checkpoint_path, device):
    """Load a compatible digit checkpoint and switch the model to inference mode."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    architecture = checkpoint.get('architecture')
    if checkpoint.get('crop_side') != CROP_SIDE:
        raise ValueError('Checkpoint architecture or crop size is incompatible')
    state = checkpoint.get('model_state')
    if not isinstance(state, dict):
        raise ValueError('Checkpoint has no model state')
    model = build_model(architecture).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_models(checkpoint_paths, device):
    """Load one or more compatible classifiers for averaged-logit inference."""
    if not checkpoint_paths:
        raise ValueError('At least one checkpoint is required')
    return tuple(load_model(path, device) for path in checkpoint_paths)


def ensemble_logits(models, images):
    """Average model scores so each classifier contributes equally."""
    model_list = models if isinstance(models, (tuple, list)) else (models,)
    if not model_list:
        raise ValueError('At least one model is required')
    logits = [model(images) for model in model_list]
    return torch.stack(logits).mean(dim=0)


def predict_crops(model, crops, device, batch_size):
    """Predict ordered codes and minimum digit confidence in bounded GPU batches."""
    if crops.ndim != 4 or crops.shape[1:] != (DIGIT_COUNT, CROP_SIDE, CROP_SIDE):
        raise ValueError('Expected crops shaped (seals, 7, 64, 64)')
    predicted, confidences = [], []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            images = torch.from_numpy(crops[start:start + batch_size].astype(np.float32) / 255.0)
            logits = ensemble_logits(model, images.unsqueeze(2).flatten(0, 1).to(device))
            probabilities = logits.softmax(dim=1)
            confidence, digits = probabilities.max(dim=1)
            predicted.extend(digits.reshape(-1, DIGIT_COUNT).cpu().numpy())
            confidences.extend(confidence.reshape(-1, DIGIT_COUNT).min(dim=1).values.cpu().numpy())
    return np.asarray(predicted), np.asarray(confidences)
