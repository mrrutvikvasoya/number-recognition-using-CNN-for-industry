import torch


def select_device(requested):
    """Resolve automatic device selection and reject unavailable CUDA requests."""
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; check PyTorch and the NVIDIA driver')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(requested)
