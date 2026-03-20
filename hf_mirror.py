import os
from pathlib import Path
from typing import Final


DEFAULT_HF_ENDPOINT: Final[str] = 'https://hf-mirror.com'
DEFAULT_HF_HOME: Final[str] = str(Path.home() / '.cache' / 'huggingface')
_HF_MIRROR_LOGGED = False


def setup_hf_mirror(verbose: bool = True) -> None:
    global _HF_MIRROR_LOGGED

    os.environ.setdefault('HF_ENDPOINT', DEFAULT_HF_ENDPOINT)
    os.environ['HF_ENDPOINT'] = os.environ['HF_ENDPOINT'].rstrip('/')

    hf_home = os.environ.setdefault('HF_HOME', DEFAULT_HF_HOME)
    hf_hub_cache = os.environ.setdefault(
        'HF_HUB_CACHE', os.path.join(hf_home, 'hub'))
    os.environ.setdefault('HUGGINGFACE_HUB_CACHE', hf_hub_cache)
    os.environ.setdefault('TRANSFORMERS_CACHE', hf_hub_cache)
    os.environ.setdefault('HUGGINGFACE_HUB_ENDPOINT', os.environ['HF_ENDPOINT'])

    if verbose and not _HF_MIRROR_LOGGED:
        print(f"[HF Mirror] HF_ENDPOINT={os.environ['HF_ENDPOINT']}")
        print(f"[HF Mirror] HF_HOME={os.environ['HF_HOME']}")
        print(f"[HF Mirror] HF_HUB_CACHE={os.environ['HF_HUB_CACHE']}")
        _HF_MIRROR_LOGGED = True
