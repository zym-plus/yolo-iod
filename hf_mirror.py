import os


DEFAULT_HF_ENDPOINT = 'https://hf-mirror.com'


def configure_hf_mirror():
    endpoint = os.environ.get('HF_MIRROR_ENDPOINT', DEFAULT_HF_ENDPOINT).rstrip('/')
    os.environ.setdefault('HF_ENDPOINT', endpoint)
    os.environ.setdefault('HUGGINGFACE_HUB_ENDPOINT', endpoint)
    return endpoint


def mirror_hf_url(url: str) -> str:
    endpoint = configure_hf_mirror()
    return url.replace('https://huggingface.co', endpoint, 1)
