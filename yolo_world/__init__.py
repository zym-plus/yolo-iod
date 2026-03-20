# Copyright (c) Tencent Inc. All rights reserved.
import importlib.metadata as importlib_metadata

from hf_mirror import configure_hf_mirror

try:
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = '0.0.0'

configure_hf_mirror()

from .models import *  # noqa
from .datasets import *  # noqa
from .engine import *  # noqa
