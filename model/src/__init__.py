"""Piano Performance Evaluation - Source Package.

This package is organized into three main subpackages:
- percepiano: PercePiano SOTA replica implementation
- crescendai: Custom multi-modal evaluation models
- shared: Common utilities used by both
"""

from . import percepiano
from . import crescendai
from . import shared

__all__ = ["percepiano", "crescendai", "shared"]
