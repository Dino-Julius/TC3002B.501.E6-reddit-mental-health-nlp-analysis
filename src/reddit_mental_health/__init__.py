"""Pipeline baseline para análisis experimental de suicidalidad en Reddit."""

from reddit_mental_health.config import BaselineConfig
from reddit_mental_health.model import BaselineModel

__version__ = "0.1.0"

__all__ = ["BaselineConfig", "BaselineModel", "__version__"]
