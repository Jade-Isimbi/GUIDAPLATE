# config.py
# Root-level shim for backward compatibility.
# Notebooks that do `import config` will use this.
# All actual config is in backend/config.py

from backend.config import *
