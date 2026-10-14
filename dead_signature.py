"""Project Override: Root-level shim for dead_signature."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "project_override"))
from project_override.core.dead_signature import *
