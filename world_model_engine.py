"""Project Override: Root-level shim for world_model_engine."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "project_override"))
from project_override.world_model.world_model_engine import *
