"""Project Override: override package shim."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "project_override"))
from project_override.world_model import *
