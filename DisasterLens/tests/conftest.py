"""
Shared pytest configuration and fixtures for DisasterLens AI tests.
"""
import sys
from pathlib import Path

# Ensure project root is on sys.path for all test modules
sys.path.insert(0, str(Path(__file__).parent.parent))
