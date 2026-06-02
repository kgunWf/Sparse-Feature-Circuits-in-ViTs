# Moved to utils/load_data.py
# This shim keeps `python data/load_data.py` working during transition.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.load_data import download_images, CLASSES, N_TARGET  # noqa: F401

if __name__ == "__main__":
    download_images()
    exit(0)
