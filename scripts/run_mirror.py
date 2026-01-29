#!/usr/bin/env python3
"""
Run the ATOM Mirror System.
Robot copies your boxing movements in real-time.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.mirror.atom import main

if __name__ == "__main__":
    main()
