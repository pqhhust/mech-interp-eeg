"""
Utility functions for tasks.
"""

from typing import List, Optional


# Default 22-channel BCIC2a montage
BCIC2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]


def parse_channel_names(s: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated channel names string."""
    if not s:
        return None
    return [ch.strip() for ch in s.split(",")]
