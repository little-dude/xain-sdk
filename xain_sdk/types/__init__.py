from typing import Dict, List, Tuple

from numpy import ndarray

Theta = List[ndarray]
History = Dict[str, List[float]]

VolumeByClass = List[int]
Metrics = Tuple[int, VolumeByClass]
