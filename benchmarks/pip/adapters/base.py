"""Base adapter class for spatial query approaches."""

import os
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

import numpy as np


class SpatialQueryAdapter(ABC):
    """Base class for spatial query approach adapters."""
    
    def __init__(self, name: str, workspace: str):
        self.name = name
        self.workspace = workspace
        os.makedirs(workspace, exist_ok=True)
    
    @abstractmethod
    def setup(self, **kwargs) -> bool:
        """One-time setup (build, initialize database, etc.)."""
        pass
    
    @abstractmethod
    def execute_query(
        self,
        geometry_path: str,
        points_path: str,
        grid_pos: Tuple[int, int, int],
        translation: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute query for given grid position.
        
        Returns dict with:
            - query_ms: Query time in milliseconds
            - inside_count: Number of points inside
            - total_points: Total number of points tested
            - success: Boolean indicating success
            - error: Error message if failed
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources."""
        pass
