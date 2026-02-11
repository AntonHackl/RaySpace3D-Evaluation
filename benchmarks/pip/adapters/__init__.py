"""Spatial query adapters for different approaches."""

from .base import SpatialQueryAdapter
from .cgal_adapter import CGALAdapter
from .sql_adapter import SQLAdapter
from .raytracer_adapter import RaytracerAdapter
from .filter_refine_adapter import FilterRefineAdapter
from .cuda_adapter import CUDAAdapter

__all__ = [
    'SpatialQueryAdapter',
    'CGALAdapter',
    'SQLAdapter',
    'RaytracerAdapter',
    'FilterRefineAdapter',
    'CUDAAdapter'
]
