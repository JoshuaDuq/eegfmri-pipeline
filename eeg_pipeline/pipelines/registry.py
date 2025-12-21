"""
Pipeline Registry
==================

Centralized registry for pipeline discovery and instantiation.
Uses a decorator pattern for automatic registration.
"""

from __future__ import annotations

from typing import Dict, Type, List, Optional, Any, Callable
from functools import wraps

from eeg_pipeline.pipelines.base import PipelineBase
from eeg_pipeline.exceptions import PipelineNotFoundError


class PipelineRegistry:
    """
    Central registry for all pipeline classes.
    
    Usage:
        @PipelineRegistry.register("features")
        class FeaturePipeline(PipelineBase):
            ...
        
        pipeline = PipelineRegistry.get("features")
        pipeline.run_batch(subjects, task="mytask")
    """
    
    _pipelines: Dict[str, Type[PipelineBase]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        description: Optional[str] = None,
        requires_epochs: bool = True,
        requires_features: bool = False,
    ) -> Callable[[Type[PipelineBase]], Type[PipelineBase]]:
        """
        Decorator to register a pipeline class.
        
        Args:
            name: Unique identifier for the pipeline
            description: Human-readable description
            requires_epochs: Whether pipeline needs epoched data
            requires_features: Whether pipeline needs extracted features
        
        Returns:
            Decorator function
        """
        def decorator(pipeline_cls: Type[PipelineBase]) -> Type[PipelineBase]:
            if name in cls._pipelines:
                raise ValueError(f"Pipeline '{name}' is already registered")
            
            cls._pipelines[name] = pipeline_cls
            cls._metadata[name] = {
                "description": description or pipeline_cls.__doc__ or "",
                "requires_epochs": requires_epochs,
                "requires_features": requires_features,
                "class_name": pipeline_cls.__name__,
            }
            
            pipeline_cls._registry_name = name
            
            return pipeline_cls
        
        return decorator
    
    @classmethod
    def get(cls, name: str, config: Optional[Any] = None) -> PipelineBase:
        """
        Get an instance of a registered pipeline.
        
        Args:
            name: Pipeline identifier
            config: Optional configuration to pass to pipeline
        
        Returns:
            Instantiated pipeline
        
        Raises:
            PipelineNotFoundError: If pipeline is not registered
        """
        if name not in cls._pipelines:
            raise PipelineNotFoundError(name, available=cls.list_names())
        
        pipeline_cls = cls._pipelines[name]
        return pipeline_cls(config=config) if config else pipeline_cls()
    
    @classmethod
    def get_class(cls, name: str) -> Type[PipelineBase]:
        """
        Get the class of a registered pipeline (without instantiation).
        
        Args:
            name: Pipeline identifier
        
        Returns:
            Pipeline class
        """
        if name not in cls._pipelines:
            raise PipelineNotFoundError(name, available=cls.list_names())
        return cls._pipelines[name]
    
    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered pipeline names."""
        return list(cls._pipelines.keys())
    
    @classmethod
    def list_all(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered pipelines with metadata.
        
        Returns:
            Dict mapping pipeline names to their metadata
        """
        return {
            name: {
                "class": cls._pipelines[name],
                **cls._metadata[name],
            }
            for name in cls._pipelines
        }
    
    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Get metadata for a specific pipeline."""
        if name not in cls._metadata:
            raise PipelineNotFoundError(name, available=cls.list_names())
        return cls._metadata[name]
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a pipeline is registered."""
        return name in cls._pipelines
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered pipelines (mainly for testing)."""
        cls._pipelines.clear()
        cls._metadata.clear()
    
    @classmethod
    def pipelines_requiring_features(cls) -> List[str]:
        """Get names of pipelines that require pre-extracted features."""
        return [
            name for name, meta in cls._metadata.items()
            if meta.get("requires_features", False)
        ]
    
    @classmethod
    def pipelines_requiring_epochs(cls) -> List[str]:
        """Get names of pipelines that require epoched data."""
        return [
            name for name, meta in cls._metadata.items()
            if meta.get("requires_epochs", False)
        ]


def register_pipeline(
    name: str,
    description: Optional[str] = None,
    requires_epochs: bool = True,
    requires_features: bool = False,
) -> Callable[[Type[PipelineBase]], Type[PipelineBase]]:
    """
    Convenience function for pipeline registration.
    
    Equivalent to @PipelineRegistry.register(...)
    """
    return PipelineRegistry.register(
        name=name,
        description=description,
        requires_epochs=requires_epochs,
        requires_features=requires_features,
    )


__all__ = [
    "PipelineRegistry",
    "register_pipeline",
]
