"""
Domain-Specific Exception Hierarchy
====================================

Custom exceptions for the EEG pipeline, providing clear error categorization
and context for debugging and error handling.
"""

from __future__ import annotations

from typing import Any, Optional


class EEGPipelineError(Exception):
    """Base exception for all EEG pipeline errors."""

    def __init__(self, message: str, context: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        context_items = ", ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message} [{context_items}]"


class ConfigurationError(EEGPipelineError):
    """Raised when configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        key: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
    ):
        context = {}
        if key:
            context["key"] = key
        if expected is not None:
            context["expected"] = expected
        if actual is not None:
            context["actual"] = actual
        super().__init__(message, context)
        self.key = key
        self.expected = expected
        self.actual = actual


class DataValidationError(EEGPipelineError):
    """Raised when data fails validation checks."""

    def __init__(
        self,
        message: str,
        subject: Optional[str] = None,
        validation_type: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        context = {}
        if subject:
            context["subject"] = subject
        if validation_type:
            context["validation"] = validation_type
        if details:
            context.update(details)
        super().__init__(message, context)
        self.subject = subject
        self.validation_type = validation_type


class SubjectProcessingError(EEGPipelineError):
    """Raised when processing a specific subject fails."""

    def __init__(
        self,
        subject: str,
        message: str,
        step: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        context = {"subject": subject}
        if step:
            context["step"] = step
        super().__init__(message, context)
        self.subject = subject
        self.step = step
        if cause is not None:
            self.cause = cause
            self.__cause__ = cause


class FeatureExtractionError(EEGPipelineError):
    """Raised when feature extraction fails."""

    def __init__(
        self,
        message: str,
        feature_type: Optional[str] = None,
        subject: Optional[str] = None,
        band: Optional[str] = None,
    ):
        context = {}
        if feature_type:
            context["feature"] = feature_type
        if subject:
            context["subject"] = subject
        if band:
            context["band"] = band
        super().__init__(message, context)
        self.feature_type = feature_type
        self.subject = subject
        self.band = band


class PreprocessingError(EEGPipelineError):
    """Raised when preprocessing steps fail."""

    def __init__(
        self,
        message: str,
        step: Optional[str] = None,
        subject: Optional[str] = None,
    ):
        context = {}
        if step:
            context["step"] = step
        if subject:
            context["subject"] = subject
        super().__init__(message, context)
        self.step = step
        self.subject = subject


class PipelineNotFoundError(EEGPipelineError):
    """Raised when a requested pipeline is not registered."""

    def __init__(self, pipeline_name: str, available: Optional[list[str]] = None):
        message = f"Pipeline '{pipeline_name}' not found"
        context = {"requested": pipeline_name}
        if available:
            context["available"] = available
        super().__init__(message, context)
        self.pipeline_name = pipeline_name
        self.available = available or []


class InsufficientDataError(DataValidationError):
    """Raised when there isn't enough data for analysis."""

    def __init__(
        self,
        message: str,
        required: int,
        actual: int,
        subject: Optional[str] = None,
        data_type: Optional[str] = None,
    ):
        details = {"required": required, "actual": actual}
        if data_type:
            details["data_type"] = data_type
        super().__init__(
            message,
            subject=subject,
            validation_type="insufficient_data",
            details=details,
        )
        self.required = required
        self.actual = actual


class FileFormatError(EEGPipelineError):
    """Raised when a file has an unexpected format."""

    def __init__(
        self,
        message: str,
        filepath: Optional[str] = None,
        expected_format: Optional[str] = None,
    ):
        context = {}
        if filepath:
            context["file"] = filepath
        if expected_format:
            context["expected"] = expected_format
        super().__init__(message, context)
        self.filepath = filepath
        self.expected_format = expected_format


__all__ = [
    "EEGPipelineError",
    "ConfigurationError",
    "DataValidationError",
    "SubjectProcessingError",
    "FeatureExtractionError",
    "PreprocessingError",
    "PipelineNotFoundError",
    "InsufficientDataError",
    "FileFormatError",
]
