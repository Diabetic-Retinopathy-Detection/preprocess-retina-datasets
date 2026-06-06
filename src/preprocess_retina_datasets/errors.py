from __future__ import annotations


class PreprocessError(Exception):
    """Base exception for all preprocessing errors."""

    __slots__ = ()


class ImageProcessingError(PreprocessError):
    """Raised when a single image cannot be processed."""

    __slots__ = ()
