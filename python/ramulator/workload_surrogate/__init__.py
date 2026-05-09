"""Structured workload-surrogate trace generators."""

from ramulator.workload_surrogate.decode_only_manifest import DECODE_ONLY_MANIFEST, get_decode_only_manifest

__all__ = ["DECODE_ONLY_MANIFEST", "get_decode_only_manifest", "generate_decode_only_artifacts"]


def generate_decode_only_artifacts(*args, **kwargs):
    from ramulator.workload_surrogate.generate_decode_only import generate_decode_only_artifacts as _impl

    return _impl(*args, **kwargs)
