"""Structured workload-surrogate trace generators.

The ``decode_only`` surface is a legacy deprecated P2 MVP replay fixture. The
current Llama2 dense-decoder surrogate lives in ``generate_full_transformer``.
"""

from ramulator.workload_surrogate.decode_only_manifest import DECODE_ONLY_MANIFEST, get_decode_only_manifest

__all__ = ["DECODE_ONLY_MANIFEST", "get_decode_only_manifest", "generate_decode_only_artifacts"]


def generate_decode_only_artifacts(*args, **kwargs):
    from ramulator.workload_surrogate.generate_decode_only import generate_decode_only_artifacts as _impl

    return _impl(*args, **kwargs)
