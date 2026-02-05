"""Adaptive precision encoding for SHAP-guided quantisation.

Features are assigned to three precision tiers based on their SHAP
importance ranking and then quantised / packed into fixed-width integer
chunks for efficient encrypted aggregation.
"""

from __future__ import annotations

import numpy as np


def adaptive_precision_map(
    importance: np.ndarray,
    k_ratio: float = 0.2,
    high_bits: int = 24,
    mid_bits: int = 16,
    low_bits: int = 8,
) -> np.ndarray:
    """Map SHAP importance scores to per-feature bit-width allocations.

    The features are ranked by descending absolute importance and
    assigned to three tiers:

    * **Top 20 %** (high precision) → *high_bits* (default 24)
    * **Next 30 %** (medium precision) → *mid_bits* (default 16)
    * **Bottom 50 %** (low precision) → *low_bits* (default 8)

    Args:
        importance: 1-D array of SHAP importance values (one per feature).
        k_ratio: Fraction of features in the top tier (default 0.2).
        high_bits: Bit-width for the top tier.
        mid_bits: Bit-width for the middle tier.
        low_bits: Bit-width for the bottom tier.

    Returns:
        1-D integer array of bit-widths with the same length as
        *importance*.

    Raises:
        ValueError: If *importance* is empty or *k_ratio* is out of range.
    """
    if importance.size == 0:
        raise ValueError("importance array must not be empty")
    if not 0.0 < k_ratio < 1.0:
        raise ValueError("k_ratio must be in (0, 1)")

    n = len(importance)
    ranked_indices = np.argsort(-np.abs(importance))

    top_count = max(1, int(np.ceil(n * k_ratio)))
    mid_ratio = 0.3
    mid_count = max(1, int(np.ceil(n * mid_ratio)))

    bit_widths = np.full(n, low_bits, dtype=np.int32)
    bit_widths[ranked_indices[:top_count]] = high_bits
    mid_end = min(top_count + mid_count, n)
    bit_widths[ranked_indices[top_count:mid_end]] = mid_bits

    return bit_widths


def quantize_value(value: float, bits: int, scale: float = 1.0) -> int:
    """Quantise a floating-point value to a signed integer.

    The value is first divided by *scale*, then clamped to the
    representable range ``[-(2^{bits-1}), 2^{bits-1} - 1]`` and
    rounded to the nearest integer.

    Args:
        value: The floating-point value to quantise.
        bits: Number of bits for the quantised representation.
        scale: Scaling factor applied before quantisation.

    Returns:
        A signed integer in the representable range.

    Raises:
        ValueError: If *bits* < 1 or *scale* is zero.
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")
    if scale == 0.0:
        raise ValueError("scale must be non-zero")

    max_val = (1 << (bits - 1)) - 1
    min_val = -(1 << (bits - 1))
    scaled = value / scale
    clamped = max(min_val, min(max_val, scaled))
    return int(round(clamped))


def dequantize_value(value: int, bits: int, scale: float = 1.0) -> float:
    """Inverse of :func:`quantize_value`.

    Args:
        value: Quantised integer.
        bits: Bit-width used during quantisation.
        scale: The same scaling factor used during quantisation.

    Returns:
        Reconstructed floating-point approximation.

    Raises:
        ValueError: If *bits* < 1 or *scale* is zero.
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")
    if scale == 0.0:
        raise ValueError("scale must be non-zero")
    return float(value) * scale


def pack_vector(
    values: np.ndarray,
    bit_widths: np.ndarray,
    chunk_bits: int = 512,
) -> list[int]:
    """Pack an array of quantised integers into fixed-width integer chunks.

    Each value is stored using the corresponding entry in *bit_widths*.
    Values are represented in two's-complement within their allocated
    width, then concatenated bit-by-bit and split into *chunk_bits*-wide
    Python integers (big-endian, most-significant chunk first).

    Args:
        values: 1-D array of quantised integers.
        bit_widths: Per-element bit-widths (same length as *values*).
        chunk_bits: Width of each output chunk in bits.

    Returns:
        List of non-negative Python integers, each < ``2**chunk_bits``.

    Raises:
        ValueError: If *values* and *bit_widths* differ in length.
    """
    if len(values) != len(bit_widths):
        raise ValueError(
            f"values length ({len(values)}) != bit_widths length "
            f"({len(bit_widths)})"
        )

    # Build a single big-endian bit-string as an integer.
    stream = 0
    total_bits = 0
    for v, bw in zip(values, bit_widths):
        bw = int(bw)
        mask = (1 << bw) - 1
        # Two's complement encoding
        encoded = int(v) & mask
        stream = (stream << bw) | encoded
        total_bits += bw

    # Split into chunk_bits-sized pieces.
    num_chunks = (total_bits + chunk_bits - 1) // chunk_bits
    # Pad stream on the right so total width is num_chunks * chunk_bits
    pad = num_chunks * chunk_bits - total_bits
    stream <<= pad

    chunk_mask = (1 << chunk_bits) - 1
    chunks: list[int] = []
    for _ in range(num_chunks):
        # Extract from the most significant end
        shift = (num_chunks - 1 - len(chunks)) * chunk_bits
        chunk = (stream >> shift) & chunk_mask
        chunks.append(chunk)

    return chunks


def unpack_vector(
    chunks: list[int],
    bit_widths: np.ndarray,
    chunk_bits: int = 512,
    length: int = 0,
) -> np.ndarray:
    """Unpack chunks produced by :func:`pack_vector` back to integers.

    Args:
        chunks: List of packed integer chunks (big-endian).
        bit_widths: Per-element bit-widths matching the original packing.
        chunk_bits: Width of each chunk in bits.
        length: Expected output length.  If 0 the length of *bit_widths*
                is used.

    Returns:
        1-D integer array of dequantised values (signed).

    Raises:
        ValueError: If *chunks* is empty or *bit_widths* is empty.
    """
    if not chunks:
        raise ValueError("chunks list must not be empty")
    n = length if length > 0 else len(bit_widths)
    if n == 0:
        raise ValueError("bit_widths must not be empty when length is 0")

    # Reconstruct the full bit-stream
    stream = 0
    for c in chunks:
        stream = (stream << chunk_bits) | int(c)

    total_packed = len(chunks) * chunk_bits
    total_bits = int(np.sum(bit_widths[:n]))
    pad = total_packed - total_bits

    # Remove right-padding
    stream >>= pad

    # Extract values in reverse order (least-significant first)
    result: list[int] = []
    for i in range(n - 1, -1, -1):
        bw = int(bit_widths[i])
        mask = (1 << bw) - 1
        encoded = stream & mask
        stream >>= bw
        # Decode two's complement
        if encoded >= (1 << (bw - 1)):
            encoded -= 1 << bw
        result.append(encoded)

    result.reverse()
    return np.array(result, dtype=np.int64)
