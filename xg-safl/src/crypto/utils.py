"""Low-level cryptographic utilities for the XG-SAFL protocol."""

from __future__ import annotations

import hashlib
import math


# ---------------------------------------------------------------------------
# Pre-computed 1024-bit safe prime for Proof-of-Concept use.
# p is safe-prime  (p = 2q + 1 where q is also prime).
# This is the 1024-bit MODP group prime from RFC 2409 / RFC 5114 §2.1.
# ---------------------------------------------------------------------------
SAFE_PRIME_1024 = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE65381"
    "FFFFFFFFFFFFFFFF",
    16,
)


def hash_to_group(data: bytes, modulus: int) -> int:
    """Map arbitrary data to a group element via 8× chained SHA-256.

    The digest is iteratively re-hashed eight times and the intermediate
    results are concatenated, producing a 256-byte (2048-bit) value that
    is then reduced modulo *modulus* to yield a uniformly distributed
    group element.

    Args:
        data: Arbitrary bytes to hash.
        modulus: Group modulus *n* (or *n²*).

    Returns:
        An integer in [0, modulus).

    Raises:
        ValueError: If *modulus* is less than 2.
    """
    if modulus < 2:
        raise ValueError("modulus must be >= 2")

    digest = hashlib.sha256(data).digest()
    parts: list[bytes] = [digest]
    for _ in range(7):
        digest = hashlib.sha256(digest).digest()
        parts.append(digest)

    combined = b"".join(parts)  # 8 × 32 = 256 bytes
    return int.from_bytes(combined, byteorder="big") % modulus


def mod_pow(base: int, exp: int, mod: int) -> int:
    """Compute *base* ** *exp* mod *mod* using Python's built-in three-arg pow.

    Args:
        base: The base integer.
        exp: The exponent (may be negative if *base* is invertible mod *mod*).
        mod: The modulus (must be > 0).

    Returns:
        The result of modular exponentiation.

    Raises:
        ValueError: If *mod* is not positive.
    """
    if mod <= 0:
        raise ValueError("mod must be positive")
    if exp < 0:
        base = mod_inv(base, mod)
        exp = -exp
    return pow(base, exp, mod)


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm returning (gcd, x, y) with a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = _extended_gcd(b % a, a)
    return gcd, y1 - (b // a) * x1, x1


def mod_inv(a: int, mod: int) -> int:
    """Compute the modular multiplicative inverse of *a* modulo *mod*.

    Uses the extended Euclidean algorithm.

    Args:
        a: The integer to invert.
        mod: The modulus (must be > 1).

    Returns:
        An integer *x* in [0, mod) such that (a * x) % mod == 1.

    Raises:
        ValueError: If *a* and *mod* are not coprime or *mod* < 2.
    """
    if mod < 2:
        raise ValueError("mod must be >= 2")
    a = a % mod
    gcd, x, _ = _extended_gcd(a, mod)
    if gcd != 1:
        raise ValueError(f"{a} has no inverse modulo {mod} (gcd={gcd})")
    return x % mod


def generate_safe_prime(bits: int) -> int:
    """Return a pre-computed safe prime for Proof-of-Concept use.

    In a production system this would generate a fresh safe prime of the
    requested bit-length.  For the PoC we return the well-known 1024-bit
    MODP safe prime (RFC 2409) regardless of the *bits* argument, which
    is retained for API compatibility.

    Args:
        bits: Desired bit-length (used only for documentation; the PoC
              always returns the 1024-bit prime).

    Returns:
        A 1024-bit safe prime integer.
    """
    return SAFE_PRIME_1024


def discrete_log_brute(
    target: int,
    base: int,
    modulus: int,
    max_val: int = 10_000,
) -> int:
    """Recover *x* such that ``base ** x ≡ target (mod modulus)`` by brute force.

    The search covers *x* in the range ``[0, max_val]`` as well as the
    corresponding negative range ``[-max_val, -1]`` (represented as
    ``modulus - x``).  This is sufficient for small aggregated model
    updates.

    Args:
        target: The group element whose discrete log is sought.
        base: The generator.
        modulus: The group modulus.
        max_val: Upper bound of the positive search range.

    Returns:
        The discrete logarithm *x*.

    Raises:
        ValueError: If no solution is found within the search range.
    """
    target = target % modulus
    acc = 1  # base^0
    for x in range(max_val + 1):
        if acc == target:
            return x
        acc = (acc * base) % modulus

    # Negative range: base^(-x) ≡ target ⟹ x = -k
    inv_base = mod_inv(base, modulus)
    acc = inv_base  # base^(-1)
    for x in range(1, max_val + 1):
        if acc == target:
            return -x
        acc = (acc * inv_base) % modulus

    raise ValueError(
        f"Discrete log not found in range [-{max_val}, {max_val}]"
    )
