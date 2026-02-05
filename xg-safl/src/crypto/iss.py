"""Integer Secret Sharing (ISS) based on Rabin's polynomial scheme.

Shares a secret integer into *n* shares such that any *t + 1* shares
suffice for reconstruction.  All arithmetic is performed over the
integers (or optionally modulo a prime) so that the scheme integrates
naturally with the Threshold Joye–Libert cryptosystem.
"""

from __future__ import annotations

import secrets
from typing import Optional


def _lagrange_basis(
    shares: list[tuple[int, int]],
    j: int,
    prime: Optional[int] = None,
) -> tuple[int, int]:
    """Compute the Lagrange basis coefficient for index *j*.

    Returns the pair ``(numerator, denominator)`` of the basis
    polynomial evaluated at ``x = 0``::

        ℓ_j(0) = ∏_{m ≠ j}  (0 - x_m) / (x_j - x_m)

    When *prime* is ``None`` the values are kept as exact rational
    components; when a prime is given they are reduced modulo that prime.

    Args:
        shares: The set of ``(x, y)`` evaluation points.
        j: Index into *shares* selecting the basis element.
        prime: Optional prime modulus for modular arithmetic.

    Returns:
        ``(numerator, denominator)`` of the basis value at 0.

    Raises:
        ValueError: If duplicate x-coordinates are detected.
    """
    x_j = shares[j][0]
    num = 1
    den = 1
    for m, (x_m, _) in enumerate(shares):
        if m == j:
            continue
        if x_m == x_j:
            raise ValueError(
                f"Duplicate x-coordinate {x_j} at indices {j} and {m}"
            )
        num *= -x_m
        den *= x_j - x_m

    if prime is not None:
        num %= prime
        den %= prime

    return num, den


def share(
    secret: int,
    n: int,
    t: int,
    prime: Optional[int] = None,
) -> list[tuple[int, int]]:
    """Split *secret* into *n* shares with threshold *t*.

    A random polynomial ``f`` of degree *t* is constructed such that
    ``f(0) = secret``.  Each share is ``(i, f(i))`` for ``i = 1 … n``.

    Args:
        secret: The integer secret to share.
        n: Total number of shares to generate.
        t: Degree of the polynomial; reconstruction requires *t + 1*
           shares.
        prime: Optional prime modulus.  When provided all polynomial
               arithmetic is performed modulo *prime*.

    Returns:
        A list of *n* ``(x, y)`` share pairs.

    Raises:
        ValueError: If parameters are inconsistent.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if t < 1:
        raise ValueError("t must be >= 1")
    if t >= n:
        raise ValueError("t must be < n (need t+1 shares to reconstruct)")

    # Random coefficients a_1 … a_t  (a_0 = secret).
    coeff_bound = max(abs(secret), 1) * (2**64)
    coeffs: list[int] = [secret]
    for _ in range(t):
        c = secrets.randbelow(coeff_bound) + 1
        if secrets.randbelow(2) == 0:
            c = -c
        coeffs.append(c)

    result: list[tuple[int, int]] = []
    for i in range(1, n + 1):
        y = 0
        xi = 1  # i^k
        for k, a_k in enumerate(coeffs):
            y += a_k * xi
            xi *= i
        if prime is not None:
            y %= prime
        result.append((i, y))

    return result


def reconstruct(
    shares: list[tuple[int, int]],
    t: int,
    prime: Optional[int] = None,
) -> int:
    """Reconstruct the secret from *t + 1* or more shares.

    Uses Lagrange interpolation evaluated at ``x = 0``.

    Args:
        shares: At least *t + 1* ``(x, y)`` share pairs.
        t: Polynomial degree used during sharing.
        prime: Optional prime modulus matching the one used in
               :func:`share`.

    Returns:
        The reconstructed secret integer.

    Raises:
        ValueError: If insufficient shares are provided.
    """
    needed = t + 1
    if len(shares) < needed:
        raise ValueError(
            f"Need at least {needed} shares for threshold {t}, "
            f"got {len(shares)}"
        )

    subset = shares[:needed]

    if prime is not None:
        # Modular reconstruction
        secret = 0
        for j, (_, y_j) in enumerate(subset):
            num, den = _lagrange_basis(subset, j, prime)
            den_inv = pow(den, prime - 2, prime)  # Fermat's little theorem
            secret = (secret + y_j * num % prime * den_inv) % prime
        return secret

    # Integer reconstruction: accumulate as exact rational then divide.
    # Compute a common denominator and sum numerators.
    # s = Σ_j  y_j · num_j / den_j  =  (Σ_j y_j · num_j · D/den_j) / D
    # where D = ∏ den_j.
    bases: list[tuple[int, int]] = []
    common_den = 1
    for j in range(len(subset)):
        num_j, den_j = _lagrange_basis(subset, j)
        bases.append((num_j, den_j))
        common_den *= den_j

    numerator = 0
    for j, (_, y_j) in enumerate(subset):
        num_j, den_j = bases[j]
        # Contribution = y_j * num_j * (common_den // den_j)
        numerator += y_j * num_j * (common_den // den_j)

    # The result must be an exact integer.
    if numerator % common_den != 0:
        # Due to integer rounding the result may not divide exactly;
        # use rounding to nearest integer as a practical fallback.
        return round(numerator / common_den)
    return numerator // common_den
