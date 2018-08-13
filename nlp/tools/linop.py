"""Utilities for constructing specialized linear operators."""
import numpy as np
from scipy.sparse.linalg import LinearOperator

def ReducedLinearOperator(op, row_indices, col_indices):
    """
    Reduce a linear operator by limiting its input to `col_indices` and its
    output to `row_indices`. `op` is assumed to be a LinearOperator as defined
    in the scipy.sparse.linalg module.
    """

    nargin, nargout = len(col_indices), len(row_indices)
    m, n = op.shape    # Shape of non-reduced operator.

    def matvec(x):
        z = np.zeros(n, dtype=x.dtype) ; z[col_indices] = x[:]
        y = op * z
        return y[row_indices]

    def rmatvec(x):
        z = np.zeros(m, dtype=x.dtype) ; z[row_indices] = x[:]
        y = op.T * z
        return y[col_indices]

    return LinearOperator((nargout, nargin),
                          matvec=matvec,
                          rmatvec=rmatvec,
                          dtype=op.dtype)


def SymmetricallyReducedLinearOperator(op, indices):
    """
    Reduce a linear operator symmetrically by reducing boths its input and
    output to `indices`.
    """

    nargin = len(indices)
    m, n = op.shape    # Shape of non-reduced operator.

    def matvec(x):
        z = np.zeros(n, dtype=x.dtype) ; z[indices] = x[:]
        y = op * z
        return y[indices]

    def rmatvec(x):
        z = np.zeros(m, dtype=x.dtype) ; z[indices] = x[:]
        y = op.T * z
        return y[indices]

    return LinearOperator((nargin, nargin),
                          matvec=matvec,
                          rmatvec=rmatvec,
                          dtype=op.dtype)
