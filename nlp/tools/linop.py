"""Utilities for constructing specialized linear operators."""
import logging
import numpy as np
from scipy.sparse.linalg import LinearOperator as LinOp
from scipy.sparse.linalg import aslinearoperator as asLinOp

__docformat__ = 'restructuredtext'

# Default (null) logger.
null_log = logging.getLogger('linop')
null_log.setLevel(logging.INFO)
null_log.addHandler(logging.NullHandler())


class ShapeError(Exception):
    """
    Exception raised when defining a linear operator of the wrong shape or
    multiplying a linear operator with a vector of the wrong shape.
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class LinearOperator(LinOp):
    """
    This is a wrapper class to the Scipy LinearOperator that adds a few
    useful capabilities for working in NLP.py. (Some of them may even
    get merged upstream at a later date.)
    """
    def __init__(self, shape, matvec, rmatvec=None, dtype=np.float, **kwargs):
        super(LinearOperator, self).__init__(dtype, shape)

        self.__nargin = shape[1]
        self.__nargout = shape[0]
        self.__symmetric = kwargs.get('symmetric', False)
        self.__hermitian = kwargs.get('hermitian', False)
        self._nMatvec = 0

        self.__matvec_impl = matvec
        self.__rmatvec_impl = rmatvec

        # Log activity.
        self.logger = kwargs.get('logger', null_log)
        self.logger.info('New linear operator with shape ' + str(self.shape))

    @property
    def nargin(self):
        """The size of an input vector."""
        return self.__nargin

    @property
    def nargout(self):
        """The size of an output vector."""
        return self.__nargout

    @property
    def symmetric(self):
        """Indicates whether the operator is symmetric."""
        return self.__symmetric

    @property
    def hermitian(self):
        """Indicates whether the operator is Hermitian."""
        return self.__hermitian

    @property
    def nMatvec(self):
        """The number of products with vectors computed so far."""
        return self._nMatvec

    def reset_counter(self):
        """Reset operator/vector product counter to zero."""
        self._nMatvec = 0

    def to_array(self):
        """Convert operator to a dense matrix."""
        n, m = self.shape
        H = np.empty((n, m), dtype=self.dtype)
        e = np.zeros(m, dtype=self.dtype)
        for j in xrange(m):
            e[j] = 1
            H[:, j] = self * e
            e[j] = 0
        return H

    def full(self):
        """Convert operator to a dense matrix. This is an alias of `to_array`."""
        return self.to_array()

    def _matvec(self, x):
        """
        Matrix-vector multiplication.

        Encapsulates the matvec routine specified at
        construct time, to ensure the consistency of the input and output
        arrays with the operator's shape.
        """
        x = np.asanyarray(x)

        # check input data consistency
        try:
            x = x.reshape(self.__nargin)
        except ValueError:
            msg = 'input array size incompatible with operator dimensions'
            raise ValueError(msg)

        y = self.__matvec_impl(x)

        # check output data consistency
        try:
            y = y.reshape(self.__nargout)
        except ValueError:
            msg = 'output array size incompatible with operator dimensions'
            raise ValueError(msg)

        return y

    def _rmatvec(self, x):
        """
        Vector-matrix multiplication.
        (Equivalently, transpose matrix-vector multiplication.)

        Encapsulates the rmatvec routine specified at construct time, (if
        available,) to ensure the consistency of the input and output
        arrays with the operator's shape.
        """
        x = np.asanyarray(x)

        # check input data consistency
        try:
            x = x.reshape(self.__nargout)
        except ValueError:
            msg = 'input array size incompatible with operator dimensions'
            raise ValueError(msg)

        if self.__rmatvec_impl is None:
            raise NotImplementedError("rmatvec is not defined")
        y = self.__rmatvec_impl(x)

        # check output data consistency
        try:
            y = y.reshape(self.__nargin)
        except ValueError:
            msg = 'output array size incompatible with operator dimensions'
            raise ValueError(msg)

        return y

    def _transpose(self):
        """
        Note: This function is NOT implemented in the base class. Only
        the definition of self.transpose() and the property self.T are.
        """
        return LinearOperator((self.shape[1], self.shape[0]),
                              self.__rmatvec_impl,
                              rmatvec=self.__matvec_impl,
                              dtype=self.dtype,
                              symmetric=self.__symmetric,
                              hermitian=self.__hermitian)


def asLinearOperator(op):
    """
    Convert a scipy LinearOperator to a specialized LinearOperator.

    If op is a compatible type (e.g., ndarray or scipy sparse matrix,) use
    the scipy aslinearoperator() function to coerce it into a scipy
    LinearOperator first.
    """
    if not isinstance(op, LinOp):
        op = asLinOp(op)

    new_op = LinearOperator(op.shape,
                            op.matvec,
                            rmatvec=op.rmatvec,
                            dtype=op.dtype)
    return new_op


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
                          matvec,
                          rmatvec=rmatvec,
                          dtype=op.dtype,
                          symmetric=False,
                          hermitian=False)


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
                          matvec,
                          rmatvec=rmatvec,
                          dtype=op.dtype,
                          symmetric=op.symmetric,
                          hermitian=op.hermitian)
