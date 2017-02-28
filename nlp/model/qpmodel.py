# -*- coding: utf-8 -*-
"""Base classes for quadratic optimization models."""

import logging
import os
import sys
import numpy as np
from nlp.model.kkt import KKTresidual
from nlp.tools.decorators import deprecated, counter
from nlp.tools.utils import where
from pykrylov.linop.linop import LinearOperator, DiagonalOperator, \
    ReducedLinearOperator
from pykrylov.linop.blkop import BlockLinearOperator


class QPModel(NLPModel):
    u"""Generic class to represent a quadratic programming (QP) problem.

    minimize    cᵀx + 1/2 xᵀHx
    subject to  L ≤ A x ≤ U
                l ≤ x ≤ u.
    """

    def __init__(self, c, H, A=None, name='GenericQP', **kwargs):
        """Initialize a QP with linear term `c`, Hessian `H` and Jacobian `A`.

        :parameters:
            :c:   Numpy array to represent the linear objective
            :A:   linear operator to represent the constraint matrix.
                  It must be possible to perform the operations `A*x`
                  and `A.T*y` for Numpy arrays `x` and `y` of appropriate size.
                  If `A` is `None`, it will be replaced with an empty linear
                  operator.
            :H:   linear operator to represent the objective Hessian.
                  It must be possible to perform the operation `H*x`
                  for a Numpy array `x` of appropriate size. The operator `H`
                  should be symmetric.

        See the documentation of `NLPModel` for futher information.
        """
        # Basic checks.
        n = c.shape[0]
        if A is None:
            m = 0
            self.A = LinearOperator(n, 0,
                                    lambda x: np.empty((0, 1)),
                                    matvec_transp=lambda y: np.empty((n, 0)),
                                    dtype=np.float)
        else:
            if A.shape[1] != n or H.shape[0] != n or H.shape[1] != n:
                raise ValueError('Shapes are inconsistent')
            m = A.shape[0]
            self.A = A

        super(QPModel, self).__init__(n=n, m=m, name=name, **kwargs)
        self.c = c
        self.H = H

        # Default classification of constraints
        self._lin = range(self.m)             # Linear    constraints
        self._nln = []                        # Nonlinear constraints
        self._net = []                        # Network   constraints
        self._nlin = len(self.lin)            # Number of linear constraints
        self._nnln = len(self.nln)            # Number of nonlinear constraints
        self._nnet = len(self.net)            # Number of network constraints

    def obj(self, x):
        """Evaluate the objective function at x."""
        cHx = self.hprod(x, 0, x)
        cHx *= 0.5
        cHx += self.c
        return np.dot(cHx, x)

    def grad(self, x):
        """Evaluate the objective gradient at x."""
        Hx = self.hprod(x, 0, x)
        Hx += self.c
        return Hx

    def cons(self, x):
        """Evaluate the constraints at x."""
        if isinstance(self.A, np.ndarray):
            return np.dot(self.A, x)
        return self.A * x

    def A(self, x):
        """Evaluate the constraints Jacobian at x."""
        return self.A

    def jac(self, x):
        """Evaluate the constraints Jacobian at x."""
        return self.A

    def jprod(self, x, p):
        """Evaluate Jacobian-vector product at x with p."""
        return self.cons(p)

    def jtprod(self, x, p):
        """Evaluate transposed-Jacobian-vector product at x with p."""
        if isinstance(self.A, np.ndarray):
            return np.dot(self.A.T, p)
        return self.A.T * p

    def hess(self, x, z):
        """Evaluate Lagrangian Hessian at (x, z)."""
        return self.H

    def hprod(self, x, z, p):
        """Hessian-vector product.

        Evaluate matrix-vector product between the Hessian of the Lagrangian at
        (x, z) and p.
        """
        if isinstance(self.H, np.ndarray):
            return np.dot(self.H, p)
        return self.H * p


