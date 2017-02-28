# -*- coding: utf-8 -*-
"""Base classes for linear optimization models."""

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


class LPModel(QPModel):
    u"""Generic class to represent a linear programming (LP) problem.

    minimize    cᵀx
    subject to  L ≤ A x ≤ U
                l ≤ x ≤ u.
    """

    def __init__(self, c, A=None, name='GenericLP', **kwargs):
        """Initialize a LP with linear term `c` and Jacobian `A`.

        :parameters:
            :c:   Numpy array to represent the linear objective
            :A:   linear operator to represent the constraint matrix.
                  It must be possible to perform the operations `A*x`
                  and `A.T*y` for Numpy arrays `x` and `y` of appropriate size.

        See the documentation of `NLPModel` for futher information.
        """
        n = c.shape[0]
        H = LinearOperator(n, n,
                           lambda x: np.zeros(n),
                           symmetric=True,
                           dtype=np.float)
        super(LPModel, self).__init__(c, H, A, name=name, **kwargs)

    def obj(self, x):
        """Evaluate the objective function at x."""
        return np.dot(self.c, x)

    def grad(self, x):
        """Evaluate the objective gradient at x."""
        return self.c


