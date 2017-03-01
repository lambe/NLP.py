# -*- coding: utf-8 -*-
"""Base classes for quadratic optimization models."""

import logging
import numpy as np
from nlp.model.kkt import KKTresidual
from nlp.model.nlpmodel import NLPModel
from pykrylov.linop.linop import LinearOperator


class QPModel(NLPModel):
    u"""Generic class to represent a quadratic programming (QP) problem.

    minimize    cᵀx + 1/2 xᵀHx
    subject to  L ≤ A x ≤ U
                l ≤ x ≤ u.
    """

    def __init__(self, name='GenericQP', **kwargs):
        """Initialize a QP.

        The QP may be created in two ways: as a set of vector and matrix-like
        objects or using a given NLPModel and a given point `x`. The latter
        is useful in SQP algorithms where a QP approximation is constructed and
        solved to improve the candidate solution of the NLP.

        :keywords:

            :fromOps:   A group of operator objects in the form (`c`,`H`,`A`)
                        where `c` is a numpy array, and `H` and `A` are
                        dense matrices, sparse matrices, or linear operators.
                        In addition, `A` may be type `None` to model
                        unconstrained problems.

                        Note: if `H` is a zero operator, the LPModel class
                        should be used instead.

            :fromProb:  A group consisting of an NLPModel (of any type) and a
                        numpy array containing the point `x` at which the
                        QP approximation to the NLPModel is formed. If `x` is
                        type `None`, a starting point of zero is used.

        See the documentation of `NLPModel` for futher information.
        """

        fromOps = kwargs.get('fromOps',None)
        fromProb = kwargs.get('fromProb',None)

        if fromOps is not None:
            c = fromOps[0]
            H = fromOps[1]
            A = fromOps[2]

            # Basic data type checks.
            n = c.shape[0]
            if H.shape[0] != n or H.shape[1] != n:
                raise ValueError('H has inconsistent shape')

            if A is None:
                m = 0
                A = LinearOperator(n, 0,
                                   lambda x: np.empty((0, 1)),
                                   matvec_transp=lambda y: np.empty((n, 0)),
                                   dtype=np.float)
            else:
                if A.shape[1] != n:
                    raise ValueError('A has inconsistent shape')
                m = A.shape[0]

        elif fromProb is not None:
            model = fromProb[0]
            x = fromProb[1]

            if x is None:
                x = np.zeros(model.n, dtype=np.float)

            # Evaluate model functions to construct the QP at x
            c = model.grad(x)
            H = model.hess(x, np.zeros(model.m, dtype=np.float))
            A = model.jac(x)

            n = c.shape[0]
            m = A.shape[0]

            # Make sure variable and constraint bounds are identified
            kwargs['Lvar'] = model.Lvar
            kwargs['Uvar'] = model.Uvar
            kwargs['Lcon'] = model.Lcon
            kwargs['Ucon'] = model.Ucon

            # Other important arguments to pass through
            name = model.name
            kwargs['nnzj'] = model.nnzj
            kwargs['nnzh'] = model.nnzh

        else:
            raise ValueError('QP could not be created.')

        # Initialize model and store key objects
        super(QPModel, self).__init__(n=n, m=m, name=name, **kwargs)
        self.c = c
        self.H = H
        self.A = A

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


