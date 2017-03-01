# -*- coding: utf-8 -*-
"""Base classes for linear optimization models."""

import logging
import numpy as np
from nlp.model.kkt import KKTresidual
from nlp.model.nlpmodel import NLPModel
from pykrylov.linop.linop import LinearOperator

class LPModel(QPModel):
    u"""Generic class to represent a linear programming (LP) problem.

    minimize    cᵀx
    subject to  L ≤ A x ≤ U
                l ≤ x ≤ u.
    """

    def __init__(self, name='GenericLP', **kwargs):
        """Initialize an LP.

        The LP may be created in two ways: as a set of vector and matrix-like
        objects or using a given NLPModel and a given point `x`. The latter
        is useful in SLP or SLQP algorithms where an LP approximation is
        constructed and solved to improve the candidate solution of the NLP.

        :keywords:

            :fromOps:   A group of operator objects in the form (`c`,`A`) where
                        `c` is a numpy array and `A` is a matrix-like object
                        (i.e., dense matrix, sparse matrix, or linear operator).

            :fromProb:  A group consisting of an NLPModel (of any type) and a
                        numpy array containing the point `x` at which the LP
                        approximation is formed. If `x` is
                        type `None`, a starting point of zero is used.

        See the documentation of `NLPModel` for futher information.
        """

        fromOps = kwargs.get('fromOps',None)
        fromProb = kwargs.get('fromProb',None)

        if fromOps is not None:
            c = fromOps[0]
            A = fromOps[1]

            # Basic data type checks.
            n = c.shape[0]

            if A is None:
                raise ValueError('Cannot have an unconstrained LP')
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
            A = model.jac(x)

            n = c.shape[0]
            m = A.shape[0]

            # Make sure variable and constraint bounds are identified
            kwargs['Lvar'] = model.Lvar
            kwargs['Uvar'] = model.Uvar
            kwargs['Lcon'] = model.Lcon
            kwargs['Ucon'] = model.Ucon

        else:
            raise ValueError('LP could not be created.')

        # Create a zero Hessian operator and use the QP model constructor
        # to do the rest of the initialization work.
        H = LinearOperator(n, n,
                           lambda x: np.zeros(n),
                           symmetric=True,
                           dtype=np.float)
        kwargs['fromOps'] = (c, H, A)
        super(LPModel, self).__init__(name=name, **kwargs)

    def obj(self, x):
        """Evaluate the objective function at x."""
        return np.dot(self.c, x)

    def grad(self, x):
        """Evaluate the objective gradient at x."""
        return self.c


