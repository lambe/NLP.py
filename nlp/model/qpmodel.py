# -*- coding: utf-8 -*-
"""Base classes for quadratic optimization models."""

import logging
import numpy as np
from nlp.model.kkt import KKTresidual
from nlp.model.nlpmodel import NLPModel
from pysparse.sparse import PysparseMatrix
from pykrylov.linop.linop import LinearOperator


class QPModel(NLPModel):
    u"""Generic class to represent a quadratic programming (QP) problem.

    minimize    q + cᵀx + ½ xᵀHx
    subject to  L ≤ A x - b ≤ U
                l ≤ x ≤ u.
    """

    def __init__(self, name='GenericQP', **kwargs):
        """Initialize a QP.

        The QP may be created in two ways: as a set of vector and matrix-like
        objects or using a given NLPModel and a given point. The latter is
        useful in SQP algorithms where a QP approximation is constructed and
        solved to improve the candidate solution of the NLP.

        :keywords:

            :fromOps:   A group of operator objects in the form (`c`,`H`,`A`)
                        where `c` is a numpy array, and `H` and `A` are
                        dense matrices, sparse matrices, or linear operators.
                        In addition, `A` may be type `None` to model
                        unconstrained problems.

                        Note: if `H` is a zero operator, the LPModel class
                        should be used instead.

            :opsConst:  A optional group of constant terms (`q`,`b`) for the
                        operator interface. If not specified, they are set
                        to zero.

            :fromProb:  A group consisting of an NLPModel (of any type) and
                        numpy arrays containing the point `x` and Lagrange
                        multipliers `z` at which the QP approximation to the
                        NLPModel is formed. If either `x` or `z` is type
                        `None`, zero arrays are initialized.

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

            # Get constants if any are specified
            opsConst = kwargs.get('opsConst',None)

            if opsConst is None:
                q = 0.
                b = np.zeros(m, dtype=np.float)
            else:
                q = opsConst[0]
                b = opsConst[1]

                if b is None:
                    b = np.zeros(m, dtype=np.float)
                else:
                    if b.shape[0] != m:
                        raise ValueError('b has inconsistent shape')

        elif fromProb is not None:
            model = fromProb[0]
            x = fromProb[1]
            z = fromProb[2]

            if x is None:
                x = np.zeros(model.n, dtype=np.float)

            if z is None:
                z = np.zeros(model.m, dtype=np.float)

            # Evaluate model functions to construct the QP at (x, z)
            q = model.obj(x)
            c = model.grad(x)
            H = model.hess(x, z)
            b = -model.cons(x)
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
        self.q = q
        self.c = c
        self.H = H
        self.b = b
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
        return np.dot(cHx, x) + self.q

    def grad(self, x):
        """Evaluate the objective gradient at x."""
        Hx = self.hprod(x, 0, x)
        Hx += self.c
        return Hx

    def cons(self, x):
        """Evaluate the constraints at x."""
        if isinstance(self.A, np.ndarray):
            return np.dot(self.A, x) - self.b
        return self.A * x - self.b

    def jac(self, x):
        """Evaluate the constraints Jacobian at x."""
        return self.A

    def jprod(self, x, p):
        """Evaluate Jacobian-vector product at x with p."""
        if isinstance(self.A, np.ndarray):
            return np.dot(self.A, p)
        return self.A * p

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


class LSQModel(QPModel):
    u"""Generic class to represent a linear least-squares (LSQ) problem.

    minimize    q + cᵀx + ½ xᵀHx + ½ ||Cx - d||²
    subject to  L ≤ A x - b ≤ U
                l ≤ x ≤ u.
    """

    def __init__(self, name='GenericLSQ', **kwargs):
        u"""Initialize an LSQ problem.

        The variables `q`, `c`, `H`, `A`, and `b` are initialized in the same
        way as in QPModel. The least-squares terms are added in the separate
        argument :lsqOps:. To account for the least-squares terms, the
        original values of `q`, `c`, and `H`, are modified according to:

        H_new = H + CᵀC
        c_new = c + dᵀC
        q_new = q + ½dᵀd

        Note that this construction requires the :fromOps: keyword to be
        defined already.

        :keywords:

            :lsqOps:    A group of operator objects in the form (`d`,`C`)
                        where `d` is a numpy array and `C` is a dense matrix
                        sparse matrix, or linear operator.

            :nnzc:      The number of nonzeros in the least-squares Jacobian.
                        Default = n*p, i.e., a dense matrix.

            :fromOps:   Same meaning as in `QPModel`

            :opsConst:  Same meaning as in `QPModel`

        See the documentation of `NLPModel` for further information.
        """

        # Extract these options and update their values
        fromOps = kwargs.pop('fromOps',None)
        opsConst = kwargs.pop('opsConst',None)

        lsqOps = kwargs.get('lsqOps',None)

        if fromOps is None or lsqOps is None:
            raise ValueError('Not enough information; LSQ model not created')

        c = fromOps[0]
        H = fromOps[1]
        A = fromOps[2]

        d = lsqOps[0]
        C = lsqOps[1]

        # Basic data and shape checks
        n = C.shape[1]
        if C.shape[0] != d.shape[0]:
            raise ValueError('C and d have inconsistent shape')

        if c is None:
            c = np.zeros(n, dtype=np.float)
        else:
            if c.shape[0] != n:
                raise ValueError('c has inconsistent shape')

        if opsConst is None:
            q = 0.
            b = None
        else:
            q = opsConst[0]
            if q is None:
                q = 0.

        # Create new arguments for general QP Model
        q = q + 0.5*np.dot(d,d)

        if isinstance(C, np.ndarray):
            c = c + np.dot(C.T,d)
        else:
            c = c + C.T * d

        if H is None:
            if isinstance(C, np.ndarray):
                H = np.dot(C,C)
            elif isinstance(C, PysparseMatrix):
                H = C.T * C
            elif isinstance(C, LinearOperator):
                H = LinearOperator(n, n, lambda x: C.T * (C * x),
                    symmetric=True, dtype=np.float)

        elif type(C) == type(H):
            if isinstance(C, np.ndarray):
                H = H + np.dot(C,C)
            elif isinstance(C, PysparseMatrix):
                H = H + C.T * C
            elif isinstance(C, LinearOperator):
                # Does the following make sense in Python?
                H = LinearOperator(n, n, lambda x: (H*x + C.T * (C * x)),
                    symmetric=True, dtype=np.float)

        else:
            raise TypeError('C and H are incompatible types')

        # Finally, call QPModel constructor to finish assembling model
        super(LSQModel, self).__init__(name=name, fromOps=(c, H, A),
            opsConst=(q, b), **kwargs)
        self.C = C
        self.d = d
        self.p = C.shape[0]
        self.nnzc = kwargs.get('nnzc',self.p*self.n)

    def lsq_cons(self, x):
        """Evaluate the least-squares residuals at x."""
        if isinstance(self.C, np.ndarray):
            return np.dot(self.C, x) - self.d
        return self.C * x - d

    def lsq_jac(self, x):
        """Evaluate the least-squares Jacobian at x."""
        return self.C

    def lsq_jprod(self, x, p):
        """Evaluate the least-squares Jacobian-vector product at x with p."""
        if isinstance(self.C, np.ndarray):
            return np.dot(self.C, p)
        return self.C * p

    def lsq_jtprod(self, x, p):
        """Evaluate the least-squares transposed-Jacobian-vector product
        at x with p."""
        if isinstance(self.C, np.ndarray):
            return np.dot(self.C.T, p)
        return self.C.T * p


class RLSQModel(QPModel):
    u"""Generic class to represent a transformed linear least-squares problem.

    The transformation consists of adding a special set of variables

    r = d - Cx

    to the problem in LSQModel. The resulting problem is:

    minimize    q + cᵀx + ½ xᵀHx + ½ ||r||²
    w.r.t.      x, r
    subject to  Cx - d + r = 0
                L ≤ A x - b ≤ U
                l ≤ x ≤ u.

    This transformation allows algorithms to exploit sparsity in `C`. The
    variables `r` need to be kept separate from `x` because of their
    special nature in solution algorithms.

    Because of the need to keep `r` and `x` explicitly separate, this
    class defines two additional methods specific to the least-squares
    problem structure. lsq_cons() returns the result of r = d - Cx, and
    lsq_jac() returns `C`. These methods may be used in dedicated least-
    squares solvers for convenience.
    """

    def __init__(self, name='GenericRLSQ', **kwargs):
        u"""Initialize a transformed LSQ problem.

        In this case, the standard QP operators pass through unchanged. All
        we have to do is check for errors in the least-squares terms in the
        :lsqOps: key word.

        :keywords:

            :lsqOps:    Same meaning as in `LSQModel`

            :nnzc:      Same meaning as in `LSQModel`

            :fromOps:   Same meaning as in `QPModel`

            :opsConst:  Same meaning as in `QPModel`

        See the documentation of `NLPModel` for further information.
        """

        lsqOps = kwargs.get('lsqOps',None)
        fromOps = kwargs.get('fromOps',None)

        if fromOps is None or lsqOps is None:
            raise ValueError('Not enough information; LSQ model not created')

        c = fromOps[0]

        d = lsqOps[0]
        C = lsqOps[1]

        # Basic data and shape checks
        n = C.shape[1]
        if C.shape[0] != d.shape[0]:
            raise ValueError('C and d have inconsistent shape')

        # In the basic case of solving an unconstrained least-squares problem,
        # we need c to be well-defined to not generate errors
        if c is None:
            c = np.zeros(n, dtype=np.float)
            fromOps[0] = c
        else:
            if c.shape[0] != n:
                raise ValueError('c has inconsistent shape')

        # Call QPModel constructor to finish assembling model
        super(RLSQModel, self).__init__(name=name, **kwargs)
        self.C = C
        self.d = d
        self.p = C.shape[0]
        self.nnzc = kwargs.get('nnzc',self.p*self.n)

    def lsq_obj(self, r):
        """Evaluate the objective terms involving r."""
        return 0.5*np.dot(r,r)

    def lsq_cons(self, x, r):
        """Evaluate the least-squares residuals at (x,r)."""
        if isinstance(self.C, np.ndarray):
            return np.dot(self.C, x) - self.d + r
        return self.C * x - d + r

    def lsq_jac(self, x):
        """Evaluate the least-squares Jacobian at x."""
        return self.C

    def lsq_jprod(self, x, p):
        """Evaluate the least-squares Jacobian-vector product at x with p."""
        if isinstance(self.C, np.ndarray):
            return np.dot(self.C, p)
        return self.C * p

    def lsq_jtprod(self, x, p):
        """Evaluate the least-squares transposed-Jacobian-vector product
        at x with p."""
        if isinstance(self.C, np.ndarray):
            return np.dot(self.C.T, p)
        return self.C.T * p
