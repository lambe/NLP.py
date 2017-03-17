# -*- coding: utf-8 -*-
"""A framework to add slack variables to any NLPModel."""

import numpy as np
from nlp.model.nlpmodel import NLPModel
from pykrylov.linop.linop import LinearOperator, ReducedLinearOperator
from pykrylov.linop.linop import SymmetricallyReducedLinearOperator

__docformat__ = 'restructuredtext'


class NoFixedVarsModel(NLPModel):
    u"""General framework for converting a nonlinear optimization problem to a
    form with no fixed variables.

    This transformation is necessary for interior-point methods, which require
    that the optimization problem have a relative interior to work properly.

    The model simply isolates all fixed variables in a separate array. Every
    time the objective or constraints are called, the fixed variables are
    added to the variable set.
    """

    def __init__(self, model, **kwargs):
        """Initialize an :class:`NLPModel` with no fixed variables.

        :parameters:
            :model: Original model for which fixed variables are removed
        """
        self.model = model

        self.original_n = model.n
        n = self.original_n - model.nfixedB
        m = model.m

        # Indices of fixed and non-fixed variables
        self._ind_fixed = model.fixedB
        self._ind_notfixed = model.rangeB + model.lowerB + model.upperB + model.freeB
        self._ind_notfixed.sort()
        self._all_cons = np.arange(m, dtype=np.int)

        # Values of the fixed variables
        self._x_fixed = model.Lvar[model.fixedB]

        # Memory for the complete x vector, including fixed variables
        self._x_all = model.x0.copy()
        self._x_all[model.fixedB] = self._x_fixed
        self._vec_all = np.zeros(n, dtype=np.float)

        # Remove fixed variable bounds
        Lvar = model.Lvar[self._ind_notfixed]
        Uvar = model.Uvar[self._ind_notfixed]

        # Other options to keep the same
        Lcon = model.Lcon
        Ucon = model.Ucon
        nnzj = kwargs.get('nnzj',0)
        nnzh = kwargs.get('nnzh',0)

        super(NoFixedVarsModel, self).__init__(n=n, m=m, name='NoFix-'+model.name,
            Lvar=Lvar, Uvar=Uvar, Lcon=Lcon, Ucon=Ucon, nnzj=nnzj, nnzh=nnzh)

        self.x0 = model.x0[self._ind_notfixed]
        self.pi0 = model.pi0[:]
        return

    def compose_x(self, x_not_fixed):
        """Create a vector of both fixed and non-fixed variables."""
        self._x_all[self._ind_notfixed] = x_not_fixed
        return

    def compose_vec(self, v_not_fixed):
        """Lengthen an input vector for matrix-vector products."""
        self._vec_all[self._ind_notfixed] = v_not_fixed
        return

    def obj(self, x):
        self.compose_x(x)
        f = self.model.obj(self._x_all)
        return f

    def cons(self, x):
        self.compose_x(x)
        c = self.model.cons(self._x_all)
        return c

    def grad(self, x):
        self.compose_x(x)
        g = self.model.grad(self._x_all)
        return g[self._ind_notfixed]

    def jac(self, x, **kwargs):
        """Whatever the Jacobian is, it needs the right dimensions."""
        self.compose_x(x)
        J = self.model.jac(self._x_all, **kwargs)
        if isinstance(J,LinearOperator):
            return ReducedLinearOperator(J,self._all_cons,self._ind_notfixed)
        else:
            return J[:,self._ind_notfixed]

    def jprod(self, x, p, **kwargs):
        self.compose_x(x)
        self.compose_vec(p)
        q = self.model.jprod(self._x_all, self._vec_all, **kwargs)
        return q

    def jtprod(self, x, p, **kwargs):
        self.compose_x(x)
        q = self.model.jtprod(self._x_all, p, **kwargs)
        return q[self._ind_notfixed]

    def hess(self, x, z=None, **kwargs):
        """Whatever the Hessian is, it needs the right dimensions."""
        self.compose_x(x)
        H = self.model.hess(self._x_all, z, **kwargs)
        if isinstance(H,LinearOperator):
            return SymmetricallyReducedLinearOperator(H, self._ind_notfixed)
        else:
            return H[self._ind_notfixed, self._ind_notfixed]

    def hprod(self, x, z, p, **kwargs):
        if z is None:
            z = np.zeros(self.m)

        self.compose_x(x)
        self.compose_vec(p)
        q = self.model.hprod(self._x_all, z, self._vec_all, **kwargs)
        return q[self._ind_notfixed]