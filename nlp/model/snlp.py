# -*- coding: utf-8 -*-
"""A framework to add slack variables to any NLPModel."""

import numpy as np
from nlp.model.nlpmodel import NLPModel
from pysparse.sparse import PysparseMatrix as psp
from pykrylov.linop.linop import LinearOperator

__docformat__ = 'restructuredtext'


class SlackModel(NLPModel):
    u"""General framework for converting a nonlinear optimization problem to a
    form using slack variables.

    Original problem::

         cᴸ ≤ c(x)
              c(x) ≤ cᵁ
        cᴿᴸ ≤ c(x) ≤ cᴿᵁ
              c(x) = cᴱ
          l ≤   x  ≤ u

    is transformed to::

        c(x) - sᴸ = 0
        c(x) - sᵁ = 0
        c(x) - sᴿ = 0
        c(x) - cᴱ = 0

         cᴸ ≤ sᴸ
              sᵁ ≤ cᵁ
        cᴿᴸ ≤ sᴿ ≤ cᴿᵁ
          l ≤ x  ≤ u

    In the latter problem, the only inequality constraints are bounds on the
    slack and original variables. The other constraints are (typically)
    nonlinear equalities.

    The order of variables in the transformed problem is as follows:

    1. x, the original problem variables.

    2. sᴸ, the slack variables corresponding to general constraints with
       a lower bound only.

    3. sᵁ, the slack variables corresponding to general constraints with
       an upper bound only.

    4. sᴿ, the slack variables corresponding to general constraints with
       a lower bound and an upper bound.

    This framework initializes the slack variables sL and sU to
    zero by default.

    Note that the slack framework does not update all members of NLPModel,
    such as the index set of constraints with an upper bound, etc., but
    rather performs the evaluations of the constraints for the updated
    model implicitly.
    """

    def __init__(self, model, **kwargs):
        """Initialize a slack form of an :class:`NLPModel`.

        :parameters:
            :model:  Original model to be transformed into a slack form.
        """
        self.model = model

        # Save number of variables and constraints prior to transformation
        self.original_n = model.n
        self.original_m = model.m

        # Number of slacks for the constaints
        n_slacks = model.nlowerC + model.nupperC + model.nrangeC
        self.n_slacks = n_slacks

        # Update effective number of variables and constraints
        n = self.original_n + n_slacks
        m = self.original_m

        Lvar = -np.infty * np.ones(n)
        Uvar = +np.infty * np.ones(n)

        # Copy orignal bounds
        Lvar[:self.original_n] = model.Lvar
        Uvar[:self.original_n] = model.Uvar

        # Add bounds corresponding to lower constraints
        bot = self.original_n
        self.sL = range(bot, bot + model.nlowerC)
        Lvar[bot:bot + model.nlowerC] = model.Lcon[model.lowerC]

        # Add bounds corresponding to upper constraints
        bot += model.nlowerC
        self.sU = range(bot, bot + model.nupperC)
        Uvar[bot:bot + model.nupperC] = model.Ucon[model.upperC]

        # Add bounds corresponding to range constraints
        bot += model.nupperC
        self.sR = range(bot, bot + model.nrangeC)
        Lvar[bot:bot + model.nrangeC] = model.Lcon[model.rangeC]
        Uvar[bot:bot + model.nrangeC] = model.Ucon[model.rangeC]

        # No more inequalities. All constraints are now equal to 0
        Lcon = Ucon = np.zeros(m)

        super(SlackModel, self).__init__(n=n, m=m, name='Slack-' + model.name,
                                         Lvar=Lvar, Uvar=Uvar,
                                         Lcon=Lcon, Ucon=Ucon)

        # Redefine primal and dual initial guesses
        self.original_x0 = model.x0[:]
        self.x0 = np.zeros(self.n)
        self.x0[:self.original_n] = self.original_x0[:]

        self.original_pi0 = model.pi0[:]
        self.pi0 = np.zeros(self.m)
        self.pi0[:self.original_m] = self.original_pi0[:]
        return

    def initialize_slacks(self, val=0.0, **kwargs):
        """Initialize all slack variables to given value.

        This method may need to be overridden.
        """
        self.x0[self.original_n:] = val
        return

    def obj(self, x):
        """Evaluate the objective function at x..

        This function is specialized since the original objective function only
        depends on a subvector of `x`.
        """
        f = self.model.obj(x[:self.original_n])

        return f

    def grad(self, x):
        """Evaluate the objective gradient at x.

        This function is specialized since the original objective function only
        depends on a subvector of `x`.
        """
        g = np.zeros(self.n)
        g[:self.original_n] = self.model.grad(x[:self.original_n])

        return g

    def cons(self, x):
        """Evaluate vector of constraints at x.

        Constraints are stored in the order in which they appear in the
        original problem.
        """
        on = self.original_n
        model = self.model

        equalC = model.equalC
        lowerC = model.lowerC
        upperC = model.upperC
        rangeC = model.rangeC

        c = model.cons(x[:on])

        c[equalC] -= model.Lcon[equalC]
        c[lowerC] -= x[self.sL]
        c[upperC] -= x[self.sU]
        c[rangeC] -= x[self.sR]

        return c

    def lsq_cons(self, x, r):
        """Evaluate the least-squares residuals at (x,r).

        This method only applies to least-squares models.
        """
        on = self.original_n
        model = self.model

        if hasattr(model,'lsq_cons'):
            return model.lsq_cons(x[:on],r)
        else:
            raise TypeError('self.model is not a least-squares model.')

    def jprod(self, x, v, **kwargs):
        """Evaluate Jacobian-vector product at x with v.

        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        model = self.model
        on = self.original_n
        lowerC = model.lowerC
        upperC = model.upperC
        rangeC = model.rangeC

        p = model.jprod(x[:on], v[:on])

        # Insert contribution of slacks on general constraints
        p[lowerC] -= v[self.sL]
        p[upperC] -= v[self.sU]
        p[rangeC] -= v[self.sR]
        return p

    def jtprod(self, x, v, **kwargs):
        """Evaluate transposed-Jacobian-vector product at x with v.

        See the documentation of :meth:`jac` for more details on how the
        constraints are ordered.
        """
        # Create some shortcuts for convenience
        model = self.model
        on = self.original_n
        n = self.n
        lowerC = model.lowerC
        upperC = model.upperC
        rangeC = model.rangeC

        p = np.zeros(n)
        p[:on] = model.jtprod(x[:on], v)

        # Insert contribution of slacks on general constraints
        p[self.sL] = -v[lowerC]
        p[self.sU] = -v[upperC]
        p[self.sR] = -v[rangeC]
        return p

    def jac(self, x, **kwargs):
        """Evaluate constraints Jacobian at x.

        The gradients of the general constraints appear in 'natural' order,
        i.e., in the order in which they appear in the problem.

        The overall Jacobian of the  constraints has the form::

            [ J    -I ]

        where the columns correspond to the variables `x` and `s`, and
        the rows correspond to the general constraints (in natural order).
        """
        n = self.n
        m = self.m
        model = self.model
        on = self.original_n

        lowerC = model.lowerC
        upperC = model.upperC
        rangeC = model.rangeC

        J = model.jac(x[:on], **kwargs)

        if isinstance(J, np.ndarray):
            # Create a numpy array, populate the two main blocks, and return
            new_J = np.zeros([m,n])

            new_J[:, :on] = J
            new_J[lowerC, self.sL] = -1.
            new_J[upperC, self.sU] = -1.
            new_J[rangeC, self.sR] = -1.

        elif isinstance(J, psp):
            # Create a new Pysparse matrix and populate
            nnzJ = self.model.nnzj + m
            new_J = psp(nrow=m, ncol=n, sizeHint=nnzJ)

            new_J[:, :on] = J
            new_J[lowerC, self.sL] = -1. # Test these calls on interface
            new_J[upperC, self.sU] = -1.
            new_J[rangeC, self.sR] = -1.

        elif isinstance(J, LinearOperator):
            # Create a new linear operator calling the SlackModel jprod() and
            # jtprod() methods
            new_J = self.jop(x)

        else:
            raise TypeError('Jacobian return type not recognized.')

        return new_J

    def lsq_jprod(self, x, v):
        """Evaluate the least-squares Jacobian-vector product at x with v.

        Because the slack variables create a zero block in the least-squares
        operator, this is trivial.
        """
        on = self.original_n
        return model.lsq_jprod(x[:on], v[:on])

    def lsq_jtprod(self, x, v):
        """Evaluate the least-squares transpose-Jacobian-vector product at x
        with v.

        Because the slack variables create a zero block in the least-squares
        operator, there is a zero block in the output vector.
        """
        on = self.original_n
        n = self.n

        p = np.zeros(n)
        p[:on] = model.lsq_jtprod(x[:on], v)
        return p

    def lsq_jac(self, x):
        """Evaluate the least-squares Jacobian at x.

        The modified operator has the form:

            [ C     0 ]

        where the columns correspond to the variables `x` and `s`.

        This method only applies to least-squares models.
        """
        n = self.n
        on = self.original_n
        model = self.model

        if hasattr(model, 'p'):
            p = model.p
        else:
            raise TypeError('self.model is not a least-squares model.')

        C = model.lsq_jac(x[:on])

        if isinstance(C, np.ndarray):
            # Create a larger numpy array and populate
            new_C = np.zeros([p,n])
            new_C[:, :on] = C

        elif isinstance(C, psp):
            # Create a larger Pysparse matrix and populate
            new_C = psp(nrow=p, ncol=n, sizeHint=model.nnzc)
            new_C[:, :on] = C

        elif isinstance(C, LinearOperator):
            # Create a new linear operator calling the SlackModel jprod() and
            # jtprod() methods
            new_C = self.lsq_jop(x)

        else:
            raise TypeError('Least-squares return type not recognized.')

        return new_C

    def lsq_jop(self, x):
        """Obtain the updated least-squares Jacobian as a linear operator."""
        return LinearOperator(self.n, self.model.p,
                              lambda v: self.lsq_jprod(x,v),
                              matvec_transp=lambda u: self.lsq_jtprod(x,u),
                              symmetric=False,
                              dtype=np.float)

    def hprod(self, x, y, v, **kwargs):
        """Hessian-vector product.

        Evaluate matrix-vector product between the Hessian of the Lagrangian at
        (x, z) and p.
        """
        if y is None:
            y = np.zeros(self.m)

        # Create some shortcuts for convenience
        model = self.model
        on = self.original_n

        Hv = np.zeros(self.n)
        Hv[:on] = model.hprod(x[:on], y, v[:on], **kwargs)
        return Hv

    def hess(self, x, z=None, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        n = self.n
        model = self.model
        on = self.original_n

        H = model.hess(x[:on],z,**kwargs)

        if isinstance(H, np.ndarray):
            # Create a larger numpy array, with slack terms zero
            new_H = np.zeros([n,n])
            new_H[:on, :on] = H

        elif isinstance(H, psp):
            # Create a new pysparse matrix and populate
            new_H = psp(nrow=self.n, ncol=self.n, symmetric=True,
                sizeHint=model.nnzh)
            new_H[:on, :on] = H

        elif isinstance(H, LinearOperator):
            # Create a new linear operator calling the SlackModel hprod()
            new_H = self.hop(x, z=z)

        else:
            raise TypeError('Hessian return type not recognized.')

        return new_H