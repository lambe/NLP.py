# -*- coding: utf-8 -*-
"""A framework to convert all constraints to equality and nonnegativity."""

import numpy as np
from nlp.model.nlpmodel import NLPModel

__docformat__ = 'restructuredtext'


class PosModel(NLPModel):
	u"""General framework for converting a nonlinear optimization problem
	to a form with nonnegative inequality constraints.

	** Full implementation to be completed **
	** Documentation to be updated **
	"""

    def cons_pos(self, x):
        """Convenience function to return constraints as non negative ones.

        Constraints are reformulated as

          ci(x) - ai  = 0  for i in equalC
          ci(x) - Li >= 0  for i in lowerC + rangeC
          Ui - ci(x) >= 0  for i in upperC + rangeC.

        The constraints appear in natural order, except for the fact that the
        'upper side' of range constraints is appended to the list.

        Scaling should be applied in cons().
        """
        m = self.m
        equalC = self.equalC
        lowerC = self.lowerC
        upperC = self.upperC
        rangeC = self.rangeC
        nrangeC = self.nrangeC

        # Set the type of c to the type of x to allow for object arrays.
        # This is useful to AD packages.
        c = np.empty(m + nrangeC, dtype=x.dtype)
        c[:m] = self.cons(x)
        c[m:] = c[rangeC]

        c[equalC] -= self.Lcon[equalC]
        c[lowerC] -= self.Lcon[lowerC]
        c[upperC] -= self.Ucon[upperC]
        c[upperC] *= -1
        c[rangeC] -= self.Lcon[rangeC]
        c[m:] -= self.Ucon[rangeC]
        c[m:] *= -1

        return c

    def jac_pos(self, x, **kwargs):
        """Evaluate the Jacobian of :meth:`cons_pos` at x."""
        raise NotImplementedError('This method must be subclassed.')

    def jop_pos(self, x):
        """Jacobian of :meth:`cons_pos` at x as a linear operator."""
        J = self.jop(x)
        e = np.ones(self.ncon + self.nrangeC)
        e[self.upperC] = -1
        e[self.ncon:] = -1
        JR = ReducedLinearOperator(J, self.rangeC, range(self.nvar))
        Jpos = BlockLinearOperator([[J], [JR]], dtype=np.float)
        D = DiagonalOperator(e)
        return D * Jpos  # Flip sign of 'upper' constraints.

    def primal_feasibility(self, x, c=None):
        """Evaluate the primal feasibility residual at x.

        If `c` is given, it should conform to :meth:`cons_pos`.
        """
        if self.m == 0:
            return np.zeros(0)

        # Shortcuts.
        eC = self.equalC
        m = self.m
        nrC = self.nrangeC
        nB = self.nbounds
        nrB = self.nrangeB

        pFeas = np.empty(m + nrC + nB + nrB)
        pFeas[:m + nrC] = -self.cons_pos(x) if c is None else -c
        not_eC = [i for i in range(m + nrC) if i not in eC]
        pFeas[eC] = np.abs(pFeas[eC])
        pFeas[not_eC] = np.maximum(0, pFeas[not_eC])
        pFeas[m:m + nrC] = np.maximum(0, pFeas[m:m + nrC])
        pFeas[m + nrC:] = -self.get_bounds(x)
        pFeas[m + nrC:] = np.maximum(0, pFeas[m + nrC:])

        return pFeas

    def dual_feasibility(self, x, y, z, g=None, J=None, **kwargs):
        """Evaluate the dual feasibility residual at (x,y,z).

        The argument `J`, if supplied, should be a linear operator representing
        the constraints Jacobian. It should conform to either :meth:`jac` or
        :meth:`jac_pos` depending on the value of `all_pos` (see below).

        The multipliers `z` should conform to :meth:`get_bounds`.

        :keywords:
            :obj_weight: weight of the objective gradient in dual feasibility.
                         Set to zero to check Fritz-John conditions instead
                         of KKT conditions. (default: 1.0)
            :all_pos:    if `True`, indicates that the multipliers `y` conform
                         to :meth:`jac_pos`. If `False`, `y` conforms to
                         :meth:`jac`. In all cases, `y` should be appropriately
                         ordered. If the positional argument `J` is specified,
                         it must be consistent with the layout of `y`.
                         (default: `True`)
        """
        # Shortcuts.
        lB = self.lowerB
        uB = self.upperB
        rB = self.rangeB
        nlB = self.nlowerB
        nuB = self.nupperB
        nrB = self.nrangeB

        obj_weight = kwargs.get('obj_weight', 1.0)
        all_pos = kwargs.get('all_pos', True)

        if J is None:
            J = self.jop_pos(x) if all_pos else self.jop(x)

        if obj_weight == 0.0:   # Checking Fritz-John conditions.
            dFeas = -J.T * y
        else:
            dFeas = self.grad(x) if g is None else g[:]
            if obj_weight != 1.0:
                dFeas *= obj_weight
            dFeas -= J.T * y
        dFeas[lB] -= z[:nlB]
        dFeas[uB] -= z[nlB:nlB + nuB]
        dFeas[rB] -= z[nlB + nuB:nlB + nuB + nrB] - z[nlB + nuB + nrB:]

        return dFeas

    def complementarity(self, x, y, z, c=None):
        """Evaluate the complementarity residuals at (x,y,z).

        If `c` is specified, it should conform to :meth:`cons_pos` and the
        multipliers `y` should appear in the same order. The multipliers `z`
        should conform to :meth:`get_bounds`.

        :returns:
            :cy:  complementarity residual for general constraints
            :xz:  complementarity residual for bound constraints.
        """
        # Shortcuts.
        lC = self.lowerC
        uC = self.upperC
        rC = self.rangeC
        nlC = self.nlowerC
        nuC = self.nupperC
        nrC = self.nrangeC

        not_eC = lC + uC + rC + range(nlC + nuC + nrC, nlC + nuC + nrC + nrC)
        if c is None:
            c = self.cons_pos(x)

        cy = c[not_eC] * y[not_eC]
        xz = self.get_bounds(x) * z

        return (cy, xz)
