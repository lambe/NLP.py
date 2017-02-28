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

