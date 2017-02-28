# -*- coding: utf-8 -*-
"""Abstract base classes to represent continuous optimization models."""

import logging
import os
import sys
import numpy as np
from nlp.model.kkt import KKTresidual
from nlp.tools.decorators import deprecated, counter
from nlp.tools.utils import where
from pykrylov.linop.linop import LinearOperator
from pysparse.sparse import PysparseMatrix as psp


class NLPModel(object):
    """Abstract continuous optimization model.

    The model features methods to evaluate the objective and constraints,
    and their derivatives. Instances of the general class do not do anything
    interesting; they must be subclassed and specialized.
    """

    _id = -1

    def __init__(self, n, m=0, name='Generic', **kwargs):
        """Initialize a model with `n` variables and `m` constraints.

        :parameters:

            :n:       number of variables
            :m:       number of general (non bound) constraints (default: 0)
            :name:    model name (default: 'Generic')

        :keywords:

            :x0:      initial point (default: all 0)
            :pi0:     vector of initial multipliers (default: all 0)
            :Lvar:    vector of lower bounds on the variables
                      (default: all -Infinity)
            :Uvar:    vector of upper bounds on the variables
                      (default: all +Infinity)
            :Lcon:    vector of lower bounds on the constraints
                      (default: all -Infinity)
            :Ucon:    vector of upper bounds on the constraints
                      (default: all +Infinity)
        """
        self._nvar = self._n = n   # Number of variables
        self._ncon = self._m = m   # Number of general constraints
        self._name = os.path.splitext(os.path.basename(name))[0]

        # Set initial point
        if 'x0' in kwargs.keys():
            self.x0 = np.ascontiguousarray(kwargs['x0'], dtype=float)
        else:
            self.x0 = np.zeros(self._n, dtype=np.float)

        # Set initial multipliers
        if 'pi0' in kwargs.keys():
            self.pi0 = np.ascontiguousarray(kwargs['pi0'], dtype=np.float)
        else:
            self.pi0 = np.zeros(self._m, dtype=np.float)

        # Set lower bounds on variables    Lvar[i] <= x[i]  i = 1,...,n
        if 'Lvar' in kwargs.keys():
            self.Lvar = np.ascontiguousarray(kwargs['Lvar'], dtype=np.float)
        else:
            self.Lvar = -np.inf * np.ones(self._n, dtype=np.float)

        # Set upper bounds on variables    x[i] <= Uvar[i]  i = 1,...,n
        if 'Uvar' in kwargs.keys():
            self.Uvar = np.ascontiguousarray(kwargs['Uvar'], dtype=np.float)
        else:
            self.Uvar = np.inf * np.ones(self._n, dtype=np.float)

        # Set lower bounds on constraints  Lcon[i] <= c[i]  i = 1,...,m
        if 'Lcon' in kwargs.keys():
            self.Lcon = np.ascontiguousarray(kwargs['Lcon'], dtype=np.float)
        else:
            self.Lcon = -np.inf * np.ones(self._m, dtype=np.float)

        # Set upper bounds on constraints  c[i] <= Ucon[i]  i = 1,...,m
        if 'Ucon' in kwargs.keys():
            self.Ucon = np.ascontiguousarray(kwargs['Ucon'], dtype=np.float)
        else:
            self.Ucon = np.inf * np.ones(self._m, dtype=np.float)

        # The number of nonzeros in the Jacobian and Lagrangian Hessian
        # Only used in derived classes with sparse matrix storage
        self.nnzj = kwargs.get('nnzj',None)
        self.nnzh = kwargs.get('nnjh',None)

        # Default classification of constraints
        self._lin = []                        # Linear    constraints
        self._nln = range(self._m)            # Nonlinear constraints
        self._net = []                        # Network   constraints
        self._nlin = len(self.lin)            # Number of linear constraints
        self._nnln = len(self.nln)            # Number of nonlinear constraints
        self._nnet = len(self.net)            # Number of network constraints

        # Maintain lists of indices for each type of constraints:
        self.rangeC = []    # Range constraints:       cL <= c(x) <= cU
        self.lowerC = []    # Lower bound constraints: cL <= c(x)
        self.upperC = []    # Upper bound constraints:       c(x) <= cU
        self.equalC = []    # Equality constraints:    cL  = c(x)  = cU

        for i in range(self._m):
            if self.Lcon[i] > -np.inf and self.Ucon[i] < np.inf:
                if self.Lcon[i] == self.Ucon[i]:
                    self.equalC.append(i)
                else:
                    self.rangeC.append(i)
            elif self.Lcon[i] > -np.inf:
                self.lowerC.append(i)
            elif self.Ucon[i] < np.inf:
                self.upperC.append(i)

        self.nlowerC = len(self.lowerC)   # Number of lower bound constraints
        self.nrangeC = len(self.rangeC)   # Number of range constraints
        self.nupperC = len(self.upperC)   # Number of upper bound constraints
        self.nequalC = len(self.equalC)   # Number of equality constraints

        # Define permutations to order constraints / multipliers.
        self.permC = self.equalC + self.lowerC + self.upperC + self.rangeC

        # Proceed similarly with bound constraints
        self.rangeB = []
        self.lowerB = []
        self.upperB = []
        self.fixedB = []
        self.freeB = []

        for i in range(self._n):
            if self.Lvar[i] > -np.inf and self.Uvar[i] < np.inf:
                if self.Lvar[i] == self.Uvar[i]:
                    self.fixedB.append(i)
                else:
                    self.rangeB.append(i)
            elif self.Lvar[i] > -np.inf:
                self.lowerB.append(i)
            elif self.Uvar[i] < np.inf:
                self.upperB.append(i)
            else:
                self.freeB.append(i)

        self.nlowerB = len(self.lowerB)
        self.nrangeB = len(self.rangeB)
        self.nupperB = len(self.upperB)
        self.nfixedB = len(self.fixedB)
        self.nfreeB = len(self.freeB)
        self.nbounds = self._n - self.nfreeB

        # Define permutations to order bound constraints / multipliers.
        self.permB = self.fixedB + self.lowerB + self.upperB + \
            self.rangeB + self.freeB

        # Define default stopping tolerances
        self._stop_d = 1.0e-6    # Dual feasibility
        self._stop_c = 1.0e-6    # Complementarty
        self._stop_p = 1.0e-6    # Primal feasibility

        # Define scaling attributes.
        self.g_max = 1.0e2      # max gradient entry (constant)
        self.scale_obj = None   # Objective scaling
        self.scale_con = None   # Constraint scaling

        # Problem-specific logger.
        self.__class__._id += 1
        self._id = self.__class__._id
        self.logger = logging.getLogger(name=self.name + '_' + str(self._id))
        self.logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(name)-10s %(levelname)-8s %(message)s')
        hndlr = logging.StreamHandler(sys.stdout)
        hndlr.setFormatter(fmt)
        self.logger.addHandler(hndlr)
        self._setup_counters()

    def _setup_counters(self):
        meths = ["obj", "grad", "hess", "cons", "icons", "igrad", "sigrad",
                 "jac", "jprod", "jtprod", "hprod", "hiprod", "ghivprod"]
        for meth in meths:
            setattr(self, meth, counter(getattr(self, meth)))

    @property
    def nvar(self):
        """Number of variables."""
        return self._nvar

    @property
    def n(self):
        """Number of variables."""
        return self._n

    @property
    def ncon(self):
        """Number of constraints (excluding bounds)."""
        return self._ncon

    @property
    def m(self):
        """Number of constraints (excluding bounds)."""
        return self._m

    @property
    def name(self):
        """Problem name."""
        return self._name

    @property
    def lin(self):
        """Return the indices of linear constraints."""
        return self._lin

    @property
    def nlin(self):
        """Number of linear constraints."""
        return self._nlin

    @property
    def nln(self):
        """Return the indices of nonlinear constraints."""
        return self._nln

    @property
    def nnln(self):
        """Number of nonlinear constraints."""
        return self._nnln

    @property
    def nnet(self):
        """Number of network constraints."""
        return self._nnet

    @property
    def net(self):
        """Return the indices of network constraints."""
        return self._net

    @property
    def stop_d(self):
        """Tolerance on dual feasibility."""
        return self._stop_d

    @stop_d.setter
    def stop_d(self, value):
        self._stop_d = max(0, value)

    @property
    def stop_c(self):
        """Tolerance on complementarity."""
        return self._stop_c

    @stop_c.setter
    def stop_c(self, value):
        self._stop_c = max(0, value)

    @property
    def stop_p(self):
        """Tolerance on primal feasibility."""
        return self._stop_p

    @stop_p.setter
    def stop_p(self, value):
        self._stop_p = max(0, value)

    @deprecated
    def get_stopping_tolerances(self):
        """Return current stopping tolerances."""
        return (self.stop_d, self.stop_p, self.stop_c)

    @deprecated
    def set_stopping_tolerances(self, stop_d, stop_p, stop_c):
        """Set stopping tolerances."""
        self.stop_d = stop_d
        self.stop_p = stop_p
        self.stop_c = stop_c
        return

    def compute_scaling_obj(self, x=None, g_max=1.0e2, reset=False):
        """Compute objective scaling.

        :parameters:

            :x: Determine scaling by evaluating functions at this
                point. Default is to use :attr:`self.x0`.
            :g_max: Maximum allowed gradient. Default: :attr:`g_max = 1e2`.
            :reset: Set to `True` to unscale the problem.

        The procedure used here closely
        follows IPOPT's behavior; see Section 3.8 of

          Waecther and Biegler, 'On the implementation of an
          interior-point filter line-search algorithm for large-scale
          nonlinear programming', Math. Prog. A (106), pp.25-57, 2006

        which is a scalar rescaling that ensures the inf-norm of the
        gradient (at x) isn't larger than 'g_max'.
        """
        # Remove scaling if requested
        if reset:
            self.scale_obj = None
            # self.pi0 = self.get_pi0()  # get original multipliers
            return

        # Quick return if the problem is already scaled
        if self.scale_obj is not None:
            return

        if x is None:
            x = self.x0
        g = self.grad(x)
        gNorm = np.linalg.norm(g, np.inf)
        self.scale_obj = g_max / max(g_max, gNorm)  # <= 1 always

        # Rescale the Lagrange multiplier
        # self.pi0 *= self.scale_obj

        return gNorm

    def compute_scaling_cons(self, x=None, g_max=1.0e2, reset=False):
        """Compute constraint scaling.

        :parameters:

            :x: Determine scaling by evaluating functions at this
                point. Default is to use :attr:`self.x0`.
            :g_max: Maximum allowed gradient. Default: :attr:`g_max = 1e2`.
            :reset: Set to `True` to unscale the problem.
        """
        if self.m == 0:
            return

        # Remove scaling if requested
        if reset:
            self.scale_con = None
            self.Lcon = self.model.get_Lcon()  # lower bounds on constraints
            self.Ucon = self.model.get_Ucon()  # upper bounds on constraints
            return

        # Quick return if the problem is already scaled
        if self.scale_con is not None:
            return

        m = self.m
        if x is None:
            x = self.x0
        d_c = np.empty(m)
        J = self.jop(x)

        # Find inf-norm of each row of J
        gmaxNorm = 0            # holds the maximum row-norm of J
        imaxNorm = 0            # holds the corresponding index
        e = np.zeros(self.ncon)
        for i in xrange(m):
            e[i] = 1
            giNorm = np.linalg.norm(J.T * e, np.inf)
            e[i] = 0
            d_c[i] = g_max / max(g_max, giNorm)  # <= 1 always
            if giNorm > gmaxNorm:
                gmaxNorm = giNorm
                imaxNorm = i
            gmaxNorm = max(gmaxNorm, giNorm)

        self.scale_con = d_c

        # Scale constraint bounds: componentwise multiplications
        self.Lcon *= d_c        # lower bounds on constraints
        self.Ucon *= d_c        # upper bounds on constraints

        # Return largest row norm and its index
        return (imaxNorm, gmaxNorm)

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

    def kkt_residuals(self, x, y, z, c=None, g=None, J=None, **kwargs):
        """Compute the first-order residuals.

        There is no check on the sign of the multipliers unless `check` is set
        to `True`. Keyword arguments not specified below are passed directly to
        :meth:`primal_feasibility`, :meth:`dual_feasibility` and
        :meth:`complementarity`.

        If `J` is specified, it should conform to :meth:`jac_pos` and the
        multipliers `y` should be consistent with the Jacobian.

        :keywords:
            :check:  check sign of multipliers.

        :returns:
            :kkt:  KKT residuals as a KKTresidual instance.
        """
        # Shortcuts.
        m = self.m
        nrC = self.nrangeC
        eC = self.equalC
        check = kwargs.get('check', True)

        if check:
            not_eC = [i for i in range(m + nrC) if i not in eC]
            if len(where(y[not_eC] < 0)) > 0:
                raise ValueError('Multipliers for inequalities must be >= 0.')
            if not np.all(z >= 0):
                raise ValueError('Multipliers for bounds must be >= 0.')

        pFeas = self.primal_feasibility(x, c=c)
        dFeas = self.dual_feasibility(x, y, z, g=g, J=J)
        cy, xz = self.complementarity(x, y, z, c=c)

        return KKTresidual(dFeas, pFeas[:m + nrC], pFeas[m + nrC:], cy, xz)

    def at_optimality(self, x, z, **kwargs):
        """Check whether the KKT residuals meet the stopping conditions."""
        kkt = self.optimality_residuals(x, z, **kwargs)
        dFeas = kkt.dFeas <= self.stop_d
        pFeas = kkt.feas <= self.stop_p
        compl = kkt.comp <= self.stop_c
        return dFeas and pFeas and compl

    def bounds(self, x):
        """Return the vector with components x[i]-Lvar[i] or Uvar[i]-x[i].

        Bound constraints on the problem variables are then equivalent to
        bounds(x) >= 0. The bounds are odered as follows:

        [lowerB | upperB | rangeB (lower) | rangeB (upper) ].
        """
        lB = self.lowerB
        uB = self.upperB
        rB = self.rangeB
        nlB = self.nlowerB
        nuB = self.nupperB
        nrB = self.nrangeB
        nB = self.nbounds
        Lvar = self.Lvar
        Uvar = self.Uvar

        b = np.empty(nB + nrB, dtype=x.dtype)
        b[:nlB] = x[lB] - Lvar[lB]
        b[nlB:nlB + nuB] = Uvar[uB] - x[uB]
        b[nlB + nuB:nlB + nuB + nrB] = x[rB] - Lvar[rB]
        b[nlB + nuB + nrB:] = Uvar[rB] - x[rB]
        return b

    def obj(self, x, **kwargs):
        """Evaluate the objective function at x."""
        raise NotImplementedError('This method must be subclassed.')

    def grad(self, x, **kwargs):
        """Evaluate the objective gradient at x."""
        raise NotImplementedError('This method must be subclassed.')

    def cons(self, x, **kwargs):
        """Evaluate vector of constraints at x."""
        raise NotImplementedError('This method must be subclassed.')

    @deprecated
    def icons(self, i, x, **kwargs):
        """Evaluate i-th constraint at x."""
        raise NotImplementedError('This method must be subclassed.')

    @deprecated
    def igrad(self, i, x, **kwargs):
        """Evalutate i-th dense constraint gradient at x."""
        raise NotImplementedError('This method must be subclassed.')

    @deprecated
    def sigrad(self, i, x, **kwargs):
        """Evaluate i-th sparse constraint gradient at x."""
        raise NotImplementedError('This method must be subclassed.')

    def jac(self, x, **kwargs):
        """Evaluate constraints Jacobian at x."""
        raise NotImplementedError('This method must be subclassed.')

    def jprod(self, x, p, **kwargs):
        """Evaluate Jacobian-vector product at x with p."""
        raise NotImplementedError('This method must be subclassed')

    def jtprod(self, x, p, **kwargs):
        """Evaluate transposed-Jacobian-vector product at x with p."""
        raise NotImplementedError('This method must be subclassed')

    def jop(self, x):
        """Obtain Jacobian at x as a linear operator."""
        return LinearOperator(self.n, self.m,
                              lambda v: self.jprod(x, v),
                              matvec_transp=lambda u: self.jtprod(x, u),
                              symmetric=False,
                              dtype=np.float)

    def lag(self, x, z, **kwargs):
        """Evaluate Lagrangian at (x, z).

        The constraints and bounds are assumed to be ordered as in
        :meth:`cons_pos` and :meth:`bounds`.
        """
        m = self.m
        nrC = self.nrangeC
        l = self.obj(x)
        # The following ifs are necessary because np.dot returns None
        # when passed empty arrays of objects (i.e., dtype = np.object).
        # This causes AD tools to error out.
        if self.m > 0:
            l -= np.dot(z[:m + nrC], self.cons_pos(x))
        if self.nbounds > 0:
            l -= np.dot(z[m + nrC:], self.bounds(x))
        return l

    def hess(self, x, z=None, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        raise NotImplementedError('This method must be subclassed.')

    def hprod(self, x, z, p, **kwargs):
        """Hessian-vector product.

        Evaluate matrix-vector product between the Hessian of the Lagrangian at
        (x, z) and p.
        """
        raise NotImplementedError('This method must be subclassed.')

    @deprecated
    def hiprod(self, i, x, p, **kwargs):
        """Constraint Hessian-vector product.

        Evaluate matrix-vector product between the Hessian of the i-th
        constraint at x and p.
        """
        raise NotImplementedError('This method must be subclassed.')

    @deprecated
    def ghivprod(self, x, g, v, **kwargs):
        """Evaluate individual dot products (g, Hi*v).

        Evaluate the vector of dot products (g, Hi*v) where Hi is the Hessian
        of the i-th constraint at x, i = 1, ..., ncon.
        """
        raise NotImplementedError('This method must be subclassed.')

    def hop(self, x, z=None, **kwargs):
        """Obtain Lagrangian Hessian at (x, z) as a linear operator."""
        return LinearOperator(self.n, self.n,
                              lambda v: self.hprod(x, z, v, **kwargs),
                              symmetric=True,
                              dtype=np.float)

    def display_basic_info(self):
        """Display vital statistics about the current model."""
        write = self.logger.info
        write('Problem Name: %s', self.name)
        write('Number of Variables: %d',  self.n)
        write('Number of Bound Constraints: %d', self.nbounds)
        write(' (%d lower, %d upper, %d two-sided)',
              self.nlowerB, self.nupperB, self.nrangeB)
        if self.nlowerB > 0:
            write('Lower bounds: %s', repr(self.lowerB))
        if self.nupperB > 0:
            write('Upper bounds: %s', repr(self.upperB))
        if self.nrangeB > 0:
            write('Two-Sided bounds: %s', repr(self.rangeB))
        if self.nlowerB + self.nupperB + self.nrangeB > 0:
            write('Vector of lower bounds: %s', repr(self.Lvar))
            write('Vectof of upper bounds: %s', repr(self.Uvar))
            write('Number of General Constraints: %d', self.m)
            write(' (%d equality, %d lower, %d upper, %d range)',
                  self.nequalC, self.nlowerC, self.nupperC, self.nrangeC)
        if self.nequalC > 0:
            write('Equality: %s', repr(self.equalC))
        if self.nlowerC > 0:
            write('Lower   : %s', repr(self.lowerC))
        if self.nupperC > 0:
            write('Upper   : %s', repr(self.upperC))
        if self.nrangeC > 0:
            write('Range   : %s', repr(self.rangeC))
        if self.nequalC + self.nlowerC + self.nupperC + self.nrangeC > 0:
            write('Vector of constraint lower bounds: %s', repr(self.Lcon))
            write('Vector of constraint upper bounds: %s', repr(self.Ucon))
            write('Number of Linear Constraints: %d', self.nlin)
            write('Number of Nonlinear Constraints: %d', self.nnln)
            write('Number of Network Constraints: %d', self.nnet)
        write('Initial Guess: %s', repr(self.x0))

        return

    def __repr__(self):
        """Brief info about model."""
        dat = (self.__class__.__name__, self.name, self.n, self.m)
        return '%s %s with %d variables and %d constraints' % dat


class DenseNLPModel(NLPModel):
    """A specialization of NLPModel where jac() and hess() always return
    (dense) 2-D Numpy arrays.

    The matrix-vector product methods jprod(), jtprod(), and hprod() are
    specialized to perform multiplications with the current Jacobian and
    Hessian using np.dot(). The user does not need to define them.
    """

    def jprod(self, x, p, **kwargs):
        """Evaluate Jacobian-vector product at x with p."""
        return np.dot(self.jac(x, **kwargs), p)

    def jtprod(self, x, p, **kwargs):
        """Evaluate transposed-Jacobian-vector product at x with p."""
        return np.dot(p, self.jac(x, **kwargs))

    def hprod(self, x, z, p, **kwargs):
        """Hessian-vector product.

        Evaluate matrix-vector product between the Hessian of the Lagrangian at
        (x, z) and p.
        """
        return np.dot(self.hess(x, z=z, **kwargs), p)


class SparseNLPModel(NLPModel):
    """A specialization of NLPModel where jac() and hess() always return
    sparse matrices of the Pysparse type.

    Note: to use this class, the user must define the jac_triple and
    hess_triple methods to return a triple of Numpy vectors in the order
    (vals, rows, cols). The object's own jac() and hess() methods then
    construct the sparse matrix.

    The matrix-vector product methods jprod(), jtprod(), and hprod() are
    specialized to perform multiplications using the * operator. The user
    does not need to define them.
    """

    def jac_triple(self, x, **kwargs):
        """Evaluate the Jacobian in coordinate (COO) format."""
        raise NotImplementedError('This method must be subclassed.')

    def jac(self, x, **kwargs):
        """Evaluate constraints Jacobian at x.

        The matrix is constructed from a Numpy vector triple defined by
        the jac_triple() method."""
        vals, rows, cols = self.jac_triple(*args, **kwargs)
        J = psp(nrow=self.ncon, ncol=self.nvar,
                sizeHint=vals.size, symmetric=False)
        if vals.size > 0:
            J.put(vals, rows, cols)
        return J

    def jprod(self, x, p, **kwargs):
        """Evaluate Jacobian-vector product at x with p."""
        return self.jac(x, **kwargs) * p

    def jtprod(self, x, p, **kwargs):
        """Evaluate transposed-Jacobian-vector product at x with p."""
        return p * self.jac(x, **kwargs)

    def hess_triple(self, x, z, **kwargs):
        """Evaluate the Lagrangian Hessian in coordinate (COO) format."""
        raise NotImplementedError('This method must be subclassed.')

    def hess(self, x, z=None, **kwargs):
        """Evaluate Lagrangian Hessian at (x, z)."""
        vals, rows, cols = self.hess_triple(*args, **kwargs)
        H = psp(size=self.nvar, sizeHint=vals.size, symmetric=True)
        H.put(vals, rows, cols)
        return H

    def hprod(self, x, z, p, **kwargs):
        """Hessian-vector product.

        Evaluate matrix-vector product between the Hessian of the Lagrangian at
        (x, z) and p.
        """
        return self.hess(x, z=z, **kwargs) * p


class MatrixFreeNLPModel(NLPModel):
    """A specialization of NLPModel for matrix-free problems.

    Unlike SparseNLPModel and DenseNLPModel, the user must provide jprod(),
    jtprod(), and hprod() directly. Since the Jacobian and Hessian are only
    accessed as linear operators, the jac() and hess() methods just return
    the linear operators from jop() and hop()
    """

    def jac(self, x, **kwargs):
        """Evaluate constraints Jacobian at x."""
        return self.jop(x)

    def hess(self, x, z=None, **kwargs):
        """Evaluate the Lagrangian Hessian at (x,z)."""
        return self.hop(x, z=z)


class BoundConstrainedNLPModel(NLPModel):
    """Generic class to represent a bound-constrained problem."""

    def __init__(self, nvar, Lvar, Uvar, **kwargs):
        """Initialize a bound-constrained problem with ``nvar`` variables.

        The bounds may be specified via the arguments ``Lvar``
        and ``Uvar``. See the documentation of :class:`NLPModel` for more
        information.
        """
        # Discard options related to constrained problems.
        kwargs.pop('m', None)
        kwargs.pop('ncon', None)
        kwargs.pop('Lcon', None)
        kwargs.pop('Ucon', None)
        kwargs.pop('Lvar', None)
        kwargs.pop('Uvar', None)
        super(BoundConstrainedNLPModel, self).__init__(nvar,
                                                       Lvar=Lvar,
                                                       Uvar=Uvar,
                                                       **kwargs)

    def cons(self, x):
        """Evaluate the constraints at x."""
        return np.zeros(self.m, dtype=np.float)

    def jprod(self, x, v):
        """Evaluate Jacobian-vector product at x with p."""
        return np.zeros(self.m, dtype=np.float)

    def jtprod(self, x, v):
        """Evaluate transposed-Jacobian-vector product at x with p."""
        return np.zeros(self.n, dtype=np.float)


class UnconstrainedNLPModel(NLPModel):
    """Generic class to represent an unconstrained problem."""

    def __init__(self, nvar, **kwargs):
        """Initialize an unconstrained problem with ``nvar`` variables.

        See the documentation of :class:`NLPModel` for more information.
        """
        # Discard options related to constrained problems.
        kwargs.pop('m', None)
        kwargs.pop('ncon', None)
        kwargs.pop('Lcon', None)
        kwargs.pop('Ucon', None)
        kwargs.pop('Lvar', None)
        kwargs.pop('Uvar', None)
        super(UnconstrainedNLPModel, self).__init__(nvar, **kwargs)

    def cons(self, x):
        """Evaluate the constraints at x."""
        return np.zeros(self.m, dtype=np.float)

    def jprod(self, x, v):
        """Evaluate Jacobian-vector product at x with p."""
        return np.zeros(self.m, dtype=np.float)

    def jtprod(self, x, v):
        """Evaluate transposed-Jacobian-vector product at x with p."""
        return np.zeros(self.n, dtype=np.float)
