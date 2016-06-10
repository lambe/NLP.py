# -*- coding: utf-8 -*-

from nlp.model.pysparsemodel import PySparseAmplModel
from nlp.model.augmented_lagrangian import AugmentedLagrangian
from nlp.model.linemodel import C1LineModel
from nlp.ls.linesearch import ArmijoLineSearch, LineSearchFailure

from nlp.tools.exceptions import UserExitRequest
from nlp.tools.norms import norm2
from nlp.tools.timing import cputime

try:
    from hsl.solvers.pyma57 import PyMa57Solver as LBLSolver
except ImportError:
    from hsl.solvers.pyma27 import PyMa27Solver as LBLSolver

import pysparse.sparse.pysparseMatrix as ps

import logging
import math
import numpy as np
import sys

np.set_printoptions(precision=16, formatter={'float': '{:0.8g}'.format})


class RegSQPSolver(object):
    """Regularized SQP method for equality-constrained optimization."""

    def __init__(self, model, **kwargs):
        u"""Regularized SQP framework for an equality-constrained problem.

        :keywords:
            :model: `NLPModel` instance.
            :abstol: Absolute stopping tolerance
            :reltol: relative required accuracy for ‖[g-J'y ; c]‖
            :theta: sufficient decrease condition for the inner iterations
            :prox: initial proximal parameter
            :penalty: initial penalty parameter
            :itermax: maximum number of iterations allowed
            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

        """
        self.model = model
        self.x = model.x0.copy()
        self.y = np.ones(model.m)

        self.abstol = kwargs.get('abstol', 1.0e-7)
        self.reltol = kwargs.get('reltol', 1.0e-7)
        self.theta = kwargs.get('theta', 0.99)
        self.itermax = kwargs.get('maxiter', max(100, 10 * model.n))

        # attributed related to quasi-Newton variants
        self.save_g = kwargs.get('save_g', False)
        self.x_old = None
        self.gL_old = None

        # Max number of times regularization parameters are increased.
        self.bump_max = kwargs.get('bump_max', 5)

        self.itn = 0
        self.K = None
        self.LBL = None
        self.short_status = "unknown"
        self.status = "unk"
        self.tsolve = 0.0

        # Set regularization parameters.
        self.prox_min = 1.0e-3  # used when increasing the prox parameter
        self.penalty_min = 1.0e-8
        prox = max(0.0, kwargs.get('prox', 0.0))
        penalty = max(self.penalty_min, kwargs.get('penalty', 1.0))
        self.merit = AugmentedLagrangian(model,
                                         penalty=1./penalty,
                                         prox=prox,
                                         xk=model.x0.copy())
        self.epsilon = 10. / penalty
        self.prox_factor = 10.  # increase factor during inertia correction
        self.penalty_factor = 10.  # increase factor during regularization

        # Initialize format strings for display
        self.hformat = "%-5s  %8s  %7s  %7s  %6s  %8s  %8s"
        self.header = self.hformat % (
            "iter", "f", u"‖c‖", u"‖∇L‖", "inner", u"ρ", u"δ")

        self.format = "%-5d  %8.1e  %7.1e  %7.1e  %6d  %8.1e  %8.1e"
        self.format0 = "%-5d  %8.1e  %7.1e  %7.1e  %6s  %8.1e  %8.1e"

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'nlp.regsqp')
        self.log = logging.getLogger(logger_name)
        self.log.addHandler(logging.NullHandler())
        self.log.propagate = False
        return

    def assemble_linear_system(self, x, y):
        u"""Assemble main saddle-point matrix.

        [ H+ρI      J' ] [∆x] = [ -g + J'y ]
        [    J     -δI ] [∆y]   [ -c       ]

        For now H is the exact Hessian of the Lagrangian.
        """
        self.log.debug('assembling linear system')

        # Some shortcuts for convenience
        n = self.model.n
        m = self.model.m
        self.K = ps.PysparseMatrix(nrow=n+m, ncol=n+m,
                                   sizeHint=model.nnzh + model.nnzj + m,
                                   symmetric=True)

        # contribution of the Hessian
        H = model.hess(x, z=y)
        (val, irow, jcol) = H.find()
        self.K.put(val, irow.tolist(), jcol.tolist())

        # add primal regularization
        if self.merit.prox > 0:
            self.K.addAt(self.merit.prox * np.ones(n), range(n), range(n))

        # contribution of the Jacobian
        J = model.jac(x)
        (val, irow, jcol) = J.find()
        self.K.put(val, (n + irow).tolist(), jcol.tolist())

        # dual regularization
        self.K.put(-1. / self.merit.penalty * np.ones(m),
                   range(n, n + m), range(n, m + n))
        return

    def initialize_rhs(self):
        """Allocate an empty vector to store the rhs of Newton systems."""
        return np.empty(self.model.n + self.model.m)

    def update_rhs(self, rhs, g, J, y, c):
        """Set the rhs of Newton system according to current information."""
        n = self.model.n
        rhs[:n] = -g + J.T * y
        rhs[n:] = -c
        return

    def assemble_rhs(self, g, J, y, c):
        """Set the rhs of Newton system according to current information."""
        return np.concatenate((-g + J.T * y, -c))

    def new_penalty(self, Fnorm):
        """Return updated penalty parameter value."""
        alpha = 0.9
        gamma = 1.1
        penalty = max(min(Fnorm,
                          min(alpha * self.merit.penalty,
                              self.merit.penalty**gamma)),
                      self.penalty_min)
        return penalty

    def solve_linear_system(self, rhs, itref_thresh=1.0e-7, nitref=1):
        u"""Compute a step by solving Newton's equations.

        Use a direct method to solve the symmetric and indefinite system

        [ H+ρI      J' ] [∆x] = [ -g + J'y ]
        [    J     -δI ] [∆y]   [ -c       ].

        We increase ρ until the inertia indicates that H+ρI is positive
        definite on the nullspace of J and increase δ in case the matrix is
        singular because the rank deficiency in J.
        """
        self.log.debug('solving linear system')
        nvar = self.model.nvar
        ncon = self.model.ncon

        self.LBL = LBLSolver(self.K, factorize=True)
        second_order_sufficient = self.LBL.inertia == (nvar, ncon, 0)
        full_rank = self.LBL.isFullRank

        nb_bump = 0
        tired = nb_bump > self.bump_max
        while not (second_order_sufficient and full_rank) and not tired:

            if not second_order_sufficient:
                self.log.debug("further convexifying model")

                if self.merit.prox == 0.0:
                    self.merit.prox = self.prox_min
                    self.K.addAt(self.merit.prox * np.ones(nvar),
                                 range(nvar), range(nvar))
                else:
                    self.merit.prox *= self.prox_factor + 1
                    self.K.addAt(
                        self.prox_factor * self.merit.prox * np.ones(nvar),
                        range(nvar), range(nvar))

            if not full_rank:
                self.log.debug("further regularizing")
                # further regularize; this isn't quite supported by theory
                # the augmented Lagrangian uses 1/δ
                self.K.addAt(
                    -self.penalty_factor / self.merit.penalty * np.ones(ncon),
                    range(nvar, nvar + ncon), range(nvar, nvar + ncon))
                self.merit.penalty /= self.penalty_factor + 1

            self.LBL = LBLSolver(self.K, factorize=True)
            second_order_sufficient = self.LBL.inertia == (nvar, ncon, 0)
            full_rank = self.LBL.isFullRank
            nb_bump += 1
            tired = nb_bump > self.bump_max

        if not second_order_sufficient:
            self.log.info("unable to convexify sufficiently")
            status = '    Unable to convexify sufficiently.'
            short_status = 'cnvx'
            solved = False
            dx = None
            dy = None
            return status, short_status, solved, dx, dy

        if not full_rank:
            self.log.info("unable to regularize sufficiently")
            status = '    Unable to regularize sufficiently.'
            short_status = 'degn'
            solved = False
            dx = None
            dy = None
            return status, short_status, solved, dx, dy

        self.LBL.solve(rhs)
        self.LBL.refine(rhs, nitref=nitref)
        (dx, dy) = self.get_dx_dy(self.LBL.x)
        self.log.debug("residual norm: %3.2e", norm2(self.LBL.residual))
        status = None
        short_status = None
        solved = True
        return status, short_status, solved, dx, dy

    def get_dx_dy(self, step):
        """Split `step` into steps along x and y.

        Outputs are *references*, not copies.
        """
        return (step[:self.model.n], -step[self.model.n:])
        # n = self.model.n
        # dx = step[:n]
        # dy = -step[n:]
        # return (dx, dy)

    def solve_inner(self, Fnorm0, x, y, g, J, c, gL, Fnorm, gLnorm, cnorm):
        u"""Perform a sequence of inner iterations.

        The objective of the inner iterations is to identify an improved
        iterate w+ = (x+, y+) such that the optimality residual satisfies
        ‖F(w+)‖ ≤ Θ ‖F(w)‖ + ϵ.
        The improved iterate is identified by minimizing the proximal augmented
        Lagrangian.
        """
        self.log.debug('starting inner iterations with target %7.1e',
                       self.theta * Fnorm0 + self.epsilon)
        self.log.info('%-3d %7.1e %7.1e %7.1e %7.1e %7.1e',
                      self.itn, Fnorm, gLnorm, cnorm,
                      self.merit.prox, self.merit.penalty)

        ls_fmt = "%7.1e  %8.1e"
        gLnorm0 = gLnorm
        cnorm0 = cnorm
        failure = False
        finished = Fnorm <= self.theta * Fnorm0 + self.epsilon
        tired = self.itn > self.itermax
        while not (failure or finished or tired):

            self.x_old = x.copy()
            self.gL_old = gL.copy()

            # compute step
            self.assemble_linear_system(x, y)
            # self.update_rhs(self.rhs, g, J, y, c)
            rhs = self.assemble_rhs(g, J, y, c)

            status, short_status, solved, dx, _ = self.solve_linear_system(rhs)

            if not solved:
                failure = True
                continue

            # Step 4: Armijo backtracking linesearch
            self.merit.pi = y
            line_model = C1LineModel(self.merit, x, dx)
            ls = ArmijoLineSearch(line_model, bkmax=5, decr=1.75)

            try:
                for step in ls:
                    self.log.debug(ls_fmt, step, ls.trial_value)

                x = ls.iterate
                g = model.grad(x)
                J = model.jop(x)
                c = model.cons(x) - model.Lcon
                cnorm = norm2(c)
                gL = g - J.T * y
                y_al = y - c * self.merit.penalty
                gphi = g - J.T * y_al
                gphi_norm = norm2(gphi)

                if gphi_norm <= self.theta * gLnorm0 + 0.5 * self.epsilon:
                    self.log.debug('optimality has sufficiently improved')
                    if cnorm <= self.theta * cnorm0 + 0.5 * self.epsilon:
                        self.log.debug('feasibility has sufficiently improved')
                        y = y_al
                    else:
                        self.merit.penalty *= 10

                self.itn += 1
                Fnorm = math.sqrt(gphi_norm**2 + cnorm**2)
                self.log.info('%-3d %7.1e %7.1e %7.1e %7.1e %7.1e',
                              self.itn, Fnorm, gphi_norm, cnorm,
                              self.merit.prox, self.merit.penalty)

                finished = Fnorm <= self.theta * Fnorm0 + self.epsilon
                tired = self.itn > self.itermax

                try:
                    self.post_inner_iteration(x, gL)

                except UserExitRequest:
                    self.status = "User exit"
                    finished = True

            except LineSearchFailure:
                step_status = "Rej"
                failure = True

        solved = Fnorm <= self.theta * Fnorm0 + self.epsilon
        return x, y, g, J, c, gphi, gphi_norm, cnorm, solved


    def solve(self, **kwargs):

        # Transfer pointers for convenience.
        model = self.model
        x = self.x
        y = self.y
        self.short_status = "fail"
        self.status = "fail"
        self.tsolve = 0.0

        # Get initial objective value
        print 'x0: ', x
        f = f0 = model.obj(x)

        # Initialize right-hand side and coefficient matrix
        # of linear systems
        # rhs = self.initialize_rhs()

        g = model.grad(x)
        J = model.jop(x)
        c = model.cons(x) - model.Lcon
        cnorm = cnorm0 = norm2(c)

        gL = g - J.T * y
        gLnorm = gLnorm0 = norm2(gL)
        Fnorm = Fnorm0 = math.sqrt(gLnorm**2 + cnorm**2)

        # Find a better initial point
        # self.assemble_linear_system(x, y, 0)
        # self.update_rhs(rhs, g, J, y, c)
        #
        # status, short_status, finished, nbumps, dx, dy = self.solve_linear_system(
        #     rhs, delta, J=J)
        #
        # xs = x + dx
        # ys = y + dy
        # gs = model.grad(xs)
        # Js = model.jop(xs)
        # cs = model.cons(xs) - model.Lcon
        # grad_Ls = gs - Js.T * ys
        # Fnorms = np.sqrt(norm2(grad_Ls)**2 + norm2(cs)**2)
        # if Fnorms <= Fnorm0:
        #     x += dx
        #     y += dy
        #     Fnorm = Fnorm0 = Fnorms
        #     g = gs.copy()
        #     J = model.jop(x)
        #     c = cs.copy()

        # Initialize penalty parameter
        # delta = min(0.1, Fnorm0)

        # set stopping tolerance
        tol = self.reltol * Fnorm0 + self.abstol

        optimal = Fnorm <= tol
        if optimal:
            status = 'Optimal solution found'
            short_status = 'opt'

        tired = self.itn > self.itermax
        finished = optimal or tired

        self.log.info(self.header)
        self.log.debug(norm2(g - J.T * y))
        self.log.info(self.format0, self.itn, f0, cnorm0, gLnorm0, "",
                      self.merit.prox, self.merit.penalty)

        self.itn = 0
        tick = cputime()

        # Main loop.
        while not finished:

            self.x_old = x.copy()
            self.gL_old = gL.copy()

            # update penalty parameter
            self.merit.penalty = 1 / self.new_penalty(Fnorm)

            # compute extrapolation step
            self.assemble_linear_system(x, y)
            # self.update_rhs(rhs, g, J, y, c)
            rhs = self.assemble_rhs(g, J, y, c)

            status, short_status, solved, dx, dy = \
                self.solve_linear_system(rhs, J)

            # TODO: check that solved = True

            # check for acceptance of extrapolation step
            self.epsilon = 10 / self.merit.penalty
            x_trial = x + dx
            y_trial = y + dy
            g_trial = model.grad(x_trial)
            J_trial = model.jop(x_trial)
            c_trial = model.cons(x_trial) - model.Lcon

            gL_trial = g_trial - J_trial.T * y_trial
            gL_trial_norm = norm2(gL_trial)
            c_trial_norm = norm2(c_trial)
            F_trial_norm = math.sqrt(gL_trial_norm**2 + c_trial_norm**2)

            if F_trial_norm <= self.theta * Fnorm + self.epsilon:
                x = x_trial
                y = y_trial
                g = g_trial
                J = J_trial
                c = c_trial
                gL = gL_trial
                # gL_norm = gL_trial_norm
                cnorm = c_trial_norm
                Fnorm = F_trial_norm

                try:
                    self.post_iteration(x, gL)
                except UserExitRequest:
                    self.status = "User exit"
                    short_status = 'user'

            else:
                # perform a sequence of inner iterations
                # starting from the extrapolated step
                x, y, g, J, c, gL, gLnorm, cnorm, solved = \
                    self.solve_inner(Fnorm, x_trial, y_trial,
                                     g_trial, J_trial, c_trial, gL_trial,
                                     F_trial_norm, gL_trial_norm, c_trial_norm)

            # Update values of the new iterate and compute stopping criterion.
            # x = x_trial.copy()
            # y = y_trial.copy()
            # g = g_trial.copy()
            # J = model.jop(x_trial)
            # c = c_trial.copy()
            # cnorm = norm2(c)
            # grad_L = g - J.T * y
            # grad_L_norm = norm2(grad_L)
            # self.log.debug(' condition outer: %6.2e <= %6.2e',
            #                np.sqrt(grad_L_norm**2 + cnorm**2), theta * Fnorm + epsilon)
            # Fnorm = np.sqrt(grad_L_norm**2 + cnorm**2)

            self.itn += 1
            optimal = Fnorm <= tol
            tired = self.itn > self.itermax
            finished = optimal or tired
            if self.itn % 20 == 0:
                self.log.info(self.header)

            f = model.obj(x)
            # self.log.info(self.format, itn, f, cnorm, grad_L_norm, inner_iter,
            #               self.merit.prox, self.merit.penalty)

            if optimal:
                status = 'Optimal solution found'
                short_status = 'opt'
                finished = True
                continue

            if tired:
                status = 'Maximum number of iterations reached'
                short_status = 'iter'
                finished = True
                continue

        # Transfer final values to class members.
        self.tsolve = cputime() - tick
        self.x = x.copy()
        self.y = y.copy()
        self.f = f
        self.cnorm = cnorm
        self.gLnorm = gLnorm
        self.optimal = optimal
        self.status = status
        self.short_status = short_status
        return

    def post_iteration(self, x, g, **kwargs):
        """
        Override this method to perform additional work at the end of a
        major iteration. For example, use this method to restart an
        approximate Hessian.
        """
        return None

    def post_inner_iteration(self, x, g, **kwargs):
        """
        Override this method to perform additional work at the end of a
        minor iteration. For example, use this method to restart an
        approximate Hessian.
        """
        return None


if __name__ == '__main__':

    # Create root logger.
    log = logging.getLogger('nlp.regsqp')
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(name)-15s %(levelname)-8s %(message)s')
    hndlr = logging.StreamHandler(sys.stdout)
    hndlr.setFormatter(fmt)
    log.addHandler(hndlr)

    # Configure the solver logger.
    sublogger = logging.getLogger('nlp.regsqp.solver')
    sublogger.setLevel(logging.DEBUG)
    sublogger.addHandler(hndlr)
    sublogger.propagate = False

    model = PySparseAmplModel("hs007.nl")         # Create a model
    solver = RegSQPSolver(model)
    solver.solve()
    print 'x:', solver.x
    print solver.status