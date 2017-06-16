# -*- coding: utf-8 -*-
"""Long-step primal-dual interior-point method for convex QP.

From Algorithm IPF on p.110 of Stephen J. Wright's book
"Primal-Dual Interior-Point Methods", SIAM ed., 1997.
The method uses the augmented system formulation. These systems are solved
using MA27 or MA57.

D. Orban, Montreal 2009-2011.
"""
try:                            # To solve augmented systems
    from hsl.solvers.pyma57 import PyMa57Solver as LBLContext
except ImportError:
    from hsl.solvers.pyma27 import PyMa27Solver as LBLContext
from hsl.scaling.mc29 import mc29ad

from qr_mumps.solver import QRMUMPSSolver

from nlp.model.qpmodel import QPModel, LSQModel
from nlp.model.snlp import SlackModel
from pysparse.sparse import PysparseMatrix
from pykrylov.linop.linop import PysparseLinearOperator
from nlp.tools.norms import norm2, norm_infty, normest
from nlp.tools.timing import cputime
import logging
import numpy as np


class RegQPInteriorPointSolver(object):
    u"""Solve a QP with the primal-dual-regularized interior-point method.

    Solve a convex quadratic program of the form::

       minimize    q + cᵀx + ½ xᵀHx
       subject to  Ax - b = 0                                  (QP)
                   l ≤ x ≤ u

    where Q is a symmetric positive semi-definite matrix. Any
    quadratic program may be converted to the above form by instantiation
    of the `SlackModel` class. The conversion to the slack formulation
    is mandatory in this implementation.

    The method is a variant of Mehrotra's predictor-corrector method where
    steps are computed by solving the primal-dual system in augmented form.
    A long-step variant is also available.

    Primal and dual regularization parameters may be specified by the user
    via the opional keyword arguments `primal_reg` and `dual_reg`. Both should be
    positive real numbers and should not be "too large". By default they
    are initialized to 1.0 and updated at each iteration.

    Problem scaling options are provided through the `scale_type` key word.

    Advantages of this method are that it is not sensitive to dense columns
    in A, no special treatment of the unbounded variables x is required,
    and a sparse symmetric quasi-definite system of equations is solved at
    each iteration. The latter, although indefinite, possesses a
    Cholesky-like factorization. Those properties make the method
    typically more robust than a standard predictor-corrector
    implementation and the linear system solves are often much faster than
    in a traditional interior-point method in augmented form.
    """

    def __init__(self, qp, **kwargs):
        """Instantiate a primal-dual-regularized IP solver for ``qp``.

        :parameters:
            :qp:       a :class:`QPModel` instance or a :class:`SlackModel`
                       containing a :class:`QPModel`. Note that
                       :class:`LPModel`s are also acceptable

        :keywords:
            :scale_type: Perform row and column scaling of the constraint
                         matrix A prior to solution (default: `none`).

            :primal_reg_init: Initial value of primal regularization parameter
                              (default: `1.0`).

            :primal_reg_min: Minimum value of primal regularization parameter
                             (default: `1.0e-8`).

            :dual_reg_init: Initial value of dual regularization parameter
                            (default: `1.0`).

            :dual_reg_min: Minimum value of dual regularization parameter
                           (default: `1.0e-8`).

            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

            :logger_name: Name of a logger to control output.

            :estimate_cond: Estimate the matrix condition number when solving
                            the linear system (default: `False`)

            :itermax: Max number of iterations. (default: max(100,10*qp.n))

            :mehrotra_pc: Use Mehrotra's predictor-corrector strategy
                          to update the step (default: `True`) If `False`,
                          use a variant of the long-step method.

            :stoptol: The convergence tolerance (default: 1.0e-6)
        """
        qp = self.check_problem_type(qp)

        # Grab logger if one was configured.
        logger_name = kwargs.get('logger_name', 'nlp.cqp')
        self.log = logging.getLogger(logger_name)

        # Either none, abs, or mc29
        self.scale_type = kwargs.get('scale_type', 'none')

        self.qp = qp
        print qp        # Let the user know we have started

        # Solver cannot support QPs with fixed variables at this time
        if qp.nfixedB > 0:
            msg = 'The QP model must not contain fixed variables'
            raise ValueError(msg)

        # Compute the size of the linear system in the problem,
        # i.e., count the number of finite bounds and fixed variables
        # present in the problem to determine the true size
        self.n = qp.n
        self.m = qp.m
        self.nl = qp.nlowerB + qp.nrangeB
        self.nu = qp.nupperB + qp.nrangeB
        if self.use_lsq:
            self.p = qp.model.p
        else:
            self.p = 0

        # Some useful index lists for associating variables with bound
        # multipliers
        self.all_lb = qp.lowerB + qp.rangeB
        self.all_lb.sort()
        self.all_ub = qp.upperB + qp.rangeB
        self.all_ub.sort()

        # Compute indices of the range variables within the all_lb and
        # all_ub arrays (used in the initial point calculation)
        self.range_in_lb = []
        self.range_in_ub = []
        for k in qp.rangeB:
            self.range_in_lb.append(self.all_lb.index(k))
            self.range_in_ub.append(self.all_ub.index(k))

        # Collect basic info about the problem.
        zero_pt = np.zeros(self.n)
        self.q = qp.obj(zero_pt)
        self.b = -qp.cons(zero_pt)
        self.c = qp.grad(zero_pt)
        self.A = qp.jac(zero_pt)
        self.H = qp.hess(zero_pt)
        if self.use_lsq:
            zero_pt_r = np.zeros(self.p)
            self.d = -qp.lsq_cons(zero_pt, zero_pt_r)
            self.C = qp.lsq_jac(zero_pt)
            if not isinstance(self.C, PysparseMatrix):
                raise TypeError('Matrix C must be a Pysparse matrix.')

            self.normd = norm2(self.d)
            self.normC = self.C.matrix.norm('fro')
        else:
            self.d = np.zeros(0, dtype=np.float)
            self.normd = 0.
            self.C = PysparseMatrix(nrow=0, ncol=self.n, sizeHint=0, symmetric=False)
            self.normC = 0.

        if not isinstance(self.A, PysparseMatrix):
            raise TypeError('Matrices H and A must be Pysparse matrices.')

        # A few useful norms to measure algorithm convergence
        self.normb = norm2(self.b)
        self.normc = norm2(self.c)
        self.normA = self.A.matrix.norm('fro')
        self.normH = self.H.matrix.norm('fro')

        # It will be more efficient to keep the diagonal of H around.
        self.diagH = self.H.take(range(self.n))

        # We perform the analyze phase on the augmented system only once.
        # self.lin_solver will be initialized in solve().
        self.lin_solver = None

        # Set regularization parameters.
        self.primal_reg_init = kwargs.get('primal_reg_init', 1.0)
        self.primal_reg_min = kwargs.get('primal_reg_min',1.0e-8)
        self.dual_reg_init = kwargs.get('dual_reg_init', 1.0)
        self.dual_reg_min = kwargs.get('dual_reg_min',1.0e-8)

        # Check input regularization parameters.
        if self.primal_reg_init < 0.0 or self.dual_reg_init < 0.0:
            raise ValueError('Regularization parameters must be nonnegative.')
        if self.primal_reg_min < 0.0 or self.dual_reg_min < 0.0:
            raise ValueError('Minimum regularization parameters must be nonnegative.')

        # Set working regularization values
        if self.primal_reg_init < self.primal_reg_min:
            self.primal_reg = self.primal_reg_min
        else:
            self.primal_reg = self.primal_reg_init

        if self.dual_reg_init < self.dual_reg_min:
            self.dual_reg = self.dual_reg_min
        else:
            self.dual_reg = self.dual_reg_init

        # Max number of times regularization parameters are increased.
        self.bump_max = kwargs.get('bump_max', 5)

        # Parameters for the LBL solve and iterative refinement
        self.itref_threshold = 1.e-5
        self.nitref = 5

        # Initialize format strings for display
        fmt_hdr = '%-4s  %-9s' + '  %-8s' * 7
        self.header = fmt_hdr % ('Iter', 'qpObj', 'pFeas', 'dFeas', 'Mu',
                                 'rho_s', 'delta_t', 'alpha_p', 'alpha_d')
        self.format = '%-4d  %-9.2e' + '  %-8.2e' * 7
        self.format0 = '%-4d  %-9.2e' + '  %-8.2e' * 5 + '  %-8s' * 2

        # Additional options to collect
        self.estimate_cond = kwargs.get('estimate_cond', False)
        self.itermax = kwargs.get('itermax', max(100, 10*qp.n))
        self.stoptol = kwargs.get('stoptol', 1.0e-6)
        self.mehrotra_pc = kwargs.get('mehrotra_pc', True)

        # Values to set in case of a problem failure
        self.status = 'fail'
        self.iter = 0
        self.tsolve = 0.0
        self.kktRes = 0.0
        self.qpObj = 0.0

        return

    def check_problem_type(self, qp):
        """Check that the input problem is a QPModel in slack form."""

        msg = 'Problem is not a valid QP.'
        if isinstance(qp, QPModel):
            qp = SlackModel(qp)
        elif isinstance(qp, SlackModel):
            if not isinstance(qp.model, QPModel):
                raise TypeError(msg)
        else:
            raise TypeError(msg)

        # Trigger special treatment for least-squares problems
        if isinstance(qp.model, LSQModel):
            self.use_lsq = True
        else:
            self.use_lsq = False

        return qp

    def scale(self):
        u"""Compute scaling factors for the problem.

        If the solver is run with scaling, this function computes scaling
        factors for the rows and columns of the Jacobian so that the scaled
        matrix entries have some nicer properties.

        In effect the original problem::

            minimize    q + cᵀx + ½ xᵀHx
            subject to  Ax - b = 0
                        (bounds)

        is converted to::

            minimize    (Cc)ᵀx + ½ xᵀ(CHCᵀ)x
            subject to  (RAC)x - Rb = 0
                        (bounds)

        where the diagonal matrices R and C contain row and column scaling
        factors respectively.

        The options for the :scale_type: keyword are as follows:
            :none:  Do not calculate scaling factors (The resulting 
                    variables are of type `None`)

            :abs:   Compute each row scaling as the largest absolute-value
                    row entry, then compute each column scaling as the largest
                    absolute-value column entry.

            :mc29:  Compute row and column scaling using the Harwell
                    Subroutine MC29, available from the HSL.py interface.

        """
        log = self.log
        m, n = self.A.shape
        self.row_scale = np.zeros(m)
        self.col_scale = np.zeros(n)
        (values, irow, jcol) = self.A.find()

        if self.scale_type == 'none':

            self.row_scale = None
            self.col_scale = None

        elif self.scale_type == 'abs':

            log.debug('Smallest and largest elements of A prior to scaling: ')
            log.debug('%8.2e %8.2e' % (np.min(np.abs(values)),
                                          np.max(np.abs(values))))

            row_max = np.zeros(m)
            col_max = np.zeros(n)

            # Find maximum row values
            for k in range(len(values)):
                row = irow[k]
                val = abs(values[k])
                row_max[row] = max(row_max[row], val)
            row_max[row_max == 0.0] = 1.0

            log.debug('Max row scaling factor = %8.2e' % np.max(row_max))

            # Modified A values after row scaling
            temp_values = values / row_max[irow]

            # Find maximum column values
            for k in range(len(temp_values)):
                col = jcol[k]
                val = abs(temp_values[k])
                col_max[col] = max(col_max[col], val)
            col_max[col_max == 0.0] = 1.0

            log.debug('Max column scaling factor = %8.2e' % np.max(col_max))

            # Invert the computed maximum values to obtain scaling factors
            # By convention, we multiply the matrices by these scaling factors
            self.row_scale = 1./row_max
            self.col_scale = 1./col_max

        elif self.scale_type == 'mc29':

            row_scale, col_scale, ifail = mc29ad(m, n, values, irow, jcol)

            # row_scale and col_scale contain in fact the logarithms of the
            # scaling factors. Modify these before storage
            self.row_scale = np.exp(row_scale)
            self.col_scale = np.exp(col_scale)

        else:

            log.info('Scaling option not recognized, no scaling will be applied.')
            self.row_scale = None
            self.col_scale = None

        # Apply the scaling factors to A, b, H, and c, if available
        if self.row_scale is not None and self.col_scale is not None:
            values *= self.row_scale[irow]
            values *= self.col_scale[jcol]
            self.A.put(values, irow, jcol)

            (values, irow, jcol) = self.H.find()
            values *= self.col_scale[irow]
            values *= self.col_scale[jcol]
            self.H.put(values, irow, jcol)

            self.b *= self.row_scale
            self.c *= self.col_scale

            # Recompute the norms to account for the problem scaling
            self.normb = norm2(self.b)
            self.normc = norm2(self.c)
            self.normA = self.A.matrix.norm('fro')
            self.normH = self.H.matrix.norm('fro')
            self.diagH = self.H.take(range(self.n))

            # Modify least-squares operator as well
            if self.use_lsq:
                (values, irow, jcol) = self.C.find()
                values *= self.col_scale[jcol]
                self.C.put(values, irow, jcol)

        return

    def solve(self, **kwargs):
        """Solve the problem.

        :returns:

            :x:            final iterate
            :y:            final value of the Lagrange multipliers associated
                           to `Ax = b`
            :zL:           final value of lower-bound multipliers
            :zU:           final value of upper-bound multipliers
            :iter:         total number of iterations
            :kktRes:       final relative residual
            :tsolve:       time to solve the QP
            :long_status:  string describing the exit status.
            :status:       short version of status, used for printing.

        """
        Lvar = self.qp.Lvar
        Uvar = self.qp.Uvar

        # Setup the problem
        self.scale()
        self.initialize_system()

        # Obtain initial point from Mehrotra's heuristic.
        (self.x, self.r, self.y, self.zL, self.zU) = self.set_initial_guess()
        x = self.x
        r = self.r
        y = self.y
        zL = self.zL
        zU = self.zU

        # Calculate optimality conditions at initial point
        kktRes = self.check_optimality()
        exitOpt = kktRes <= self.stoptol

        # Set up other stopping conditions
        exitInfeasP = False
        exitInfeasD = False
        exitIter = False
        iter = 0

        # Compute initial perturbation vectors
        # Note: r is primal feas. perturbation, s is dual feas. perturbation
        s = -self.dFeas / self.primal_reg
        t = -self.pFeas / self.dual_reg
        pr_infeas_count = 0
        du_infeas_count = 0
        rho_s = norm2(self.dFeas) / (1 + self.normc)
        rho_s_min = rho_s
        delta_t = norm2(self.pFeas) / (1 + self.normb)
        delta_t_min = delta_t
        mu0 = self.mu

        # Display header and initial point info
        self.log.info(self.header)
        self.log.info('-' * len(self.header))
        output_line = self.format0 % (iter, self.qpObj, self.pResid, self.dResid,
                                     self.dual_gap, rho_s, delta_t, ' ', ' ')
        self.log.info(output_line)

        setup_time = cputime()

        # Main loop.
        while not (exitOpt or exitInfeasP or exitInfeasD or exitIter):

            iter += 1

            # Compute augmented matrix and factorize it, checking for
            # degeneracy along the way
            self.set_system_matrix()

            # Exit immediately if regularization is unsuccessful.
            if not self.check_degeneracy():
                self.log.debug('Unable to regularize sufficiently. Exiting')
                break

            # Compute the right-hand side based on the step computation method
            if self.mehrotra_pc:
                # Compute affine-scaling step, i.e. with centering = 0.
                self.set_system_rhs(sigma=0.0)
                self.solve_system()
                dx_aff, dr_aff, dy_aff, dzL_aff, dzU_aff = self.extract_xyz(sigma=0.0)

                # Compute largest allowed primal and dual stepsizes.
                (alpha_p, index_p, is_up_p) = self.max_primal_step_length(dx_aff)
                (alpha_d, index_d, is_up_d) = self.max_dual_step_length(dzL_aff, dzU_aff)

                # Estimate complementarity after affine-scaling step.
                (mu_aff, _, _) = self._check_complementarity(x + alpha_p*dx_aff,
                    zL + alpha_d*dzL_aff, zU + alpha_d*dzU_aff)

                # Incorporate predictor information for corrector step.
                if self.mu > 0:
                    sigma = (mu_aff / self.mu)**3
                else:
                    sigma = 0.0

            else:
                # Use long-step method: Compute centering parameter.
                sigma = min(0.1, 100 * self.mu)

            # Solve augmented system.
            self.set_system_rhs(sigma=sigma)
            self.solve_system()
            dx, dr, dy, dzL, dzU = self.extract_xyz(sigma=sigma)

            # Update regularization parameters before calculating the 
            # step sizes
            self.dual_reg = max(self.dual_reg / 10, self.dual_reg_min)
            self.primal_reg = max(self.primal_reg / 10, self.primal_reg_min)

            (alpha_p, alpha_d) = self._compute_max_steps(dx, dzL, dzU)

            # Update iterates and perturbation vectors.
            x += alpha_p * dx
            r += alpha_d * dr
            y += alpha_d * dy
            zL += alpha_d * dzL
            zU += alpha_d * dzU
            s = (1 - alpha_p)*s + alpha_p*dx
            t = (1 - alpha_d)*t + alpha_d*dy

            sNorm = norm2(s)
            tNorm = norm2(t)
            rho_s = self.primal_reg * sNorm / (1 + self.normc)
            rho_s_min = min(rho_s_min, rho_s)
            delta_t = self.dual_reg * tNorm / (1 + self.normb)
            delta_t_min = min(delta_t_min, delta_t)

            # Check for dual infeasibility
            if self.mu < 0.01 * self.stoptol * mu0 and \
                    rho_s > rho_s_min * (1.e-6 / self.stoptol) :
                du_infeas_count += 1
                if du_infeas_count > 6:
                    exitInfeasD = True
            else:
                du_infeas_count = 0

            # Check for primal infeasibility
            if self.mu < 0.01 * self.stoptol * mu0 and \
                    delta_t > delta_t_min * (1.e-6 / self.stoptol) :
                pr_infeas_count += 1
                if pr_infeas_count > 6:
                    exitInfeasP = True
            else:
                pr_infeas_count = 0

            # Check for optimality at new point
            kktRes = self.check_optimality()
            exitOpt = kktRes <= self.stoptol

            # Check iteration limit
            exitIter = iter == self.itermax

            # Log updated point info
            if iter % 20 == 0:
                self.log.info(self.header)
                self.log.info('-' * len(self.header))
            output_line = self.format % (iter, self.qpObj, self.pResid, self.dResid,
                                         self.dual_gap, rho_s, delta_t,
                                         alpha_p, alpha_d)
            self.log.info(output_line)

        # Determine solution time
        tsolve = cputime() - setup_time

        self.log.info('-' * len(self.header))

        # Resolve why the iteration stopped and print status
        if exitOpt:
            long_status = 'Optimal solution found'
            status = 'opt'
        elif exitInfeasD:
            long_status = 'Problem seems to be (locally) dual infeasible'
            status = 'dInf'
        elif exitInfeasP:
            long_status = 'Problem seems to be (locally) primal infeasible'
            status = 'pInf'
        elif exitIter:
            long_status = 'Maximum number of iterations reached'
            status = 'iter'
        else:
            long_status = 'Problem could not be regularized sufficiently.'
            status = 'degn'

        self.log.info(long_status)
        self.unscale()

        # Transfer final values to class members.
        self.iter = iter
        self.kktRes = kktRes
        self.tsolve = tsolve
        self.long_status = long_status
        self.status = status

        return

    def set_initial_guess(self):
        u"""Compute initial guess according the Mehrotra's heuristic.

        Initial values of x are computed as the solution to the
        least-squares problem::

            minimize    ½ xᵀHx + ½||rᴸ||² + ½||rᵁ||²
            subject to  Ax - b = 0
                        rᴸ = x - l
                        rᵁ = u - x

        The solution is also the solution to the augmented system::

            [ H   Aᵀ   I   I] [x ]   [0 ]
            [ A   0    0   0] [y']   [b ]
            [ I   0   -I   0] [zᴸ'] = [l ]
            [ I   0    0  -I] [zᵁ']   [u ].

        Initial values for the multipliers y and z are chosen as the
        solution to the least-squares problem::

            minimize    ½ x'ᵀHx' + ½||zᴸ||² + ½||zᵁ||²
            subject to  Hx' + c - Aᵀy - zᴸ + zᵁ = 0
                        zᴸ = -(x - l)
                        zᵁ = -(u - x)

        which can be computed as the solution to the augmented system::

            [ H   Aᵀ   I   I] [x']   [-c]
            [ A   0    0   0] [y ]   [ 0]
            [ I   0   -I   0] [zᴸ] = [ l]
            [ I   0    0  -I] [zᵁ]   [ u].

        To ensure stability and nonsingularity when A does not have full row
        rank or H is singluar, the (1,1) block is perturbed by
        sqrt(self.primal_reg_min) * I and the (2,2) block is perturbed by
        sqrt(self.dual_reg_min) * I.

        The values of x', y', zᴸ', and zᵁ' are discarded after solving the
        linear systems.

        The values of x and z are subsequently adjusted to ensure they
        are strictly within their bounds. See [Methrotra, 1992] for details.
        """
        # n = self.n
        # m = self.m
        nl = self.nl
        nu = self.nu
        Lvar = self.qp.Lvar
        Uvar = self.qp.Uvar

        self.log.debug('Computing initial guess')

        # Let the class know we are initializing the problem for now
        self.initial_guess = True

        # Set up augmented system matrix
        self.set_system_matrix()

        # Analyze and factorize the matrix
        self.initialize_linear_solver()

        # Assemble first right-hand side
        self.set_system_rhs()

        # Solve system and collect solution
        self.solve_system()
        x, _, _, _, _ = self.extract_xyz()

        # Assemble second right-hand side
        self.set_system_rhs(dual=True)

        # Solve system and collect solution
        self.solve_system()
        _, r, y, zL_guess, zU_guess = self.extract_xyz(dual=True)

        # Use Mehrotra's heuristic to compute a strictly feasible starting
        # point for all x and z
        if nl > 0:
            rL_guess = x[self.all_lb] - Lvar[self.all_lb]
            drL = 1.5 + max(0.0, -1.5*np.min(rL_guess))
            dzL = 1.5 + max(0.0, -1.5*np.min(zL_guess))

            rL_shift = drL + 0.5*np.dot(rL_guess + drL, zL_guess + dzL) / \
                ((zL_guess + dzL).sum())
            zL_shift = dzL + 0.5*np.dot(rL_guess + drL, zL_guess + dzL) / \
                ((rL_guess + drL).sum())

            rL = rL_guess + rL_shift
            zL = zL_guess + zL_shift
            x[self.all_lb] = Lvar[self.all_lb] + rL
        else:
            zL = zL_guess

        if nu > 0:
            rU_guess = Uvar[self.all_ub] - x[self.all_ub]

            drU = 1.5 + max(0.0, -1.5*np.min(rU_guess))
            dzU = 1.5 + max(0.0, -1.5*np.min(zU_guess))

            rU_shift = drU + 0.5*np.dot(rU_guess + drU, zU_guess + dzU) / \
                ((zU_guess + dzU).sum())
            zU_shift = dzU + 0.5*np.dot(rU_guess + drU, zU_guess + dzU) / \
                ((rU_guess + drU).sum())

            rU = rU_guess + rU_shift
            zU = zU_guess + zU_shift
            x[self.all_ub] = Uvar[self.all_ub] - rU
        else:
            zU = zU_guess

        # An additional normalization step for the range-bounded variables
        #
        # This normalization prevents the shift computed in rL and rU from
        # taking us outside the feasible range, and yields the same final
        # x value whether we take (Lvar + rL*norm) or (Uvar - rU*norm) as x
        if nl > 0 and nu > 0:
            intervals = Uvar[self.qp.rangeB] - Lvar[self.qp.rangeB]
            norm_factors = intervals / (intervals + rL_shift + rU_shift)
            x[self.qp.rangeB] = Lvar[self.qp.rangeB] + rL[self.range_in_lb]*norm_factors

        # Initialization complete
        self.initial_guess = False

        # Check strict feasibility
        if not np.all(x > Lvar) or not np.all(x < Uvar) or \
        not np.all(zL > 0) or not np.all(zU > 0):
            raise ValueError('Initial point not strictly feasible')

        return (x, r, y, zL, zU)

    def max_primal_step_length(self, dx):
        """Compute the maximum step to the boundary in the primal variables.

        The function also returns the component index that produces the
        minimum steplength. (If the minimum steplength is 1, this value is
        set to -1.)
        """
        self.log.debug('Computing primal step length')
        xl = self.x[self.all_lb]
        xu = self.x[self.all_ub]
        dxl = dx[self.all_lb]
        dxu = dx[self.all_ub]
        l = self.qp.Lvar[self.all_lb]
        u = self.qp.Uvar[self.all_ub]
        eps = 1.e-20

        if self.nl == 0:
            alphaL_max = 1.0
        else:
            # If dxl == 0., shift it slightly to prevent division by zero
            dxl_mod = np.where(dxl == 0., eps, dxl)
            alphaL = np.where(dxl < 0, -(xl - l)/dxl_mod, 1.)
            alphaL_max = min(1.0, alphaL.min())

        if self.nu == 0:
            alphaU_max = 1.0
        else:
            # If dxu == 0., shift it slightly to prevent division by zero
            dxu_mod = np.where(dxu == 0., -eps, dxu)
            alphaU = np.where(dxu > 0, (u - xu)/dxu_mod, 1.)
            alphaU_max = min(1.0, alphaU.min())

        if min(alphaL_max,alphaU_max) == 1.0:
            return (1.0, -1, False)

        if alphaL_max < alphaU_max:
            alpha_max = alphaL_max
            ind_max = self.all_lb[np.argmin(alphaL)]
            is_upper = False
        else:
            alpha_max = alphaU_max
            ind_max = self.all_ub[np.argmin(alphaU)]
            is_upper = True

        return (alpha_max, ind_max, is_upper)

    def max_dual_step_length(self, dzL, dzU):
        """Compute the maximum step to the boundary in the dual variables."""
        self.log.debug('Computing dual step length')
        eps = 1.e-20

        if self.nl == 0:
            alphaL_max = 1.0
        else:
            # If dzL == 0., shift it slightly to prevent division by zero
            dzL_mod = np.where(dzL == 0., eps, dzL)
            alphaL = np.where(dzL < 0, -self.zL/dzL_mod, 1.)
            alphaL_max = min(1.0,alphaL.min())

        if self.nu == 0:
            alphaU_max = 1.0
        else:
            # If dzU == 0., shift it slightly to prevent division by zero
            dzU_mod = np.where(dzU == 0., -eps, dzU)
            alphaU = np.where(dzU < 0, -self.zU/dzU_mod, 1.)
            alphaU_max = min(1.0,alphaU.min())

        if min(alphaL_max,alphaU_max) == 1.0:
            return (1.0, -1, False)

        if alphaL_max < alphaU_max:
            alpha_max = alphaL_max
            ind_max = self.all_lb[np.argmin(alphaL)]
            is_upper = False
        else:
            alpha_max = alphaU_max
            ind_max = self.all_ub[np.argmin(alphaU)]
            is_upper = True

        return (alpha_max, ind_max, is_upper)

    def _compute_max_steps(self, dx, dzL, dzU):
        """Compute the maximum step lengths given the directions."""

        x = self.x
        zL = self.zL
        zU = self.zU
        Uvar = self.qp.Uvar
        Lvar = self.qp.Lvar

        # Compute largest allowed primal and dual stepsizes.
        (alpha_p, index_p, is_up_p) = self.max_primal_step_length(dx)
        (alpha_d, index_d, is_up_d) = self.max_dual_step_length(dzL, dzU)

        # Define fraction-to-the-boundary factor and compute the true
        # step sizes
        tau = max(.995, 1.0 - self.mu)

        if self.mehrotra_pc:
            # Compute actual stepsize using Mehrotra's heuristic.

            if index_p == index_d and is_up_p == is_up_d:
                # If both are -1, do nothing, since the step remains
                # strictly feasible and alpha_p = alpha_d = 1; otherwise,
                # there is a division by zero in Mehrotra's heuristic, so
                # we fall back on the standard fraction-to-boundary rule.
                if index_p != -1:
                    alpha_p *= tau
                    alpha_d *= tau
            else:
                mult = 0.01

                (mu_temp, _, _) = self._check_complementarity(x + alpha_p*dx,
                    zL + alpha_d*dzL, zU + alpha_d*dzU)

                # If alpha_p < 1.0, compute a gamma_p such that the
                # complementarity of the updated (x,z) pair is mult*mu_temp
                if index_p != -1:
                    if is_up_p:
                        ref_index = self.all_ub.index(index_p)
                        gamma_p = mult * mu_temp
                        gamma_p /= (zU[ref_index] + alpha_d*dzU[ref_index])
                        gamma_p -= (Uvar[index_p] - x[index_p])
                        gamma_p /= -(alpha_p*dx[index_p])
                    else:
                        ref_index = self.all_lb.index(index_p)
                        gamma_p = mult * mu_temp
                        gamma_p /= (zL[ref_index] + alpha_d*dzL[ref_index])
                        gamma_p -= (x[index_p] - Lvar[index_p])
                        gamma_p /= (alpha_p*dx[index_p])

                    # If mu_temp is very small, gamma_p = 1. is possible due to
                    # a cancellation error in the gamma_p calculation above.
                    # Therefore, set a maximum value of alpha_p < 1 to prevent
                    # division-by-zero errors later in the program.
                    alpha_p *= min(max(1 - mult, gamma_p), 1. - 1.e-8)

                # If alpha_d < 1.0, compute a gamma_d such that the
                # complementarity of the updated (x,z) pair is mult*mu_temp
                if index_d != -1:
                    if is_up_d:
                        ref_index = self.all_ub.index(index_d)
                        gamma_d = mult * mu_temp
                        gamma_d /= (Uvar[index_d] - x[index_d] - alpha_p*dx[index_d])
                        gamma_d -= zU[ref_index]
                        gamma_d /= (alpha_d*dzU[ref_index])
                    else:
                        ref_index = self.all_lb.index(index_d)
                        gamma_d = mult * mu_temp
                        gamma_d /= (x[index_d] + alpha_p*dx[index_d] - Lvar[index_d])
                        gamma_d -= zL[ref_index]
                        gamma_d /= (alpha_d*dzL[ref_index])

                    # If mu_temp is very small, gamma_d = 1. is possible due to
                    # a cancellation error in the gamma_d calculation above.
                    # Therefore, set a maximum value of alpha_d < 1 to prevent
                    # division-by-zero errors later in the program.
                    alpha_d *= min(max(1 - mult, gamma_d), 1. - 1.e-8)

        else:
            # Use the standard fraction-to-the-boundary rule
            alpha_p *= tau
            alpha_d *= tau

        return (alpha_p, alpha_d)

    def initialize_system(self):
        """Initialize the system matrix and right-hand side.

        The A and H blocks of the matrix are also put in place since they
        are common to all problems. (The C block is also included for least-
        squares problems.)
        """
        n = self.n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu

        self.sys_size = n + p + m + nl + nu
        size_hint = nl + nu + self.A.nnz + self.H.nnz + self.C.nnz
        size_hint += self.sys_size

        self.K = PysparseMatrix(size=self.sys_size, sizeHint=size_hint,
            symmetric=True)
        self.K[:n, :n] = self.H
        self.K[n:n+p, :n] = self.C
        self.K[n+p:n+p+m, :n] = self.A

        self.K.put(-1.0, range(n, n+p))

        self.rhs = np.zeros(self.sys_size)
        return

    def initialize_linear_solver(self):
        """Set up the linear solver, given the constructed matrix."""
        self.lin_solver = LBLContext(self.K,
            sqd=(self.primal_reg_min > 0.0 and self.dual_reg_min > 0.0))
        return

    def set_system_matrix(self):
        """Set up the linear system matrix."""
        n = self.n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu

        if self.initial_guess:
            self.log.debug('Setting up matrix for initial guess')

            self.K.put(self.diagH + self.primal_reg_min**0.5, range(n))
            self.K.put(-self.dual_reg_min**0.5, range(n+p,n+p+m))

            self.K.put(-1.0, range(n+p+m, n+p+m+nl+nu))
            self.K.put(1.0, range(n+p+m, n+p+m+nl), self.all_lb)
            self.K.put(1.0, range(n+p+m+nl, n+p+m+nl+nu), self.all_ub)

        else:
            self.log.debug('Setting up matrix for current iteration')
            Lvar = self.qp.Lvar
            Uvar = self.qp.Uvar
            x = self.x
            zL = self.zL
            zU = self.zU

            self.K.put(self.diagH + self.primal_reg, range(n))
            self.K.put(-self.dual_reg, range(n+p,n+p+m))

            self.K.put(Lvar[self.all_lb] - x[self.all_lb], range(n+p+m, n+p+m+nl))
            self.K.put(x[self.all_ub] - Uvar[self.all_ub], range(n+p+m+nl, n+p+m+nl+nu))
            self.K.put(zL**0.5, range(n+p+m, n+p+m+nl), self.all_lb)
            self.K.put(zU**0.5, range(n+p+m+nl, n+p+m+nl+nu), self.all_ub)

        return

    def check_degeneracy(self):
        """Return True if the system can be sufficiently regularized for
        LDL factorization and False otherwise."""
        self.lin_solver.factorize(self.K)
        if not self.lin_solver.isFullRank:
            nbump = 0
            while not self.lin_solver.isFullRank and nbump < self.bump_max:
                self.log.debug('Primal-Dual Matrix Rank Deficient' +
                                  '... bumping up reg parameters')
                nbump += 1
                self.primal_reg *= 100
                self.dual_reg *= 100
                self.update_system_matrix()
                self.lin_solver.factorize(self.K)

        return self.lin_solver.isFullRank

    def update_system_matrix(self):
        """Update the linear system matrix with the new regularization
        parameters. This is a helper method when checking the system for
        degeneracy."""
        n = self.n
        m = self.m
        p = self.p
        self.log.debug('Updating matrix')
        self.K.put(self.diagH + self.primal_reg, range(n))
        self.K.put(-self.dual_reg, range(n+p,n+p+m))
        return

    def set_system_rhs(self, **kwargs):
        """Set up the linear system right-hand side."""
        n = self.n
        m = self.m
        p = self.p
        nl = self.nl
        self.log.debug('Setting up linear system right-hand side')

        if self.initial_guess:
            self.rhs[n+p+m:n+p+m+nl] = self.qp.Lvar[self.all_lb]
            self.rhs[n+p+m+nl:] = self.qp.Uvar[self.all_ub]
            if not kwargs.get('dual',False):
                # Primal initial point RHS
                self.rhs[:n] = 0.
                self.rhs[n:n+p] = self.d
                self.rhs[n+p:n+p+m] = self.b
            else:
                # Dual initial point RHS
                self.rhs[:n] = -self.c
                self.rhs[n:n+p] = self.d
                self.rhs[n+p:n+p+m] = 0.
        else:
            sigma = kwargs.get('sigma',0.0)
            self.rhs[:n] = -self.dFeas
            self.rhs[n:n+p] = -self.lsqRes
            self.rhs[n+p:n+p+m] = -self.pFeas
            self.rhs[n+p+m:n+p+m+nl] = -self.lComp + sigma*self.mu
            self.rhs[n+p+m:n+p+m+nl] *= self.zL**-0.5
            self.rhs[n+p+m+nl:] = self.uComp - sigma*self.mu
            self.rhs[n+p+m+nl:] *= self.zU**-0.5

        return

    def solve_system(self):
        """Solve the augmented system with current right-hand side.

        The solution may be iteratively refined based on solver options
        self.itref_threshold and self.nitref.

        The self.lin_solver object contains all of the solution data.
        """
        self.log.debug('Solving linear system')
        self.lin_solver.solve(self.rhs)
        self.lin_solver.refine(self.rhs, tol=self.itref_threshold, nitref=self.nitref)

        # Estimate matrix l2-norm condition number.
        if self.estimate_cond:
            rhsNorm = norm2(rhs)
            solnNorm = norm2(self.lin_solver.x)
            Kop = PysparseLinearOperator(self.K, symmetric=True)
            normK, _ = normest(Kop, tol=1.0e-3)
            if rhsNorm > 0 and solnNorm > 0:
                self.cond_est = solnNorm * normK / rhsNorm
            else:
                self.cond_est = 1.0

        return

    def extract_xyz(self, **kwargs):
        """Return the partitioned solution vector"""
        n = self.n
        m = self.m
        p = self.p
        nl = self.nl

        x = self.lin_solver.x[:n].copy()
        r = -self.lin_solver.x[n:n+p].copy()
        y = -self.lin_solver.x[n+p:n+p+m].copy()
        if self.initial_guess:
            zL = -self.lin_solver.x[n+p+m:n+p+m+nl].copy()
            zU = self.lin_solver.x[n+p+m+nl:].copy()
        else:
            zL = -(self.zL**0.5)*self.lin_solver.x[n+p+m:n+p+m+nl].copy()
            zU = (self.zU**0.5)*self.lin_solver.x[n+p+m+nl:].copy()

        return x,r,y,zL,zU

    def check_optimality(self):
        """Compute feasibility and complementarity for the current point"""
        x = self.x
        r = self.r
        y = self.y
        zL = self.zL
        zU = self.zU
        Lvar = self.qp.Lvar
        Uvar = self.qp.Uvar

        # Residual and complementarity vectors
        Hx = self.H*x
        self.qpObj = self.q + np.dot(self.c,x) + 0.5*np.dot(x,Hx) + 0.5*np.dot(r,r)
        self.pFeas = self.A*x - self.b
        if self.use_lsq:
            self.lsqRes = self.C*x + r - self.d
        else:
            self.lsqRes = np.zeros(0, dtype=np.float)
        self.dFeas = Hx + self.c - y*self.A - r*self.C
        self.dFeas[self.all_lb] -= zL
        self.dFeas[self.all_ub] += zU

        (self.mu, self.lComp, self.uComp) = self._check_complementarity(x, zL, zU)

        pFeasNorm = norm2(self.pFeas)
        dFeasNorm = norm2(self.dFeas)
        lsqNorm = norm2(self.lsqRes)

        # Scaled residual norms and duality gap
        norm_sum = self.normA + self.normH + self.normC
        self.pResid = pFeasNorm / (1 + self.normb + norm_sum)
        self.dResid = dFeasNorm / (1 + self.normc + norm_sum)
        self.lsqResid = lsqNorm / (1 + self.normd + norm_sum)
        self.dual_gap = self.mu / (1 + abs(np.dot(self.c,x)) + norm_sum)

        # Overall residual for stopping condition
        return max(self.pResid, self.lsqResid, self.dResid, self.dual_gap)

    def _check_complementarity(self, x, zL, zU):
        """Compute the complementarity given x, zL, and zU."""
        lComp = zL*(x[self.all_lb] - self.qp.Lvar[self.all_lb])
        uComp = zU*(self.qp.Uvar[self.all_ub] - x[self.all_ub])

        if (self.nl + self.nu) > 0:
            mu = (lComp.sum() + uComp.sum()) / (self.nl + self.nu)
        else:
            mu = 0.0

        return (mu, lComp, uComp)

    def unscale(self):
        """Undo scaling operations, if any, to return QP to original state."""

        if self.row_scale is not None and self.col_scale is not None:
            (values, irow, jcol) = self.A.find()
            values /= self.row_scale[irow]
            values /= self.col_scale[jcol]
            self.A.put(values, irow, jcol)

            (values, irow, jcol) = self.H.find()
            values /= self.col_scale[irow]
            values /= self.col_scale[jcol]
            self.H.put(values, irow, jcol)

            self.b /= self.row_scale
            self.c /= self.col_scale

            # Recompute the norms to account for the problem scaling
            self.normb = norm2(self.b)
            self.normc = norm2(self.c)
            self.normA = self.A.matrix.norm('fro')
            self.normH = self.H.matrix.norm('fro')
            self.diagH = self.H.take(range(self.n))

            # Modify least-squares operator, if any
            if self.use_lsq:
                (values, irow, jcol) = self.C.find()
                values /= self.col_scale[jcol]
                self.C.put(values, irow, jcol)

        return


class RegQPInteriorPointSolver2x2(RegQPInteriorPointSolver):
    """A 2x2 block variant of the regularized interior-point method.

    Linear system is based on the (reduced) 2x2 block system instead of
    the 3x3 block system.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a primal-dual-regularized IP solver for ``qp``.

        :parameters:
            :qp:       a :class:`QPModel` instance.

        :keywords:
            :scale_type: Perform row and column scaling of the constraint
                         matrix A prior to solution (default: `none`).

            :primal_reg_init: Initial value of primal regularization parameter
                              (default: `1.0`).

            :primal_reg_min: Minimum value of primal regularization parameter
                             (default: `1.0e-8`).

            :dual_reg_init: Initial value of dual regularization parameter
                            (default: `1.0`).

            :dual_reg_min: Minimum value of dual regularization parameter
                           (default: `1.0e-8`).

            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

            :logger_name: Name of a logger to control output.

            :estimate_cond: Estimate the matrix condition number when solving
                            the linear system (default: `False`)

            :itermax: Max number of iterations. (default: max(100,10*qp.n))

            :mehrotra_pc: Use Mehrotra's predictor-corrector strategy
                          to update the step (default: `True`) If `False`,
                          use a variant of the long-step method.

            :stoptol: The convergence tolerance (default: 1.0e-6)
        """
        super(RegQPInteriorPointSolver2x2, self).__init__(*args, **kwargs)

    def initialize_system(self):
        """Initialize the system matrix and right-hand side.

        The A and H blocks of the matrix are also put in place since they
        are common to all problems.
        """
        self.sys_size = self.n + self.m + self.p

        self.K = PysparseMatrix(size=self.sys_size,
            sizeHint=self.A.nnz + self.H.nnz + self.C.nnz + self.sys_size,
            symmetric=True)
        self.K[:self.n, :self.n] = self.H
        self.K[self.n:self.n+self.p, :self.n] = self.C
        self.K[self.n+self.p:self.n+self.p+self.m, :self.n] = self.A

        self.K.put(-1.0, range(self.n, self.n+self.p))

        self.rhs = np.zeros(self.sys_size)
        return

    def set_system_matrix(self):
        """Set up the linear system matrix."""
        n = self.n
        m = self.m
        p = self.p

        if self.initial_guess:
            self.log.debug('Setting up matrix for initial guess')

            new_diag = self.diagH + self.primal_reg_min**0.5
            new_diag[self.all_lb] += 1.0
            new_diag[self.all_ub] += 1.0

            self.K.put(new_diag, range(n))
            self.K.put(-self.dual_reg_min**0.5, range(n+p,n+p+m))

        else:
            self.log.debug('Setting up matrix for current iteration')

            x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
            u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]

            new_diag = self.diagH + self.primal_reg
            new_diag[self.all_lb] += self.zL / x_minus_l
            new_diag[self.all_ub] += self.zU / u_minus_x

            self.K.put(new_diag, range(n))
            self.K.put(-self.dual_reg, range(n+p,n+p+m))

        return

    def update_system_matrix(self):
        """Update the linear system matrix with the new regularization
        parameters. This is a helper method when checking the system for
        degeneracy."""

        self.log.debug('Updating matrix')
        x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
        u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]

        new_diag = self.diagH + self.primal_reg
        new_diag[self.all_lb] += self.zL / x_minus_l
        new_diag[self.all_ub] += self.zU / u_minus_x

        n = self.n
        m = self.m
        p = self.p
        self.K.put(new_diag, range(n))
        self.K.put(-self.dual_reg, range(n+p,n+p+m))
        return

    def set_system_rhs(self, **kwargs):
        """Set up the linear system right-hand side."""
        n = self.n
        m = self.m
        p = self.p
        self.log.debug('Setting up linear system right-hand side')

        if self.initial_guess:
            self.rhs[:n] = 0.
            self.rhs[self.all_lb] += self.qp.Lvar[self.all_lb]
            self.rhs[self.all_ub] += self.qp.Uvar[self.all_ub]
            if not kwargs.get('dual',False):
                # Primal initial point RHS
                self.rhs[n:n+p] = self.d
                self.rhs[n+p:n+p+m] = self.b
            else:
                # Dual initial point RHS
                self.rhs[:n] -= self.c
                self.rhs[n:n+p] = self.d
                self.rhs[n+p:n+p+m] = 0.
        else:
            sigma = kwargs.get('sigma',0.0)
            x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
            u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]

            self.rhs[:n] = -self.dFeas
            self.rhs[n:n+p] = -self.lsqRes
            self.rhs[n+p:n+p+m] = -self.pFeas
            self.rhs[self.all_lb] += -self.zL + sigma*self.mu/x_minus_l
            self.rhs[self.all_ub] += self.zU - sigma*self.mu/u_minus_x

        return

    def extract_xyz(self, **kwargs):
        """Return the partitioned solution vector"""
        n = self.n
        m = self.m
        p = self.p

        x = self.lin_solver.x[:n].copy()
        r = -self.lin_solver.x[n:n+p].copy()
        y = -self.lin_solver.x[n+p:n+p+m].copy()
        if self.initial_guess:
            zL = self.qp.Lvar[self.all_lb] - x[self.all_lb]
            zU = x[self.all_ub] - self.qp.Uvar[self.all_ub]
        else:
            sigma = kwargs.get('sigma',0.0)
            x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
            u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]

            zL = (-self.zL*(x_minus_l + x[self.all_lb]) + sigma*self.mu)
            zL /= x_minus_l
            zU = (-self.zU*(u_minus_x - x[self.all_ub]) + sigma*self.mu)
            zU /= u_minus_x

        return x,r,y,zL,zU


class RegQPInteriorPointSolverQR(RegQPInteriorPointSolver):
    u"""A regularized interior-point method using QR Factorization.

    Note that this solver only works for convex quadratic problems that have
    a diagonal Hessian; linear and least-squares problems are solved as well.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a primal-dual-regularized IP solver for ``qp``.

        :parameters:
            :qp:       a :class:`QPModel` instance.

        :keywords:
            :scale_type: Perform row and column scaling of the constraint
                         matrix A prior to solution (default: `none`).

            :primal_reg_init: Initial value of primal regularization parameter
                              (default: `1.0`).

            :primal_reg_min: Minimum value of primal regularization parameter
                             (default: `1.0e-8`).

            :dual_reg_init: Initial value of dual regularization parameter
                            (default: `1.0`).

            :dual_reg_min: Minimum value of dual regularization parameter
                           (default: `1.0e-8`).

            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

            :logger_name: Name of a logger to control output.

            :estimate_cond: Estimate the matrix condition number when solving
                            the linear system (default: `False`)

            :itermax: Max number of iterations. (default: max(100,10*qp.n))

            :mehrotra_pc: Use Mehrotra's predictor-corrector strategy
                          to update the step (default: `True`) If `False`,
                          use a variant of the long-step method.

            :stoptol: The convergence tolerance (default: 1.0e-6)
        """
        super(RegQPInteriorPointSolverQR, self).__init__(*args, **kwargs)

        # Decide whether we form the least squares problem for the primal
        # variables or the dual variables
        self.primal_solve = kwargs.get('primal_solve',False)

        # Decide whether we apply further scaling to the QR system
        self.extra_scale = kwargs.get('extra_scale',False)

    def initialize_system(self):
        """Initialize the system matrix and right-hand side.

        For the QR-based solver, the system is rectangular.
        """
        n = self.n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu

        self.sys_size = p + m + nl + nu
        size_hint = self.A.nnz + self.C.nnz + nl + nu

        if self.primal_solve:
            self.K = PysparseMatrix(nrow=self.sys_size+n, ncol=n,
                sizeHint=size_hint+n)

        else:
            # *** DEV NOTE: Pysparse doesn't contain a transpose operator,
            # so the code below constructs the *transpose* of the matrix in 
            # the least-squares system. At solve time, we swap the column and row
            # index arrays in the COO format to obtain the same effect
            self.K = PysparseMatrix(nrow=self.sys_size, ncol=self.sys_size+n,
                sizeHint=size_hint+self.sys_size)

        # Store the diagonal and main least-squares block separately for clear
        # code later on
        self.K_block = PysparseMatrix(nrow=self.sys_size, ncol=n,
            sizeHint=size_hint)
        self.K_diag_11 = np.zeros(n)
        self.K_diag_22 = np.zeros(self.sys_size)
        self.K_scaling = np.ones(self.sys_size + n)

        self.rhs = np.zeros(self.sys_size + n)
        return

    def initialize_linear_solver(self):
        """Set up the linear solver, given the constructed matrix."""

        if self.primal_solve:
            nrow = self.sys_size + self.n
            ncol = self.n
            kval, krow, kcol = self.K.find()

            self.lin_solver = QRMUMPSSolver((nrow, ncol, krow, kcol, kval),
                verbose=False)

        else:
            nrow = self.sys_size + self.n
            ncol = self.sys_size
            kval, krow, kcol = self.K.find()

            # *** DEV NOTE: rows and columns swapped because we cannot construct
            # the transpose matrix directly in Pysparse (which we would like)
            self.lin_solver = QRMUMPSSolver((nrow, ncol, kcol, krow, kval),
                verbose=False)

        self.lin_solver.analyze('metis')

        return

    def set_system_matrix(self):
        """Set up the linear system matrix."""
        n = self.n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu

        if self.initial_guess:
            self.log.debug('Setting up matrix for initial guess')

            # Main block column / row
            self.K_block[:p, :] = self.C
            self.K_block[p:p+m, :] = self.A
            self.K_block.put(1.0, range(p+m, p+m+nl), self.all_lb)
            self.K_block.put(1.0, range(p+m+nl, p+m+nl+nu), self.all_ub)

            # Diagonals
            self.K_diag_22[:p] = 1.0
            self.K_diag_22[p:p+m] = self.dual_reg_min**0.5
            self.K_diag_22[p+m:] = 1.0

            self.K_diag_11[:] = self.diagH + self.primal_reg_min**0.5

        else:
            self.log.debug('Setting up matrix for current iteration')
            Lvar = self.qp.Lvar
            Uvar = self.qp.Uvar
            x = self.x
            zL = self.zL
            zU = self.zU
            x_minus_l = x[self.all_lb] - Lvar[self.all_lb]
            u_minus_x = Uvar[self.all_ub] - x[self.all_ub]

            # Main block column / row
            self.K_block[:p, :] = self.C
            self.K_block[p:p+m, :] = self.A
            self.K_block.put(zL**0.5, range(p+m, p+m+nl), self.all_lb)
            self.K_block.put(zU**0.5, range(p+m+nl, p+m+nl+nu), self.all_ub)

            # Diagonals
            self.K_diag_22[:p] = 1.0
            self.K_diag_22[p:p+m] = self.dual_reg
            self.K_diag_22[p+m:p+m+nl] = x_minus_l
            self.K_diag_22[p+m+nl:] = u_minus_x

            self.K_diag_11[:] = self.diagH + self.primal_reg

        # Form rectangular K for the QR solve step
        self.K_diag_11 = self.K_diag_11**0.5
        self.K_diag_22 = self.K_diag_22**0.5

        if self.primal_solve:
            self.K[:self.sys_size, :] = self.K_block
            self.K.put(1.0, range(self.sys_size, self.sys_size+n), range(n))

            self.K_scaling[:self.sys_size] = 1./self.K_diag_22
            self.K_scaling[self.sys_size:] = self.K_diag_11
            self.K.row_scale(self.K_scaling)

            if self.extra_scale:
                self.K.col_scale(1./self.K_diag_11)

        else:
            self.K[:, :n] = self.K_block
            self.K.put(1.0, range(self.sys_size), range(n,n+self.sys_size))

            self.K_scaling[:n] = 1./self.K_diag_11
            self.K_scaling[n:] = self.K_diag_22
            self.K.col_scale(self.K_scaling)

            if self.extra_scale:
                self.K.row_scale(1./self.K_diag_22)

        return

    def check_degeneracy(self):
        """This function is not needed for the QR implementation."""
        return True

    def set_system_rhs(self, **kwargs):
        """Set up the linear system right-hand side."""

        super(RegQPInteriorPointSolverQR, self).set_system_rhs(**kwargs)
        self.rhs_cp = self.rhs.copy()

        if self.primal_solve:
            # Similar to RHS in parent class, but block order is
            # flipped and scaling is applied
            temp_vec = self.rhs[:self.n].copy()
            self.rhs[:self.sys_size] = self.rhs[self.n:]
            self.rhs[self.sys_size:] = temp_vec

            self.rhs[:self.sys_size] *= 1./self.K_diag_22
            self.rhs[self.sys_size:] *= 1./self.K_diag_11

        else:
            # Same RHS as parent class, but scaling terms attached
            self.rhs[:self.n] *= 1./self.K_diag_11
            self.rhs[self.n:] *= -1./self.K_diag_22

        return

    def solve_system(self):
        """Solve the linear system with qr_mumps."""

        kval, krow, kcol = self.K.find()

        if self.primal_solve:
            self.lin_solver.get_matrix_data(krow, kcol, kval)
        else:
            # *** DEV NOTE: rows and columns swapped because we cannot construct
            # the transpose matrix directly in Pysparse (which we would like)
            self.lin_solver.get_matrix_data(kcol, krow, kval)

        # Factorize and get the solution and residual vectors
        self.lin_solver.factorize()
        delta_x, res_vec, _ = self.lin_solver.solve(self.rhs, compute_residuals=True)
        self.soln_vec = np.zeros(self.sys_size + self.n)

        if self.primal_solve:
            if self.extra_scale:
                self.soln_vec[:self.n] = (1./self.K_diag_11)*delta_x
            else:
                self.soln_vec[:self.n] = delta_x
            self.soln_vec[self.n:] = (-1./self.K_diag_22)*res_vec[:self.sys_size]
        else:
            self.soln_vec[:self.n] = (1./self.K_diag_11)*res_vec[:self.n]
            if self.extra_scale:
                self.soln_vec[self.n:] = (1./self.K_diag_22)*delta_x
            else:
                self.soln_vec[self.n:] = delta_x

        # Compute the SQD system residual using the computed solution vector
        # ** NOTE: RHS has been scaled, need original RHS for baseline **
        new_rhs = self.rhs_cp.copy()
        new_rhs[:self.n] -= self.K_diag_11**2*self.soln_vec[:self.n] + self.soln_vec[self.n:]*self.K_block
        new_rhs[self.n:] += self.K_diag_22**2*self.soln_vec[self.n:] - self.K_block*self.soln_vec[:self.n]

        # Perform repeated iterative refinement if necessary to obtain
        # an accurate step
        old_norm = np.dot(self.rhs_cp, self.rhs_cp)**0.5
        new_norm = np.dot(new_rhs, new_rhs)**0.5
        while new_norm / old_norm > 1.e-8:
            # Apply appropriate scaling to updated RHS
            if self.primal_solve:
                temp_vec = new_rhs[:self.n].copy()
                new_rhs[:self.sys_size] = new_rhs[self.n:]
                new_rhs[self.sys_size:] = temp_vec

                new_rhs[:self.sys_size] *= 1./self.K_diag_22
                new_rhs[self.sys_size:] *= 1./self.K_diag_11

            else:
                new_rhs[:self.n] *= 1./self.K_diag_11
                new_rhs[self.n:] *= -1./self.K_diag_22

            # Second linear solve (no extra factorization necessary)
            delta_x, res_vec, _ = self.lin_solver.solve(new_rhs, compute_residuals=True)

            # Apply refinement step and check norms again
            if self.primal_solve:
                if self.extra_scale:
                    self.soln_vec[:self.n] += (1./self.K_diag_11)*delta_x
                else:
                    self.soln_vec[:self.n] += delta_x
                self.soln_vec[self.n:] += (-1./self.K_diag_22)*res_vec[:self.sys_size]
            else:
                self.soln_vec[:self.n] += (1./self.K_diag_11)*res_vec[:self.n]
                if self.extra_scale:
                    self.soln_vec[self.n:] += (1./self.K_diag_22)*delta_x
                else:
                    self.soln_vec[self.n:] += delta_x

            new_rhs = self.rhs_cp.copy()
            new_rhs[:self.n] -= self.K_diag_11**2*self.soln_vec[:self.n] + self.soln_vec[self.n:]*self.K_block
            new_rhs[self.n:] += self.K_diag_22**2*self.soln_vec[self.n:] - self.K_block*self.soln_vec[:self.n]

            # old_norm = np.dot(self.rhs_cp, self.rhs_cp)**0.5
            new_norm = np.dot(new_rhs, new_rhs)**0.5

        return

    def extract_xyz(self, **kwargs):
        """Return the partitioned solution vector"""
        n = self.n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu

        x = self.soln_vec[:n].copy()
        r = -self.soln_vec[n:n+p].copy()
        y = -self.soln_vec[n+p:n+p+m].copy()
        if self.initial_guess:
            zL = -self.soln_vec[n+p+m:n+p+m+nl].copy()
            zU = self.soln_vec[n+p+m+nl:].copy()
        else:
            zL = -(self.zL**0.5)*self.soln_vec[n+p+m:n+p+m+nl].copy()
            zU = (self.zU**0.5)*self.soln_vec[n+p+m+nl:].copy()

        return x,r,y,zL,zU


class RegQPInteriorPointSolver2x2QR(RegQPInteriorPointSolver2x2):
    """A 2x2 block variant of the regularized interior-point method using
    QR factorization to solve the linear system.

    Note that this solver only works for convex quadratic problems that have
    a diagonal Hessian; linear and least-squares problems are solved as well.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a primal-dual-regularized IP solver for ``qp``.

        :parameters:
            :qp:       a :class:`QPModel` instance.

        :keywords:
            :scale_type: Perform row and column scaling of the constraint
                         matrix A prior to solution (default: `none`).

            :primal_reg_init: Initial value of primal regularization parameter
                              (default: `1.0`).

            :primal_reg_min: Minimum value of primal regularization parameter
                             (default: `1.0e-8`).

            :dual_reg_init: Initial value of dual regularization parameter
                            (default: `1.0`).

            :dual_reg_min: Minimum value of dual regularization parameter
                           (default: `1.0e-8`).

            :bump_max: Max number of times regularization parameters are
                       increased when a factorization fails (default 5).

            :logger_name: Name of a logger to control output.

            :estimate_cond: Estimate the matrix condition number when solving
                            the linear system (default: `False`)

            :itermax: Max number of iterations. (default: max(100,10*qp.n))

            :mehrotra_pc: Use Mehrotra's predictor-corrector strategy
                          to update the step (default: `True`) If `False`,
                          use a variant of the long-step method.

            :stoptol: The convergence tolerance (default: 1.0e-6)
        """
        super(RegQPInteriorPointSolver2x2QR, self).__init__(*args, **kwargs)

        # Decide whether we form the least squares problem for the primal
        # variables or the dual variables
        self.primal_solve = kwargs.get('primal_solve',False)

        # Decide whether we apply further scaling to the QR system
        self.extra_scale = kwargs.get('extra_scale',False)

    def initialize_system(self):
        """Initialize the system matrix and right-hand side.

        For the QR-based solver, the system is rectangular.
        """
        n = self.n
        m = self.m
        p = self.p

        self.sys_size = p + m
        size_hint = self.A.nnz + self.C.nnz

        if self.primal_solve:
            self.K = PysparseMatrix(nrow=self.sys_size + n, ncol=n,
                sizeHint=size_hint+n)

        else:
            # *** DEV NOTE: Pysparse doesn't contain a transpose operator,
            # so the code below constructs the *transpose* of the matrix in 
            # the least-squares system. At solve time, we swap the column and row
            # index arrays in the COO format to obtain the same effect
            self.K = PysparseMatrix(nrow=self.sys_size, ncol=self.sys_size+n,
                sizeHint=size_hint+self.sys_size)

        # Store the diagonal and main least-squares block separately
        # for clear code later on
        self.K_block = PysparseMatrix(nrow=self.sys_size, ncol=n,
            sizeHint=size_hint)
        self.K_diag_11 = np.zeros(n)
        self.K_diag_22 = np.zeros(self.sys_size)
        self.K_scaling = np.ones(self.sys_size + n)

        self.rhs = np.zeros(self.sys_size + n)
        return

    def initialize_linear_solver(self):
        """Set up the linear solver, given the constructed matrix."""

        if self.primal_solve:
            nrow = self.sys_size + self.n
            ncol = self.n
            kval, krow, kcol = self.K.find()

            self.lin_solver = QRMUMPSSolver((nrow, ncol, krow, kcol, kval),
                verbose=False)

        else:
            nrow = self.sys_size + self.n
            ncol = self.sys_size
            kval, krow, kcol = self.K.find()

            # *** DEV NOTE: rows and columns swapped because we cannot construct
            # the transpose matrix directly in Pysparse (which we would like)
            self.lin_solver = QRMUMPSSolver((nrow, ncol, kcol, krow, kval),
                verbose=False)

        return

    def set_system_matrix(self):
        """Set up the linear system matrix."""
        n = self.n
        m = self.m
        p = self.p
        # nl = self.nl
        # nu = self.nu

        if self.initial_guess:
            self.log.debug('Setting up matrix for initial guess')

            # Main block column
            self.K_block[:p, :n] = self.C
            self.K_block[p:, :n] = self.A

            # Diagonals
            self.K_diag_22[:p] = 1.0
            self.K_diag_22[p:] = self.dual_reg_min**0.5

            self.K_diag_11[:] = self.diagH + self.primal_reg_min**0.5
            self.K_diag_11[self.all_lb] += 1.0
            self.K_diag_11[self.all_ub] += 1.0

        else:
            self.log.debug('Setting up matrix for current iteration')
            Lvar = self.qp.Lvar
            Uvar = self.qp.Uvar
            x = self.x
            zL = self.zL
            zU = self.zU
            x_minus_l = x[self.all_lb] - Lvar[self.all_lb]
            u_minus_x = Uvar[self.all_ub] - x[self.all_ub]

            # Main block column
            self.K_block[:p, :n] = self.C
            self.K_block[p:, :n] = self.A

            # Diagonals
            self.K_diag_22[:p] = 1.0
            self.K_diag_22[p:] = self.dual_reg

            self.K_diag_11 = self.diagH + self.primal_reg
            self.K_diag_11[self.all_lb] += zL / x_minus_l
            self.K_diag_11[self.all_ub] += zU / u_minus_x

        # Form rectangular K for the QR solve step
        self.K_diag_11 = self.K_diag_11**0.5
        self.K_diag_22 = self.K_diag_22**0.5

        if self.primal_solve:
            self.K[:self.sys_size, :] = self.K_block
            self.K.put(1.0, range(self.sys_size, self.sys_size+n), range(n))

            self.K_scaling[:self.sys_size] = 1./self.K_diag_22
            self.K_scaling[self.sys_size:] = self.K_diag_11
            self.K.row_scale(self.K_scaling)

            if self.extra_scale:
                self.K.col_scale(1./self.K_diag_11)

        else:
            self.K[:, :n] = self.K_block
            self.K.put(1.0, range(self.sys_size), range(n,self.sys_size+n))

            self.K_scaling[:n] = 1./self.K_diag_11
            self.K_scaling[n:] = self.K_diag_22
            self.K.col_scale(self.K_scaling)

            if self.extra_scale:
                self.K.row_scale(1./self.K_diag_22)

        return

    def check_degeneracy(self):
        """This function is not needed for the QR implementation."""
        return True

    def set_system_rhs(self, **kwargs):
        """Set up the linear system right-hand side."""

        super(RegQPInteriorPointSolver2x2QR, self).set_system_rhs(**kwargs)
        self.rhs_cp

        if self.primal_solve:
            # Similar to RHS in parent class, but block order is
            # flipped and scaling is applied
            temp_vec = self.rhs[:self.n].copy()
            self.rhs[:self.sys_size] = self.rhs[self.n:]
            self.rhs[self.sys_size:] = temp_vec

            self.rhs[:self.sys_size] *= 1./self.K_diag_22
            self.rhs[self.sys_size:] *= 1./self.K_diag_11

        else:
            # Same RHS as parent class, but scaling terms attached
            self.rhs[:self.n] *= 1./self.K_diag_11
            self.rhs[self.n:] *= -1./self.K_diag_22

        return

    def solve_system(self):
        """Solve the linear system with qr_mumps."""

        kval, krow, kcol = self.K.find()

        if self.primal_solve:
            self.lin_solver.get_matrix_data(krow, kcol, kval)
        else:
            # *** DEV NOTE: rows and columns swapped because we cannot construct
            # the transpose matrix directly in Pysparse (which we would like)
            self.lin_solver.get_matrix_data(kcol, krow, kval)

        # Factorize and get the solution and residual vectors
        self.lin_solver.factorize()
        delta_x, res_vec, _ = self.lin_solver.solve(self.rhs, compute_residuals=True)
        self.soln_vec = np.zeros(self.sys_size + self.n)

        if self.primal_solve:
            if self.extra_scale:
                self.soln_vec[:self.n] = (1./self.K_diag_11)*delta_x
            else:
                self.soln_vec[:self.n] = delta_x
            self.soln_vec[self.n:] = (-1./self.K_diag_22)*res_vec[:self.sys_size]
        else:
            self.soln_vec[:self.n] = (1./self.K_diag_11)*res_vec[:self.n]
            if self.extra_scale:
                self.soln_vec[self.n:] = (1./self.K_diag_22)*delta_x
            else:
                self.soln_vec[self.n:] = delta_x

        # Compute the SQD system residual using the computed solution vector
        # ** NOTE: RHS has been scaled, need original RHS for baseline **
        new_rhs = self.rhs_cp.copy()
        new_rhs[:self.n] -= self.K_diag_11**2*self.soln_vec[:self.n] + self.soln_vec[self.n:]*self.K_block
        new_rhs[self.n:] += self.K_diag_22**2*self.soln_vec[self.n:] - self.K_block*self.soln_vec[:self.n]

        # Perform repeated iterative refinement if necessary to obtain
        # an accurate step
        old_norm = np.dot(self.rhs_cp, self.rhs_cp)**0.5
        new_norm = np.dot(new_rhs, new_rhs)**0.5
        while new_norm / old_norm > 1.e-8:
            # Apply appropriate scaling to updated RHS
            if self.primal_solve:
                temp_vec = new_rhs[:self.n].copy()
                new_rhs[:self.sys_size] = new_rhs[self.n:]
                new_rhs[self.sys_size:] = temp_vec

                new_rhs[:self.sys_size] *= 1./self.K_diag_22
                new_rhs[self.sys_size:] *= 1./self.K_diag_11

            else:
                new_rhs[:self.n] *= 1./self.K_diag_11
                new_rhs[self.n:] *= -1./self.K_diag_22

            # Second linear solve (no extra factorization necessary)
            delta_x, res_vec, _ = self.lin_solver.solve(new_rhs, compute_residuals=True)

            # Apply refinement step and check norms again
            if self.primal_solve:
                if self.extra_scale:
                    self.soln_vec[:self.n] += (1./self.K_diag_11)*delta_x
                else:
                    self.soln_vec[:self.n] += delta_x
                self.soln_vec[self.n:] += (-1./self.K_diag_22)*res_vec[:self.sys_size]
            else:
                self.soln_vec[:self.n] += (1./self.K_diag_11)*res_vec[:self.n]
                if self.extra_scale:
                    self.soln_vec[self.n:] += (1./self.K_diag_22)*delta_x
                else:
                    self.soln_vec[self.n:] += delta_x

            new_rhs = self.rhs_cp.copy()
            new_rhs[:self.n] -= self.K_diag_11**2*self.soln_vec[:self.n] + self.soln_vec[self.n:]*self.K_block
            new_rhs[self.n:] += self.K_diag_22**2*self.soln_vec[self.n:] - self.K_block*self.soln_vec[:self.n]

            # old_norm = np.dot(self.rhs_cp, self.rhs_cp)**0.5
            new_norm = np.dot(new_rhs, new_rhs)**0.5

        return

    def extract_xyz(self, **kwargs):
        """Return the partitioned solution vector"""
        n = self.n
        m = self.m
        p = self.p

        x = self.soln_vec[:n].copy()
        r = -self.soln_vec[n:n+p].copy()
        y = -self.soln_vec[n+p:n+p+m].copy()
        if self.initial_guess:
            zL = self.qp.Lvar[self.all_lb] - x[self.all_lb]
            zU = x[self.all_ub] - self.qp.Uvar[self.all_ub]
        else:
            sigma = kwargs.get('sigma',0.0)
            x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
            u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]

            zL = (-self.zL*(x_minus_l + x[self.all_lb]) + sigma*self.mu)
            zL /= x_minus_l
            zU = (-self.zU*(u_minus_x - x[self.all_ub]) + sigma*self.mu)
            zU /= u_minus_x

        return x,r,y,zL,zU


class RegL1QPInteriorPointSolver(RegQPInteriorPointSolver):
    u"""Solve a QP with an L1 norm regularization of the variables.

    Solve a convex quadratic program of the form::

       minimize    q + cᵀx + ½ xᵀHx + λ||x||₁
       subject to  Ax - b = 0                                  (L1QP)
                   l ≤ x ≤ u

    where Q is a symmetric positive semi-definite matrix. Any
    quadratic program may be converted to the above form by instantiation
    of the `SlackModel` class. The conversion to the slack formulation
    is mandatory in this implementation.

    Note that the L1 norm does not apply to the slack variables, if any.

    For convenience, (L1QP) is transformed into the following form::

       minimize    q + cᵀx + ½ xᵀHx + λeᵀv
       subject to  Ax - b = 0                                  (L1QP2)
                   l ≤ x ≤ u
                  -v ≤ x ≤ v

    where e is a vector of ones. The added variables v are appended to
    the x vector in the solver code because they are treated similarly to
    x. Likewise, additional lower- and upper-bound multipliers are appended
    to those vectors in the original algorithm.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the problem, similar to the base solver."""
        super(RegL1QPInteriorPointSolver,self).__init__(*args, **kwargs)

        # Parameters specific to the L1 regularization
        self.original_n = self.qp.model.n
        self.lam = kwargs.get('lam',1.0)
        return

    def set_initial_guess(self):
        u"""Compute initial guess according to Mehrotra's heuristic.

        (finish documentation later)
        """
        nl = self.nl
        nu = self.nu
        on = self.original_n
        n = self.n
        Lvar = self.qp.Lvar
        Uvar = self.qp.Uvar

        self.log.debug('Computing initial guess')

        # Let the class know we are initializing the problem for now
        self.initial_guess = True

        # Set up augmented system matrix
        self.set_system_matrix()

        # Analyze and factorize the matrix
        self.initialize_linear_solver()

        # Assemble first right-hand side
        self.set_system_rhs()

        # Solve system and collect solution
        self.solve_system()
        x, _, _, _, _ = self.extract_xyz()

        # Assemble second right-hand side
        self.set_system_rhs(dual=True)

        # Solve system and collect solution
        self.solve_system()
        _, r, y, zL, zU = self.extract_xyz()

        # The following section is different for the L1 regularized problem.
        # We need to shift the x variables to be feasible with respect to the
        # bounds, and then shift the v variables to be feasible with respect
        # to the update x variables

        # Compute a strictly feasible starting point for the terms with fixed
        # bounds
        if nl > 0:
            rL_guess = x[self.all_lb] - Lvar[self.all_lb]
            zLx_guess = zL[:nl]

            drL = 1.5 + max(0.0, -1.5*np.min(rL_guess))
            dzLx = 1.5 + max(0.0, -1.5*np.min(zLx_guess))

            rL_shift = drL + 0.5*np.dot(rL_guess + drL, zLx_guess + dzLx) / \
                ((zLx_guess + dzLx).sum())
            zLx_shift = dzLx + 0.5*np.dot(rL_guess + drL, zLx_guess + dzLx) / \
                ((rL_guess + drL).sum())

            rL = rL_guess + rL_shift
            zL[:nl] = zLx_guess + zLx_shift
            x[self.all_lb] = Lvar[self.all_lb] + rL

        if nu > 0:
            rU_guess = Uvar[self.all_ub] - x[self.all_ub]
            zUx_guess = zU[:nu]

            drU = 1.5 + max(0.0, -1.5*np.min(rU_guess))
            dzUx = 1.5 + max(0.0, -1.5*np.min(zUx_guess))

            rU_shift = drU + 0.5*np.dot(rU_guess + drU, zUx_guess + dzUx) / \
                ((zUx_guess + dzUx).sum())
            zUx_shift = dzU + 0.5*np.dot(rU_guess + drU, zUx_guess + dzUx) / \
                ((rU_guess + drU).sum())

            rU = rU_guess + rU_shift
            zU[:nu] = zUx_guess + zUx_shift
            x[self.all_ub] = Uvar[self.all_ub] - rU

        # An additional normalization step for the range-bounded variables
        #
        # This normalization prevents the shift computed in rL and rU from
        # taking us outside the feasible range, and yields the same final
        # x value whether we take (Lvar + rL*norm) or (Uvar - rU*norm) as x
        if nl > 0 and nu > 0:
            intervals = Uvar[self.qp.rangeB] - Lvar[self.qp.rangeB]
            norm_factors = intervals / (intervals + rL_shift + rU_shift)
            x[self.qp.rangeB] = Lvar[self.qp.rangeB] + rL[self.range_in_lb]*norm_factors

        # Now shift the variables for the L1 regularization term
        sL_guess = x[n:] + x[:on]
        sU_guess = x[n:] - x[:on]
        zLv_guess = zL[nl:]
        zUv_guess = zU[nu:]

        dsL = 1.5 + max(0.0, -1.5*np.min(sL_guess))
        dsU = 1.5 + max(0.0, -1.5*np.min(sU_guess))
        dzLv = 1.5 + max(0.0, -1.5*np.min(zLv_guess))
        dzUv = 1.5 + max(0.0, -1.5*np.min(zUv_guess))

        sL_shift = dsL + 0.5*np.dot(sL_guess + dsL, zLv_guess + dzLv) / \
            ((zLv_guess + dzLv).sum())
        zLv_shift = dzLv + 0.5*np.dot(sL_guess + dsL, zLv_guess + dzLv) / \
            ((sL_guess + dsL).sum())

        sU_shift = dsU + 0.5*np.dot(sU_guess + dsU, zUv_guess + dzUv) / \
            ((zUv_guess + dzUv).sum())
        zUv_shift = dzUv + 0.5*np.dot(sU_guess + dsU, zUv_guess + dzUv) / \
            ((sU_guess + dsU).sum())

        sL = sL_guess + sL_shift
        sU = sU_guess + sU_shift
        zL[nl:] = zLv_guess + zLv_shift
        zU[nu:] = zUv_guess + zUv_shift
        x[n:] = np.maximum(sL - x[:on], sU + x[:on])

        # Initialization complete
        self.initial_guess = False

        # Check strict feasibility
        if not np.all(x[:n] > Lvar) or not np.all(x[:n] < Uvar) or \
        not np.all(zL > 0) or not np.all(zU > 0) or \
        not np.all(x[n:] > np.abs(x[:on])):
            raise ValueError('Initial point not strictly feasible')

        return (x, r, y, zL, zU)

    def max_primal_step_length(self, dx):
        """Compute the maximum step to the boundary in the primal variables.

        The function also returns the component index that produces the
        minimum steplength. (If the minimum steplength is 1, this value is
        set to -1.)

        For the L1 regularized problem, we need to account for the case
        where v becomes too small too quickly.
        """
        self.log.debug('Computing primal step length')
        xl = self.x[self.all_lb]
        xu = self.x[self.all_ub]
        v = self.x[self.n:]
        dxl = dx[self.all_lb]
        dxu = dx[self.all_ub]
        dv = dx[self.n:]
        l = self.qp.Lvar[self.all_lb]
        u = self.qp.Uvar[self.all_ub]
        on = self.original_n
        eps = 1.e-20

        if self.nl == 0:
            alphaL_max = 1.0
        else:
            # If dxl == 0., shift it slightly to prevent division by zero
            dxl_mod = np.where(dxl == 0., eps, dxl)
            alphaL = np.where(dxl < 0, -(xl - l)/dxl_mod, 1.)
            alphaL_max = min(1.0, alphaL.min())

        if self.nu == 0:
            alphaU_max = 1.0
        else:
            # If dxu == 0., shift it slightly to prevent division by zero
            dxu_mod = np.where(dxu == 0., -eps, dxu)
            alphaU = np.where(dxu > 0, (u - xu)/dxu_mod, 1.)
            alphaU_max = min(1.0, alphaU.min())

        # Additional work to account for added variables
        dv_mod = np.where(dv == 0., eps, dv)
        temp_a_max = min(alphaL_max, alphaU_max)
        temp_x = self.x[:on] + temp_a_max*dx[:on]
        temp_v = v + temp_a_max*dv_mod

        alpha_vL = np.where(temp_v + temp_x < 0., -(v + self.x[:on])/(dv_mod + dx[:on]), 1.)
        alpha_vU = np.where(temp_v - temp_x < 0., -(v - self.x[:on])/(dv_mod - dx[:on]), 1.)

        alpha_vL_max = min(1.0, alpha_vL.min())
        alpha_vU_max = min(1.0, alpha_vU.min())

        min_alpha_limit = min(alphaL_max,alphaU_max,alpha_vL_max,alpha_vU_max)

        if min_alpha_limit == 1.0:
            return (1.0, -1, False)

        if min_alpha_limit == alphaL_max:
            alpha_max = alphaL_max
            ind_max = self.all_lb[np.argmin(alphaL)]
            is_upper = False
        elif min_alpha_limit == alphaU_max:
            alpha_max = alphaU_max
            ind_max = self.all_ub[np.argmin(alphaU)]
            is_upper = True
        elif min_alpha_limit == alpha_vL_max:
            alpha_max = alpha_vL_max
            ind_max = self.n + np.argmin(alpha_vL)
            is_upper = False
        else:
            alpha_max = alpha_vU_max
            ind_max = self.n + np.argmin(alpha_vU)
            is_upper = True

        return (alpha_max, ind_max, is_upper)

    def max_dual_step_length(self, dzL, dzU):
        """Compute the maximum step to the boundary in the dual variables.

        This function is similar to the base class except for the case where
        the step length is restricted by the added variables.
        """
        self.log.debug('Computing dual step length')
        eps = 1.e-20

        # No if-blocks on bounds since we always have at least one regularized
        # variable

        # If dzL == 0., shift it slightly to prevent division by zero
        dzL_mod = np.where(dzL == 0., eps, dzL)
        alphaL = np.where(dzL < 0, -self.zL/dzL_mod, 1.)
        alphaL_max = min(1.0,alphaL.min())

        # If dzU == 0., shift it slightly to prevent division by zero
        dzU_mod = np.where(dzU == 0., -eps, dzU)
        alphaU = np.where(dzU < 0, -self.zU/dzU_mod, 1.)
        alphaU_max = min(1.0,alphaU.min())

        if min(alphaL_max,alphaU_max) == 1.0:
            return (1.0, -1, False)

        if alphaL_max < alphaU_max:
            alpha_max = alphaL_max
            arg = np.argmin(alphaL)
            if arg >= self.nl:
                ind_max = self.n + arg - self.nl
            else:
                ind_max = self.all_lb[arg]
            is_upper = False
        else:
            alpha_max = alphaU_max
            arg = np.argmin(alphaU)
            if arg >= self.nu:
                ind_max = self.n + arg - self.nu
            else:
                ind_max = self.all_ub[arg]
            is_upper = True

        return (alpha_max, ind_max, is_upper)

    def _compute_max_steps(self, dx, dzL, dzU):
        """Compute the maximum step lengths given the directions."""

        x = self.x
        zL = self.zL
        zU = self.zU
        Uvar = self.qp.Uvar
        Lvar = self.qp.Lvar

        # Compute largest allowed primal and dual stepsizes.
        (alpha_p, index_p, is_up_p) = self.max_primal_step_length(dx)
        (alpha_d, index_d, is_up_d) = self.max_dual_step_length(dzL, dzU)

        # Define fraction-to-the-boundary factor and compute the true
        # step sizes
        tau = max(.995, 1.0 - self.mu)

        if self.mehrotra_pc:
            # Compute actual stepsize using Mehrotra's heuristic.

            if index_p == index_d and is_up_p == is_up_d:
                # If both are -1, do nothing, since the step remains
                # strictly feasible and alpha_p = alpha_d = 1; otherwise,
                # there is a division by zero in Mehrotra's heuristic, so
                # we fall back on the standard fraction-to-boundary rule.
                if index_p != -1:
                    alpha_p *= tau
                    alpha_d *= tau
            else:
                mult = 0.01

                (mu_temp, _, _) = self._check_complementarity(x + alpha_p*dx,
                    zL + alpha_d*dzL, zU + alpha_d*dzU)

                # If alpha_p < 1.0, compute a gamma_p such that the
                # complementarity of the updated (x,z) pair is mult*mu_temp
                if index_p != -1:
                    if is_up_p:
                        if index_p < self.n:
                            ref_index = self.all_ub.index(index_p)
                            gamma_p = mult * mu_temp
                            gamma_p /= (zU[ref_index] + alpha_d*dzU[ref_index])
                            gamma_p -= (Uvar[index_p] - x[index_p])
                            gamma_p /= -(alpha_p*dx[index_p])
                        else:
                            ref_index = self.nu + (index_p - self.n)
                            gamma_p = mult * mu_temp
                            gamma_p /= (zU[ref_index] + alpha_d*dzU[ref_index])
                            gamma_p -= (x[index_p] - x[index_p - self.n])
                            gamma_p /= alpha_p*(dx[index_p] - dx[index_p - self.n])
                    else:
                        if index_p < self.n:
                            ref_index = self.all_lb.index(index_p)
                            gamma_p = mult * mu_temp
                            gamma_p /= (zL[ref_index] + alpha_d*dzL[ref_index])
                            gamma_p -= (x[index_p] - Lvar[index_p])
                            gamma_p /= (alpha_p*dx[index_p])
                        else:
                            ref_index = self.nl + (index_p - self.n)
                            gamma_p = mult * mu_temp
                            gamma_p /= (zL[ref_index] + alpha_d*dzL[ref_index])
                            gamma_p -= (x[index_p] + x[index_p - self.n])
                            gamma_p /= alpha_p*(dx[index_p] + dx[index_p - self.n])

                    # If mu_temp is very small, gamma_p = 1. is possible due to
                    # a cancellation error in the gamma_p calculation above.
                    # Therefore, set a maximum value of alpha_p < 1 to prevent
                    # division-by-zero errors later in the program.
                    alpha_p *= min(max(1 - mult, gamma_p), 1. - 1.e-8)

                # If alpha_d < 1.0, compute a gamma_d such that the
                # complementarity of the updated (x,z) pair is mult*mu_temp
                if index_d != -1:
                    if is_up_d:
                        if index_d < self.n:
                            ref_index = self.all_ub.index(index_d)
                            gamma_d = mult * mu_temp
                            gamma_d /= (Uvar[index_d] - x[index_d] - alpha_p*dx[index_d])
                            gamma_d -= zU[ref_index]
                            gamma_d /= (alpha_d*dzU[ref_index])
                        else:
                            ref_index = self.nu + (index_d - self.n)
                            gamma_d = mult * mu_temp
                            gamma_d /= (x[index_d] - x[index_d - self.n] + alpha_p*(dx[index_d] - dx[index_d - self.n]))
                            gamma_d -= zU[ref_index]
                            gamma_d /= (alpha_d*dzU[ref_index])
                    else:
                        if index_d < self.n:
                            ref_index = self.all_lb.index(index_d)
                            gamma_d = mult * mu_temp
                            gamma_d /= (x[index_d] + alpha_p*dx[index_d] - Lvar[index_d])
                            gamma_d -= zL[ref_index]
                            gamma_d /= (alpha_d*dzL[ref_index])
                        else:
                            ref_index = self.nl + (index_d - self.n)
                            gamma_d = mult * mu_temp
                            gamma_d /= (x[index_d] + x[index_d - self.n] + alpha_p*(dx[index_d] + dx[index_d - self.n]))
                            gamma_d -= zL[ref_index]
                            gamma_d /= (alpha_d*dzL[ref_index])

                    # If mu_temp is very small, gamma_d = 1. is possible due to
                    # a cancellation error in the gamma_d calculation above.
                    # Therefore, set a maximum value of alpha_d < 1 to prevent
                    # division-by-zero errors later in the program.
                    alpha_d *= min(max(1 - mult, gamma_d), 1. - 1.e-8)

        else:
            # Use the standard fraction-to-the-boundary rule
            alpha_p *= tau
            alpha_d *= tau

        return (alpha_p, alpha_d)

    def initialize_system(self):
        """Initialize the system matrix and right-hand side.

        The A and H blocks of the matrix are also put in place since they
        are common to all problems. (The C block is also included for least-
        squares problems.)
        """
        n = self.n
        on = self.original_n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu

        self.sys_size = n + on + p + m + nl + on + nu + on
        size_hint = nl + nu + 4*on + self.A.nnz + self.H.nnz + self.C.nnz
        size_hint += self.sys_size

        self.K = PysparseMatrix(size=self.sys_size, sizeHint=size_hint,
            symmetric=True)
        self.K[:n, :n] = self.H
        self.K[n+on:n+on+p, :n] = self.C
        self.K[n+on+p:n+on+p+m, :n] = self.A

        self.K.put(-1.0, range(n+on, n+on+p))

        self.rhs = np.zeros(self.sys_size)
        return

    def set_system_matrix(self):
        """Set up the linear system matrix."""
        n = self.n
        on = self.original_n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu

        # Convenience index
        z_start = n+on+p+m

        if self.initial_guess:
            self.log.debug('Setting up matrix for initial guess')

            self.K.put(self.diagH + self.primal_reg_min**0.5, range(n))
            self.K.put(self.primal_reg_min**0.5, range(n,n+on))
            self.K.put(-self.dual_reg_min**0.5, range(n+on+p,z_start))

            self.K.put(-1.0, range(z_start, self.sys_size))
            self.K.put(1.0, range(z_start, z_start+nl), self.all_lb)
            self.K.put(1.0, range(z_start+nl, z_start+nl+on), range(on))
            self.K.put(1.0, range(z_start+nl, z_start+nl+on), range(n,n+on))

            self.K.put(1.0, range(z_start+nl+on, z_start+nl+on+nu), self.all_ub)
            self.K.put(1.0, range(z_start+nl+on+nu, self.sys_size), range(on))
            self.K.put(-1.0, range(z_start+nl+on+nu, self.sys_size), range(n,n+on))

        else:
            self.log.debug('Setting up matrix for current iteration')
            Lvar = self.qp.Lvar
            Uvar = self.qp.Uvar
            x = self.x
            zL = self.zL
            zU = self.zU

            # Main diagonal terms
            self.K.put(self.diagH + self.primal_reg, range(n))
            self.K.put(self.primal_reg, range(n,n+on))
            self.K.put(-self.dual_reg, range(n+on+p,z_start))

            self.K.put(Lvar[self.all_lb] - x[self.all_lb], range(z_start, z_start+nl))
            self.K.put(-x[:on] - x[n:], range(z_start+nl, z_start+nl+on))
            self.K.put(x[self.all_ub] - Uvar[self.all_ub], range(z_start+nl+on, z_start+nl+on+nu))
            self.K.put(x[:on] - x[n:], range(z_start+nl+on+nu, self.sys_size))

            # Bound multiplier blocks
            self.K.put(zL[:nl]**0.5, range(z_start, z_start+nl), self.all_lb)
            self.K.put(zL[nl:]**0.5, range(z_start+nl, z_start+nl+on), range(on))
            self.K.put(zL[nl:]**0.5, range(z_start+nl, z_start+nl+on), range(n,n+on))

            self.K.put(zU[:nu]**0.5, range(z_start+nl+on, z_start+nl+on+nu), self.all_ub)
            self.K.put(zU[nu:]**0.5, range(z_start+nl+on+nu, self.sys_size), range(on))
            self.K.put(-zU[nu:]**0.5, range(z_start+nl+on+nu, self.sys_size), range(n,n+on))

        return

    def update_system_matrix(self):
        """Update the linear system matrix with the new regularization
        parameters. This is a helper method when checking the system for
        degeneracy."""
        n = self.n
        on = self.original_n
        m = self.m
        p = self.p
        self.log.debug('Updating matrix')
        self.K.put(self.diagH + self.primal_reg, range(n))
        self.K.put(self.primal_reg, range(n,n+on))
        self.K.put(-self.dual_reg, range(n+on+p,n+on+p+m))
        return

    def set_system_rhs(self, **kwargs):
        """Set up the linear system right-hand side."""
        n = self.n
        on = self.original_n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu
        self.log.debug('Setting up linear system right-hand side')

        # Convenience index
        z_start = n+on+p+m

        if self.initial_guess:
            self.rhs[z_start:z_start+nl] = self.qp.Lvar[self.all_lb]
            self.rhs[z_start+nl+on:z_start+nl+on+nu] = self.qp.Uvar[self.all_ub]
            if not kwargs.get('dual',False):
                # Primal initial point RHS
                self.rhs[:n] = 0.
                self.rhs[n:n+on] = 0.
                self.rhs[n+on:n+on+p] = self.d
                self.rhs[n+on+p:z_start] = self.b
            else:
                # Dual initial point RHS
                self.rhs[:n] = -self.c
                self.rhs[n:n+on] = -self.lam
                self.rhs[n+on:n+on+p] = self.d
                self.rhs[n+on+p:z_start] = 0.
        else:
            sigma = kwargs.get('sigma',0.0)
            self.rhs[:n+on] = -self.dFeas
            self.rhs[n+on:n+on+p] = -self.lsqRes
            self.rhs[n+on+p:z_start] = -self.pFeas
            self.rhs[z_start:z_start+nl+on] = -self.lComp + sigma*self.mu
            self.rhs[z_start:z_start+nl+on] *= self.zL**-0.5
            self.rhs[z_start+nl+on:] = self.uComp - sigma*self.mu
            self.rhs[z_start+nl+on:] *= self.zU**-0.5

        return

    def extract_xyz(self, **kwargs):
        """Return the partitioned solution vector"""
        n = self.n
        on = self.original_n
        m = self.m
        p = self.p
        nl = self.nl
        z_start = n+on+p+m

        x = self.lin_solver.x[:n+on].copy()
        r = -self.lin_solver.x[n+on:n+on+p].copy()
        y = -self.lin_solver.x[n+on+p:n+on+p+m].copy()
        if self.initial_guess:
            zL = -self.lin_solver.x[z_start:z_start+nl+on].copy()
            zU = self.lin_solver.x[z_start+nl+on:].copy()
        else:
            zL = -(self.zL**0.5)*self.lin_solver.x[z_start:z_start+nl+on].copy()
            zU = (self.zU**0.5)*self.lin_solver.x[z_start+nl+on:].copy()

        return x,r,y,zL,zU

    def check_optimality(self):
        u"""Compute feasibility and complementarity for the current point.

        For the L1 regularized problem, dual feasibility includes the special
        bound multiplier constraints

            λe - zᴸ - zᵁ = 0

        where zᴸ and zᵁ are multipliers associated with x ≥ -v and x ≤ v
        respectively.
        """
        x = self.x
        r = self.r
        y = self.y
        zL = self.zL
        zU = self.zU
        Lvar = self.qp.Lvar
        Uvar = self.qp.Uvar

        n = self.n
        on = self.original_n
        nl = self.nl
        nu = self.nu

        x_n = x[:n]
        v = x[n:]

        # Residual and complementarity vectors
        Hx = self.H*x_n
        self.qpObj = self.q + np.dot(self.c,x_n) + 0.5*np.dot(x_n,Hx)
        self.qpObj += 0.5*np.dot(r,r) + self.lam*v.sum()

        self.pFeas = self.A*x_n - self.b

        if self.use_lsq:
            self.lsqRes = self.C*x_n + r - self.d
        else:
            self.lsqRes = np.zeros(0, dtype=np.float)

        self.dFeas = np.zeros(n+on, dtype=np.float)
        self.dFeas[:n] = Hx + self.c - y*self.A - r*self.C
        self.dFeas[self.all_lb] -= zL[:nl]
        self.dFeas[self.all_ub] += zU[:nu]
        self.dFeas[:on] -= zL[nl:]
        self.dFeas[:on] += zU[nu:]
        self.dFeas[n:] = self.lam - zL[nl:] - zU[nu:]

        (self.mu, self.lComp, self.uComp) = self._check_complementarity(x, zL, zU)

        pFeasNorm = norm2(self.pFeas)
        dFeasNorm = norm2(self.dFeas)
        lsqNorm = norm2(self.lsqRes)

        # Scaled residual norms and duality gap
        norm_sum = self.normA + self.normH + self.normC
        self.pResid = pFeasNorm / (1 + self.normb + norm_sum)
        self.dResid = dFeasNorm / (1 + self.normc + norm_sum)
        self.lsqResid = lsqNorm / (1 + self.normd + norm_sum)
        self.dual_gap = self.mu / (1 + abs(np.dot(self.c,x_n)) + norm_sum)

        # Overall residual for stopping condition
        return max(self.pResid, self.lsqResid, self.dResid, self.dual_gap)

    def _check_complementarity(self, x, zL, zU):
        """Compute the complementarity given x, zL, and zU."""
        nl = self.nl
        nu = self.nu
        on = self.original_n
        n = self.n

        lComp = np.zeros(nl+on, dtype=np.float)
        lComp[:nl] = zL[:nl]*(x[self.all_lb] - self.qp.Lvar[self.all_lb])
        lComp[nl:] = zL[nl:]*(x[n:] + x[:on])

        uComp = np.zeros(nu+on, dtype=np.float)
        uComp[:nu] = zU[:nu]*(self.qp.Uvar[self.all_ub] - x[self.all_ub])
        uComp[nu:] = zU[nu:]*(x[n:] - x[:on])

        # No if block because we always have at least one v variable
        mu = (lComp.sum() + uComp.sum()) / (nl + nu + 2*on)

        return (mu, lComp, uComp)


class RegL1QPInteriorPointSolver2x2(RegL1QPInteriorPointSolver):
    """A 2x2 block variant of the L1 regularized interior-point method.

    Linear system is based on the (reduced) 2x2 block system instead of
    the 3x3 block system.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the problem, similar to the base solver."""
        super(RegL1QPInteriorPointSolver2x2,self).__init__(*args, **kwargs)

    def initialize_system(self):
        """Initialize the system matrix and right-hand side.

        The A and H blocks of the matrix are also put in place since they
        are common to all problems. (The C block is also included for least-
        squares problems.)
        """
        n = self.n
        on = self.original_n
        m = self.m
        p = self.p

        self.sys_size = n + on + p + m
        size_hint = self.A.nnz + self.H.nnz + self.C.nnz + on + self.sys_size

        self.K = PysparseMatrix(size=self.sys_size, sizeHint=size_hint,
            symmetric=True)
        self.K[:n, :n] = self.H
        self.K[n+on:n+on+p, :n] = self.C
        self.K[n+on+p:n+on+p+m, :n] = self.A

        self.K.put(-1.0, range(n+on, n+on+p))

        self.rhs = np.zeros(self.sys_size)
        return

    def set_system_matrix(self):
        """Set up the linear system matrix."""
        n = self.n
        on = self.original_n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu

        if self.initial_guess:
            self.log.debug('Setting up matrix for initial guess')

            new_diag = self.diagH + self.primal_reg_min**0.5
            new_diag[self.all_lb] += 1.0
            new_diag[self.all_ub] += 1.0
            new_diag[:on] += 2.0

            new_diag_2 = self.primal_reg_min**0.5 * np.ones(on, dtype=np.float)
            new_diag_2 += 2.0

            # Use an epsilon-value in these positions to identify them as stored
            # in the sparse matrix. Otherwise, the symbolic factorization will
            # be incorrect in the main loop
            off_diag = 1.e-20 * np.ones(on, dtype=np.float)

            self.K.put(new_diag, range(n))
            self.K.put(new_diag_2, range(n,n+on))
            self.K.put(off_diag, range(n,n+on), range(on))
            self.K.put(-self.dual_reg_min**0.5, range(n+on+p,n+on+p+m))

        else:
            self.log.debug('Setting up matrix for current iteration')

            x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
            u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]
            v_plus_x = self.x[n:] + self.x[:on]
            v_minus_x = self.x[n:] - self.x[:on]

            new_diag = self.diagH + self.primal_reg
            new_diag[self.all_lb] += self.zL[:nl] / x_minus_l
            new_diag[self.all_ub] += self.zU[:nu] / u_minus_x
            new_diag[:on] += self.zL[nl:] / v_plus_x
            new_diag[:on] += self.zU[nu:] / v_minus_x

            new_diag_2 = self.primal_reg * np.ones(on, dtype=np.float)
            new_diag_2 += self.zL[nl:] / v_plus_x
            new_diag_2 += self.zU[nu:] / v_minus_x

            off_diag = (self.zL[nl:] / v_plus_x) - (self.zU[nu:] / v_minus_x)

            self.K.put(new_diag, range(n))
            self.K.put(new_diag_2, range(n,n+on))
            self.K.put(off_diag, range(n,n+on), range(on))
            self.K.put(-self.dual_reg, range(n+on+p,n+on+p+m))

        return

    def update_system_matrix(self):
        """Update the linear system matrix with the new regularization
        parameters. This is a helper method when checking the system for
        degeneracy."""

        self.log.debug('Updating matrix')
        n = self.n
        on = self.original_n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu

        x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
        u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]
        v_plus_x = self.x[n:] + self.x[:on]
        v_minus_x = self.x[n:] - self.x[:on]

        new_diag = self.diagH + self.primal_reg
        new_diag[self.all_lb] += self.zL[:nl] / x_minus_l
        new_diag[self.all_ub] += self.zU[:nu] / u_minus_x
        new_diag[:on] += self.zL[nl:] / v_plus_x
        new_diag[:on] += self.zU[nu:] / v_minus_x

        new_diag_2 = self.primal_reg * np.ones(on, dtype=np.float)
        new_diag_2 += self.zL[nl:] / v_plus_x
        new_diag_2 += self.zU[nu:] / v_minus_x

        self.K.put(new_diag, range(n))
        self.K.put(new_diag_2, range(n,n+on))
        self.K.put(-self.dual_reg, range(n+on+p,n+on+p+m))

        return

    def set_system_rhs(self, **kwargs):
        """Set up the linear system right-hand side."""
        n = self.n
        on = self.original_n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu
        self.log.debug('Setting up linear system right-hand side')

        if self.initial_guess:
            self.rhs[:n] = 0.
            self.rhs[self.all_lb] += self.qp.Lvar[self.all_lb]
            self.rhs[self.all_ub] += self.qp.Uvar[self.all_ub]
            if not kwargs.get('dual',False):
                # Primal initial point RHS
                self.rhs[n:n+on] = 0.
                self.rhs[n+on:n+on+p] = self.d
                self.rhs[n+on+p:n+on+p+m] = self.b
            else:
                # Dual initial point RHS
                self.rhs[:n] -= self.c
                self.rhs[n:n+on] = -self.lam
                self.rhs[n+on:n+on+p] = self.d
                self.rhs[n+on+p:n+on+p+m] = 0.

        else:
            sigma = kwargs.get('sigma',0.0)
            x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
            u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]
            v_plus_x = self.x[n:] + self.x[:on]
            v_minus_x = self.x[n:] - self.x[:on]

            self.rhs[:n+on] = -self.dFeas
            self.rhs[n+on:n+on+p] = -self.lsqRes
            self.rhs[n+on+p:n+on+p+m] = -self.pFeas

            self.rhs[self.all_lb] += -self.zL[:nl] + sigma*self.mu/x_minus_l
            self.rhs[self.all_ub] += self.zU[:nu] - sigma*self.mu/u_minus_x
            self.rhs[:on] += -self.zL[nl:] + sigma*self.mu/v_plus_x
            self.rhs[:on] += self.zU[nu:] - sigma*self.mu/v_minus_x

            self.rhs[n:n+on] += -self.zL[nl:] + sigma*self.mu/v_plus_x
            self.rhs[n:n+on] += -self.zU[nu:] + sigma*self.mu/v_minus_x

        return

    def extract_xyz(self, **kwargs):
        """Return the partitioned solution vector"""
        n = self.n
        on = self.original_n
        m = self.m
        p = self.p
        nl = self.nl
        nu = self.nu

        x = self.lin_solver.x[:n+on].copy()
        r = -self.lin_solver.x[n+on:n+on+p].copy()
        y = -self.lin_solver.x[n+on+p:n+on+p+m].copy()
        zL = np.zeros(nl+on, dtype=np.float)
        zU = np.zeros(nu+on, dtype=np.float)

        if self.initial_guess:
            zL[:nl] = self.qp.Lvar[self.all_lb] - x[self.all_lb]
            zL[nl:] = -(x[n:] + x[:on])
            zU[:nu] = x[self.all_ub] - self.qp.Uvar[self.all_ub]
            zU[nu:] = -(x[n:] - x[:on])

        else:
            sigma = kwargs.get('sigma',0.0)
            x_minus_l = self.x[self.all_lb] - self.qp.Lvar[self.all_lb]
            u_minus_x = self.qp.Uvar[self.all_ub] - self.x[self.all_ub]
            v_plus_x = self.x[n:] + self.x[:on]
            v_minus_x = self.x[n:] - self.x[:on]

            zL[:nl] = (-self.zL[:nl]*(x_minus_l + x[self.all_lb]) + sigma*self.mu)
            zL[:nl] /= x_minus_l
            zL[nl:] = (-self.zL[nl:]*(v_plus_x + x[:on] + x[n:]) + sigma*self.mu)
            zL[nl:] /= v_plus_x

            zU[:nu] = (-self.zU[:nu]*(u_minus_x - x[self.all_ub]) + sigma*self.mu)
            zU[:nu] /= u_minus_x
            zU[nu:] = (-self.zU[nu:]*(v_minus_x - x[:on] + x[n:]) + sigma*self.mu)
            zU[nu:] /= v_minus_x

        return x,r,y,zL,zU
