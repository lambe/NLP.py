#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nlp_cqp_lsq.py

Solve an NLP from the CUTEst collection using the NLP.py CQP solver.

CQP assumes the problem is convex quadratic. If a general NLP is selected,
CQP will try to minimize the quadratic approximation at the initial point.
"""

from cutest.model.cutestmodel import CUTEstModel
from nlp.model.qpmodel import QPModel, LSQModel
from nlp.model.nofixedvars import NoFixedVarsModel
from nlp.optimize.cqp import RegQPInteriorPointSolver2x2QR
from nlp.optimize.cqp import RegQPInteriorPointSolverQR
from nlp.tools.logs import config_logger
import numpy as np
import logging
import sys
import argparse

# Set up the problem loggers
def cqp_stats(cqp):
    """Obtain CQP statistics and indicate failures with negatives."""
    print cqp.status
    if cqp.status == "opt":
        it = cqp.iter
        fc, gc = cqp.qp.obj.ncalls, cqp.qp.grad.ncalls
        pg = cqp.kktRes
        ts = cqp.tsolve
    else:
        it = -cqp.iter
        fc, gc = -cqp.qp.obj.ncalls, -cqp.qp.grad.ncalls
        pg = -1.0 if cqp.kktRes is None else -cqp.kktRes
        ts = -1.0 if cqp.tsolve is None else -cqp.tsolve
    return (it, fc, gc, pg, ts)

parser = argparse.ArgumentParser()
parser.add_argument("name_list", nargs='+', help="list of SIF files to process")

parser.add_argument("--use_pc", dest='use_pc', action='store_true', 
    help="Use Mehrotra's predictor-corrector strategy (default).")
parser.add_argument("--use_long_step", dest='use_pc', action='store_false', 
    help="Use a long-step line search strategy.")
parser.set_defaults(use_pc=True)

parser.add_argument("--use_scale", type=str, default="none", choices=["none","abs","mc29"],
    help="Choose no scaling (default), scaling based on the Jacobian values (abs), or use MC29 to compute factors (mc29)")

parser.add_argument("--use_K2", dest='use_K2', action='store_true',
    help="Use the 2x2 block linear system instead of the (default) 3x3 system")
parser.add_argument("--use_K3", dest='use_K2', action='store_false',
    help="Use the 3x3 block linear system (default)")
parser.set_defaults(use_K2=False)

parser.add_argument("--primal_solve", dest='primal_solve', action='store_true',
    help="In the least-squares solver, solve for the primal variables directly")
parser.add_argument("--dual_solve", dest='primal_solve', action='store_false',
    help="In the least-squares solver, solve for the dual variables directly (default)")
parser.set_defaults(primal_solve=False)

parser.add_argument("--extra_scale", dest='extra_scale', action='store_true',
    help="In the least-squares solver, solve for a scaled set of variables")
parser.add_argument("--no_extra_scale", dest='extra_scale', action='store_false',
    help="In the least-squares solver, solve for the original variables (default)")
parser.set_defaults(extra_scale=False)

args = parser.parse_args()

nprobs = len(args.name_list)

# Create root logger.
# logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s")
logger = config_logger("nlp", "%(name)-3s %(levelname)-5s %(message)s",
                                stream=None,
                                filename="cutest_qp_rough.txt", filemode="a")

# Create Cqp logger.
cqp_logger = config_logger("nlp.cqp",
                              "%(name)-8s %(levelname)-5s %(message)s",
                              level=logging.WARN if nprobs > 1 else logging.INFO)
# cqp_logger = config_logger("nlp.cqp",
#                               "%(name)-8s %(levelname)-5s %(message)s",
#                               stream=None,
#                               filename="test_prob_qr_lsq.txt", filemode="w",
#                               level=logging.WARN if nprobs > 1 else logging.INFO)

if nprobs > 1:
    logger.info("%9s %5s %5s %6s %8s %8s %6s %6s %5s %7s",
                "name", "nvar", "ncon", "iter", "f", u"‖F(w)‖", "#f", u"#∇f", "stat",
                "time")

# Solve problems
for name in args.name_list:
    if name[-4:] == ".SIF":
        name = name[:-4]

    prob = CUTEstModel(name)
    prob.compute_scaling_obj()
    prob.compute_scaling_cons()

    # Remove the fixed variables, if any
    if prob.nfixedB > 0:
        modprob = NoFixedVarsModel(prob)
    else:
        modprob = prob

    # Test either the standard QP or the least-squares version
    # qp = QPModel(fromProb=(modprob,None,None))
    qp = LSQModel(fromProb=(modprob,None,None))

    if args.use_K2:
        cqp = RegQPInteriorPointSolver2x2QR(qp, mehrotra_pc=args.use_pc, scale_type=args.use_scale,
            primal_reg_min=1.0e-8, dual_reg_min=1.0e-8, primal_solve=args.primal_solve,
            extra_scale=args.extra_scale)
    else:
        cqp = RegQPInteriorPointSolverQR(qp, mehrotra_pc=args.use_pc, scale_type=args.use_scale,
            primal_reg_min=1.0e-8, dual_reg_min=1.0e-8, primal_solve=args.primal_solve,
            extra_scale=args.extra_scale)

    # Solve the problem and print the result
    try:
        cqp.solve()
        status = cqp.status
        niter, fcalls, gcalls, kktRes, tsolve = cqp_stats(cqp)
    except:
        msg = sys.exc_info()[1].message
        status = msg if len(msg) > 0 else "xfail"  # unknown failure
        niter, fcalls, gcalls, kktRes, tsolve = cqp_stats(cqp)

    logger.info("%9s %5d %5d %6d %8.1e %8.1e %6d %6d %5s %7.3f",
            prob.name, prob.nvar, prob.ncon, niter, cqp.qpObj, kktRes,
            fcalls, gcalls, status, tsolve)
