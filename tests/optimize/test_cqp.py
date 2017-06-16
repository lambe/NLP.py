"""
test_cqp.py

A short test script to test functionality of the new CQP code.
The test problem used here is Hock-Schittkowski 21.
"""

from nlp.model.nlpmodel import SparseNLPModel
from nlp.model.qpmodel import QPModel
from nlp.optimize.cqp import RegQPInteriorPointSolver, RegQPInteriorPointSolver2x2
from nlp.optimize.cqp import RegQPInteriorPointSolverQR, RegQPInteriorPointSolver2x2QR
from nlp.tools.logs import config_logger
import numpy as np
import logging


class HS21(SparseNLPModel):

    def __init__(self, **kwargs):
        super(HS21, self).__init__(2, m=1, name='test_class', Lcon=np.array([10]),
            Lvar=np.array([2,-50]), Uvar=np.array([50,50]), nnzj=2, nnzh=2,
            **kwargs)

    def obj(self, x):
        return 0.01*x[0]**2 + x[1]**2 - 100

    def grad(self, x):
        g = np.empty(self.n)
        g[0] = 0.02*x[0]
        g[1] = 2*x[1]
        return g

    def cons(self, x):
        c = np.empty(self.m)
        c[0] = 10*x[0] - x[1]
        return c

    def jac_triple(self, x):
        # Return triple of numpy arrays in coordinate format
        vals = np.array([10, -1])
        rows = np.array([0, 0])
        cols = np.array([0, 1])
        return vals, rows, cols

    def hess_triple(self, x, z=None):
        # Return triple of numpy arrays in coordinate format
        vals = np.array([0.02, 2])
        rows = np.array([0, 1])
        cols = np.array([0, 1])
        return vals, rows, cols


# Configure the logger for CQP
cqp_logger = config_logger("nlp.cqp","%(name)-8s %(levelname)-5s %(message)s")

# main script
test_prob = HS21()
test_prob_qp = QPModel(fromProb=(test_prob,None,None))
use_pc = True
use_scale = 'mc29'

solver = RegQPInteriorPointSolver(test_prob_qp, mehrotra_pc=use_pc,
    scale_type=use_scale)
solver.solve()
print solver.status
print solver.tsolve

solver2 = RegQPInteriorPointSolver2x2(test_prob_qp, mehrotra_pc=use_pc,
    scale_type=use_scale)
solver2.solve()
print solver2.status
print solver2.tsolve

solverQR = RegQPInteriorPointSolverQR(test_prob_qp, mehrotra_pc=use_pc,
    scale_type=use_scale, primal_solve=True)
solverQR.solve()
print solverQR.status
print solverQR.tsolve

solverQR2 = RegQPInteriorPointSolver2x2QR(test_prob_qp, mehrotra_pc=use_pc,
    scale_type=use_scale, primal_solve=True)
solverQR.solve()
print solverQR.status
print solverQR.tsolve
