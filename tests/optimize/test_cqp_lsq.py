"""
test_cqp_lsq.py

A short test script to test functionality of the new CQP code on a
least-squares problem.
The test problem used here problem 1 from the TEST_LLS collection.
The problem has no constraints.
"""

# from nlp.model.nlpmodel import SparseNLPModel
from nlp.model.qpmodel import LSQModel
from pysparse.sparse import PysparseMatrix
from nlp.optimize.cqp import RegQPInteriorPointSolver, RegQPInteriorPointSolver2x2
from nlp.optimize.cqp import RegL1QPInteriorPointSolver, RegL1QPInteriorPointSolver2x2
from nlp.tools.logs import config_logger
import numpy as np
import logging

# Create the operators for the problem
n = 3
p = 5

c = np.zeros(n, np.float)
H = PysparseMatrix(nrow=n, ncol=n, sizeHint=0, symmetric=True)

d = np.array([1.0, 2.3, 4.6, 3.1, 1.2])
C = PysparseMatrix(nrow=p, ncol=n, sizeHint=15, symmetric=False)
for i in range(p):
	C[i,0] = 1.0
	for j in range(1,n):
		C[i,j] = C[i,j-1]*(i+1)

# Configure the logger for CQP
cqp_logger = config_logger("nlp.cqp","%(name)-8s %(levelname)-5s %(message)s")

prob = LSQModel(name='LLS_1',fromOps=(c,H,None),lsqOps=(d,C),
	nnzc=n*p)

solver = RegQPInteriorPointSolver(prob)
# solver = RegQPInteriorPointSolver2x2(prob)
# solver = RegL1QPInteriorPointSolver(prob, lam=1.0)
# solver = RegL1QPInteriorPointSolver2x2(prob, lam=1.0)
solver.solve()

print solver.status
print solver.tsolve

print "Solution comparison:"
print "CQP solution:"
print solver.x
print "Provided solution:"
print np.array([-3.0200000, 4.4914286, -0.72857143])