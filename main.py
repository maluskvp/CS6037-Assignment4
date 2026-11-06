# Assignment 4 - PAC Learning and VC Dimension
# HMWK GROUP 4: Veronica Malusky and Laila Nawara

# This is the main python file for the programming section (Part 1) of Assignment 4


# imports for entire file
import itertools
import numpy as np
from scipy.optimize import linprog
from numpy.polynomial.legendre import legvander
from sklearn.svm import LinearSVC

# P1 The VC confidence term

# implement f_m(h) from (1) with learning rate = 0.05

# P2 A shattering checker 

# skeleton code provided 
def separable(X, y):
  """Return True if labels y in {-1,+1} are linearly separable on X."""
  m, n = X.shape
  # variables z = (w_1..w_n, b); constraint -y_i (w.x_i + b) <= -1
  A_ub = -y[:, None] * np.hstack([X, np.ones((m, 1))])
  b_ub = -np.ones(m)
  res = linprog(c=np.zeros(n + 1), A_ub=A_ub, b_ub=b_ub,
  bounds=[(None, None)] * (n + 1), method="highs")
  return res.status == 0 # 0 = feasible solution found

# can_shatter(X) decides whether oriented hyperplanes sign(w · x + b) shatter a point set
def can_shatter(X):
  # TODO: loop over all labelings y in itertools.product([-1, 1], repeat=len(X))
  return

# P3 PAC learning in action: rectangles and thresholds

# skeleton code provided for P4
def features(x, d):
  return legvander(x, d)[:, 1:] # columns P_1(x)..P_d(x); constant dropped


if __name__ == "__main__":
  # P1 - The VC confidence term

  # P2 - A shattering checker 

  # P3 - PAC learning in action: rectangles and thresholds

  # P4 - Structural Risk Minimization versus cross-validation 

  # Initialize training set needed for Part 4

  # skeleton code provided
  clf = LinearSVC(C=1e4, max_iter=200000).fit(features(x_tr, d), y_tr)
  p_test = clf.decision_function(features(x_te, d)) # this is the learned p(x)
  y_hat = np.where(p_test > 0, 1, -1) # sign(p(x))

