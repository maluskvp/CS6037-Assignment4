# Assignment 4 - PAC Learning and VC Dimension
# HMWK GROUP 4: Veronica Malusky and Laila Nawara

# This is the main python file for the programming section (Part 1) of Assignment 4


# imports for entire file
import itertools
import numpy as np
from scipy.optimize import linprog
from numpy.polynomial.legendre import legvander
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


np.random.seed(42) # seed set to 42

# P1 The VC confidence term


def fm(h,m):
  lr = 0.05
  return np.sqrt(( h * (np.log((2 * m) / h) + 1) - np.log(lr/4)) / m) # fm(h) (VC confidence)
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

  m_val = [10**3, 10**4, 10**5] 

  # (a) plot fm(h) against ratio h/m
  plt.figure(figsize=(8,5))

  for m in m_val:

    h_values = np.linspace(1/m, 1, 200) # evenly space of 1/m to 1
    h_over_m = np.round(h_values * m) # round to integers
    h_over_m = np.maximum(h_over_m,1) # make sure h is a positive integer (1 or the value obtained at h_over_m rounded

    ratio = h_over_m / m # get the ratio of h/m
    fm_values = fm(h_over_m, m) # get f_m(h)
    
    plt.plot(ratio, fm_values, label=f"m = {m}")

  plt.xlabel("h/m ratio")
  plt.ylabel("f_m(h)")
  plt.title("F_m(h) vs ratio h/m for m in [10^3, 10^4, 10^5]}")
  plt.legend()
  plt.show()

  # P2 - A shattering checker 

  # P3 - PAC learning in action: rectangles and thresholds

  # P4 - Structural Risk Minimization versus cross-validation 

  # Initialize training set needed for Part 4

  # skeleton code provided
  # clf = LinearSVC(C=1e4, max_iter=200000).fit(features(x_tr, d), y_tr)
  # p_test = clf.decision_function(features(x_te, d)) # this is the learned p(x)
  # y_hat = np.where(p_test > 0, 1, -1) # sign(p(x))

