# Assignment 4 - PAC Learning and VC Dimension
# HMWK GROUP 4: Veronica Malusky and Laila Nawara

# This is the main python file for the programming section (Part 1) of Assignment 4


# imports for entire file
import itertools
import numpy as np
from scipy.optimize import linprog, brentq
from numpy.polynomial.legendre import legvander
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


np.random.seed(42) # seed set to 42

# P1 The VC confidence term

# F_m(h) function VC confidence
def fm(h,m):
  lr = 0.05
  return np.sqrt(( h * (np.log((2 * m) / h) + 1) - np.log(lr/4)) / m) # fm(h) (VC confidence)

# squared version of f_m function used for part b
def squared_fm(r,m):
  lr = 0.05
  return (( (r * (np.log((2 * m) / r) + 1)) - np.log(lr/4)) / m) - 1



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

  print("\nPart 1 results:")
  for m in m_val:

    print(f"\nm = {m} analysis:")
    h_values = np.linspace(1/m, 1, 200) # evenly space of 1/m to 1
    h_over_m = np.round(h_values * m) # round to integers
    h_over_m = np.maximum(h_over_m,1) # make sure h is a positive integer (1 or the value obtained at h_over_m rounded

    ratio = h_over_m / m # get the ratio of h/m
    fm_values = fm(h_over_m, m) # get f_m(h)

    print(f"Starting Value for ratio h/m = {ratio[0]} where m = {m}")
    print(f"Ending Value for ratio h/m = {ratio[-1]} where m = {m}")

    print(f"Starting Value for f_m(h) = {fm_values[0]} where m = {m}")
    print(f"Ending Value for f_m(h) = {fm_values[-1]} where m = {m}")

    # (b) find numerically the ration h/m at which f_m(h) = 1
    print("\nPart 1 (b)")
    root_finder = brentq(
      lambda r: squared_fm(r*m, m),
      10e-6,
      1
    )
    print(f"m = {m}, h/m when f_m(h) = {root_finder}")
   
    plt.plot(ratio, fm_values, label=f"m = {m}")

  plt.xlabel("h/m ratio")
  plt.ylabel("f_m(h)")
  plt.title("F_m(h) vs ratio h/m for m in {10^3, 10^4, 10^5}")
  plt.legend()
  plt.show()

  print("\nEnd of Part 1 (a) and Part 1 (b)")

  print("\nPart 1 (c)")
  # (c) largest integer VC dimension h where R(a) <= 0.1 and R_emp = 0 and R_emp = 0.05
  R_emp = [0, 0.05] # emp = 0 and 0.05
  for emp in R_emp:
    VC_int = []
    for h in range(1, 10**4 + 1): # h > 0 integers and limit is 10^4
      R = emp + fm(h,10**4) # calculate R based on the given R_emp and f_m(h) where m = 10^4
      if(R <= 0.1):
        VC_int.append(h)
    print(f"Largest integer VC dimension h where R_emp = {emp} is {max(VC_int)}")
  
  print("\nEnd of Part 1")

  # P2 - A shattering checker 
  print("\nPart 2 Results:")


  print("\nEnd of Part 2")

  # P3 - PAC learning in action: rectangles and thresholds
  print("\nPart 3 Results:")
  x1, x2 = 0.2, 0.7
  y1, y2 = 0.3, 0.6
  area_c = (x2 - x1) * (y2 - y1) # get the area of c*

  print(f"area(c*) = {area_c}")
  T = 2000 # number of independent trials
  m_val = [10,20,50,100,200,500,1000] # range of m
  e = 0.05

  mean_errors = []
  probs_R_g_e = []
  for m in m_val:
    total_errors = []
    for i in range(T):
      points = np.random.uniform(0,1,(m,2)) # get [mx2] uniform points between 0 and 1

      in_c = (
        (points[:,0] >= x1) & # min x value
        (points[:,0] <= x2) & # max x value
        (points[:,1] >= y1) & # min y value
        (points[:,1] <= y2)   # max y value
        ) # range for when point will be marked positive

      positive_p = points[in_c] # return all the points that would be positive

      if len(positive_p) == 0:
        l_area = 0
      else:
        l_area = ((np.max(positive_p[:,0]) - np.min(positive_p[:,0])) * (np.max(positive_p[:,1]) - np.min(positive_p[:,1]))) # get the learned area from the max - min of the x and y points
      
      total_errors.append(area_c - l_area) # add the error to the array to get mean after looping through T trials

    # get the mean and probabilities of when error > 0.05 (e)
    mean_error = np.mean(total_errors)
    prob = np.mean(np.array(total_errors) > e)

    mean_errors.append(mean_error)
    probs_R_g_e.append(prob)
  # plot (i) Mean True Error vs m
  plt.figure(figsize=(8,5))
  plt.loglog(m_val,mean_errors, label="Mean true error", marker="o")
  plt.xlabel("Sample size m")
  plt.ylabel("Mean True Error")
  plt.title("Mean True Error vs m samples")
  plt.legend()
  plt.show()

  # plot (ii) 

  plt.figure(figsize=(8,5))

  plt.plot(m_val,probs_R_g_e, label="P(R > 0.05)", marker="o")
  plt.axhline(
    0.05,
    linestyle=':',
    label=r"$\delta = 0.05$"
  ) # (c) add marker for when epsilon = delta = 0.05

  plt.xlabel("Sample size m")
  plt.ylabel("Fraction of trials where R > 0.05")
  plt.title("Probability R is greater than 0.05")
  plt.legend()
  plt.show()

  # (c) smallest m where P(R > 0.05) < 0.05
  new_m_vals = range(100,200, 10) # get new range between 100 and 200 to the nearest 10 to find where it first passes the threshold marker (0.05)
  smallest_ms = []
  for m in new_m_vals:
    # repeat (b)
    total_errors = []
    for i in range(T):
      points = np.random.uniform(0,1,(m,2)) # get [mx2] uniform points between 0 and 1

      in_c = (
        (points[:,0] >= x1) & # min x value
        (points[:,0] <= x2) & # max x value
        (points[:,1] >= y1) & # min y value
        (points[:,1] <= y2)   # max y value
        ) # range for when point will be marked positive

      positive_p = points[in_c] # return all the points that would be positive

      if len(positive_p) == 0:
        l_area = 0
      else:
        l_area = ((np.max(positive_p[:,0]) - np.min(positive_p[:,0])) * (np.max(positive_p[:,1]) - np.min(positive_p[:,1]))) # get the learned area from the max - min of the x and y points
      
      total_errors.append(area_c - l_area) # add the error to the array to get mean after looping through T trials

    # get the mean and probabilities of when error > 0.05 (e)
    mean_error = np.mean(total_errors)
    prob = np.mean(np.array(total_errors) > e)

    if prob <= 0.05:
      smallest_ms.append(m)

  print(f"Smallest m where empirical P(R > e) <= delta = {min(smallest_ms)}")

  print("\nEnd of Part 3")

  # P4 - Structural Risk Minimization versus cross-validation 
  print("\nPart 4 Results:")
  

  # Initialize training set needed for Part 4

  # skeleton code provided
  clf = LinearSVC(C=1e4, max_iter=200000).fit(features(x_tr, d), y_tr)
  p_test = clf.decision_function(features(x_te, d)) # this is the learned p(x)
  y_hat = np.where(p_test > 0, 1, -1) # sign(p(x))


  print("\nEnd of Part 4")

