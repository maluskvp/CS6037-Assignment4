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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


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
    # check every possible labeling
    for labels in itertools.product([-1, 1], repeat=len(X)):
        y = np.array(labels)

        # stop if one labeling is not separable
        if not separable(X, y):
            return False

    return True


# part 2 experiments

def rectangle_can_shatter(X):
    # check every labeling using rectangles
    for labels in itertools.product([-1, 1], repeat=len(X)):
        y = np.array(labels)
        positive = X[y == 1]
        negative = X[y == -1]

        # all negative points can always be separated
        if len(positive) == 0:
            continue

        # find the smallest rectangle around positive points
        lower = positive.min(axis=0)
        upper = positive.max(axis=0)

        # check if any negative point is inside
        inside = np.all(
            (negative >= lower) & (negative <= upper),
            axis=1
        )

        if np.any(inside):
            return False

    return True


def P2():
    rng = np.random.default_rng(42)

    # part 2a: check if three points can be shattered
    print("\npart 2a")

    noncollinear = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])

    collinear = np.array([
        [0, 0],
        [1, 0],
        [2, 0]
    ])

    print("noncollinear points:", can_shatter(noncollinear))
    print("collinear points:", can_shatter(collinear))

    # part 2b: test random points in different dimensions
    print("\npart 2b")

    results = []

    for n in [2, 3, 4, 5]:
        fractions = []

        for m in [n + 1, n + 2]:
            shattered = 0

            for i in range(1000):
                X = rng.standard_normal((m, n))  # generate m points with n features

                if can_shatter(X):
                    shattered += 1  # count sets that can be shattered

            fractions.append(shattered / 1000)  # fraction out of 1000 trials

        results.append([n, fractions[0], fractions[1]])

        print(
            f"n={n}: n+1={fractions[0]:.3f}, "
            f"n+2={fractions[1]:.3f}",
            flush=True
        )

    print("\nn | n+1 points | n+2 points")

    for row in results:
        print(f"{row[0]} | {row[1]:.3f} | {row[2]:.3f}")

    # part 2c: find labelings that a straight line cannot separate
    print("\npart 2c")

    X_four = rng.standard_normal((4, 2))
    impossible = []

    for labels in itertools.product([-1, 1], repeat=4):
        y = np.array(labels)

        if not separable(X_four, y):
            impossible.append(y)  # save the labeling if no separating line exists

    print("four points:")
    print(X_four)

    print("nonseparable labelings:")

    for y in impossible:
        print(y.tolist())

    # check how the four points are arranged
    from scipy.spatial import ConvexHull

    hull = ConvexHull(X_four)

    if len(hull.vertices) == 4:
        case = "convex quadrilateral"
    else:
        case = "one point inside a triangle"

    print("point arrangement:", case)
    print("number of nonseparable labelings:", len(impossible))

    # plot one of the nonseparable labelings
    y_plot = impossible[0]

    plt.figure(figsize=(7, 5))

    for label, color in [(-1, "blue"), (1, "orange")]:
        mask = y_plot == label  # select points with this label

        plt.scatter(
            X_four[mask, 0],
            X_four[mask, 1],
            c=color,
            label=f"label {label}",
            s=100
        )

    for i, point in enumerate(X_four):
        plt.annotate(
            str(i + 1),
            (point[0], point[1]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title("part 2c: nonseparable labeling")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig("part2c_nonseparable.png", dpi=200)
    plt.show()

    # part 2d: check shattering using axis-aligned rectangles
    print("\npart 2d")

    rectangle_points = np.array([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1]
    ])

    print(
        "four rectangle points shattered:",
        rectangle_can_shatter(rectangle_points)
    )

    shattered_five = 0

    for i in range(1000):
        X = rng.standard_normal((5, 2))  # generate five random points

        if rectangle_can_shatter(X):
            shattered_five += 1  # count sets that can be shattered

    print(
        "shattered sets out of 1000 five-point sets:",
        shattered_five
    )


# P3 PAC learning in action: rectangles and thresholds
def P3():
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
  plt.loglog(m_val, mean_errors, label="Mean true error", marker="o")
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
  ) 

  # (c) 
  print("Part 3 (c)")
  m = 4/0.05 * np.log(4/0.05)
  print(f"m >= 4/0.05 * np.log(4/0.05) = {m}")

  plt.axvline(
    x=np.round(m),
    color="green",
    linestyle=":",
    label=f"Classical PAC bound = {np.round(m)}"
  ) # (c) adding the classical PAC bound for when m >= 4/e ln(4/delta)

  plt.xlabel("Sample size m")
  plt.ylabel("Fraction of trials where R > 0.05")
  plt.title("Probability R is greater than 0.05")
  plt.legend()
  plt.show()

  # (c) smallest m where P(R > 0.05) < 0.05
  smallest_ms = 0
  for m in range(100,200,10): # get new range between 100 and 200 to the nearest 10 to find where it first passes the threshold marker (0.05)
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
      smallest_ms = m
      break # since it it iterating from 100 to 200 by 10 the first m that reaches this threshold can end the loop
  print(f"Smallest m where empirical P(R > e) <= delta = {smallest_ms}")

  print("End of Part 3 (c)")

  # (d) Fit a straight line to the log–log plot of mean error versus m 
  print("\nPart 3 (d)")
  slope, intercept = np.polyfit(
    np.log(m_val),        # from (b)
    np.log(mean_errors),  # from (b)
    1                     # 1 = linear line
  )

  beta = -slope
  print(f"slope = {slope}")
  print(f"beta = {beta}")

  # ln R = -b ln m + C
  line = np.exp(intercept) * np.array(m_val) ** slope

  plt.figure(figsize=(8,5))
  plt.loglog(
    m_val,
    line,
    ':',
    label=f"Line w beta = {beta}"
  )
  plt.loglog(m_val,mean_errors, label="Mean true error", marker="o")
  plt.xlabel("Sample size m")
  plt.ylabel("Mean True Error")
  plt.title("Mean True Error vs m samples")
  plt.legend()
  plt.show()

  # beta is an exponent of m so subtract beta by 1 (exponent for m) and 0.5 (exponent for sqrt(m))
  print(f"|{beta} - 1| = {np.abs(beta - 1)}") 
  print(f"|{beta} - 0.5| = {np.abs(beta - 0.5)}")

  print("\nEnd of Part 3 (d)")
  
# P4 Structural Risk Minimization versus cross-validation

# skeleton code provided for P4
def features(x, d):
  return legvander(x, d)[:, 1:] # columns P_1(x)..P_d(x); constant dropped


def P4():
    rng = np.random.default_rng(42)

    # generate random training and test samples
    x_tr = rng.uniform(-1, 1, 200)
    x_te = rng.uniform(-1, 1, 10000)

    # assign labels based on the sine function
    y_tr = np.where(np.sin(3 * np.pi * x_tr) >= 0, 1, -1)
    y_te = np.where(np.sin(3 * np.pi * x_te) >= 0, 1, -1)

    # add 10 percent label noise
    y_tr[rng.random(len(y_tr)) < 0.1] *= -1
    y_te[rng.random(len(y_te)) < 0.1] *= -1

    degrees = np.arange(1, 16)

    train_errors = []
    test_errors = []
    vc_bounds = []
    cv_errors = []

    # use five folds to estimate validation error
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # test polynomial models with different degrees
    for d in degrees:
        X_train = features(x_tr, d)
        X_test = features(x_te, d)

        model = LogisticRegression(
            C=1000,
            max_iter=5000
        )

        model.fit(X_train, y_tr)

        # calculate training and test errors
        train_error = np.mean(model.predict(X_train) != y_tr)
        test_error = np.mean(model.predict(X_test) != y_te)

        # add the vc confidence term to training error
        bound = train_error + fm(d + 1, len(y_tr))

        # calculate the average validation accuracy
        scores = cross_val_score(
            LogisticRegression(C=1000, max_iter=5000),
            X_train,
            y_tr,
            cv=cv,
            scoring="accuracy"
        )

        cv_error = 1 - scores.mean()

        train_errors.append(train_error)
        test_errors.append(test_error)
        vc_bounds.append(bound)
        cv_errors.append(cv_error)

    train_errors = np.array(train_errors)
    test_errors = np.array(test_errors)
    vc_bounds = np.array(vc_bounds)
    cv_errors = np.array(cv_errors)

    # display results for each degree
    print("\npart 4b results")
    print("degree | train error | vc bound | test error | cv error")

    for i, d in enumerate(degrees):
        print(
            f"{d:6d} | "
            f"{train_errors[i]:11.4f} | "
            f"{vc_bounds[i]:8.4f} | "
            f"{test_errors[i]:10.4f} | "
            f"{cv_errors[i]:8.4f}"
        )

    # find the best degree using each method
    srm_degree = int(degrees[np.argmin(vc_bounds)])
    cv_degree = int(degrees[np.argmin(cv_errors)])
    test_degree = int(degrees[np.argmin(test_errors)])

    print("\npart 4c results")
    print("degree selected by srm:", srm_degree)
    print("degree selected by cross validation:", cv_degree)
    print("degree with lowest test error:", test_degree)

    # compare training error, vc bound, and test error
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        degrees,
        train_errors,
        marker="o",
        label="training error"
    )

    ax.plot(
        degrees,
        vc_bounds,
        marker="s",
        label="vc bound"
    )

    ax.plot(
        degrees,
        test_errors,
        marker="^",
        label="test error"
    )

    ax.set_xlabel("polynomial degree d")
    ax.set_ylabel("error")
    ax.set_title("part 4: srm vs test error")
    ax.set_xticks(degrees)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("part4_srm_comparison.png", dpi=200)
    plt.show()
    plt.close(fig)

    # check if the vc bound gives useful values
    print("\npart 4d results")
    print("minimum vc bound:", round(float(vc_bounds.min()), 4))
    print("degrees with vc bound below 0.5:", degrees[vc_bounds < 0.5])

    # compare the polynomial models with 1 nearest neighbor
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_tr.reshape(-1, 1), y_tr)

    knn_train_error = np.mean(
        knn.predict(x_tr.reshape(-1, 1)) != y_tr
    )

    knn_test_error = np.mean(
        knn.predict(x_te.reshape(-1, 1)) != y_te
    )

    print("\npart 4e results")
    print("1nn training error:", round(float(knn_train_error), 4))
    print("1nn test error:", round(float(knn_test_error), 4))

    print("\nend of part 4")


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
  P2()
  print("\nEnd of Part 2")

  # P3 - PAC learning in action: rectangles and thresholds
  print("\nPart 3 Results:")
  
  P3()
  

  print("\nEnd of Part 3")

  # p4 structural risk minimization versus cross validation
  print("\nPart 4 Results:")
  P4()

  # Initialize training set needed for Part 4

  # skeleton code provided
  clf = LinearSVC(C=1e4, max_iter=200000).fit(features(x_tr, d), y_tr)
  p_test = clf.decision_function(features(x_te, d)) # this is the learned p(x)
  y_hat = np.where(p_test > 0, 1, -1) # sign(p(x))


  print("\nEnd of Part 4")

# test part 2 only
# P2()