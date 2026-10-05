# ==============================================================================
# EXPERIMENT 05: K-MEANS CLUSTERING
# ==============================================================================
#
# AIM:
# To implement and understand the K-Means clustering algorithm, perform distance-based
# partitioning on multi-attribute data, and examine cluster convergence.
#
# ------------------------------------------------------------------------------
# DETAILED THEORY:
# ------------------------------------------------------------------------------
# 1. CLUSTERING:
#    Clustering is an unsupervised machine learning technique where unlabelled
#    data is grouped into clusters such that objects within the same cluster
#    share high intra-cluster similarity, while objects across different clusters
#    exhibit high inter-cluster dissimilarity.
#
# 2. K-MEANS ALGORITHM FUNDAMENTALS:
#    K-Means is a centroid-based iterative partitioning algorithm that divides
#    N data observations into K pre-specified clusters.
#    - Centroid: The mean coordinate position of all points currently assigned
#      to that cluster.
#    - Objective: Minimize the Within-Cluster Sum of Squared Errors (SSE / Inertia):
#          SSE = SUM_k [ SUM_{x in C_k} ||x - c_k||^2 ]
#
# 3. EUCLIDEAN DISTANCE METRIC:
#    Distance between a 2D data point P(x1, y1) and a centroid C(c1, c2):
#        d(P, C) = sqrt( (x1 - c1)^2 + (y1 - c2)^2 )
#
# 4. STEP-BY-STEP ALGORITHM:
#    1. Specify K (number of clusters).
#    2. Initialize K centroids randomly from the dataset observations.
#    3. Assignment Step: Measure Euclidean distance from every point to every
#       centroid. Assign each point to its closest centroid.
#    4. Centroid Update Step: Recalculate each centroid as the arithmetic mean
#       of all data points assigned to that cluster.
#    5. Convergence Check: Repeat Assignment and Update steps until centroids
#       do not change position (or max iterations reached).
#
# 5. ELBOW METHOD FOR OPTIMAL K:
#    Runs K-Means across a range of K values and plots SSE vs K. The value of K
#    at which the rate of SSE decrease sharply bends or levels off marks the
#    "elbow", indicating the optimal number of clusters.
#
# ------------------------------------------------------------------------------
# WORKED EXAMPLE (2D STUDENT PERFORMANCE):
# ------------------------------------------------------------------------------
# Target: K = 2 clusters.
# Given 8 points: P1(2,10), P2(2,5), P3(8,4), P4(5,8), P5(7,5), P6(6,4), P7(1,2), P8(4,9)
#
# Step 1: Initial Centroids:
#    Choose C1 = P1(2, 10), C2 = P2(2, 5)
#
# Step 2: Distance & Cluster Assignment (Iteration 1):
#    P1(2,10): d to C1=0.00, d to C2=5.00 => Assigned to Cluster 1
#    P2(2,5):  d to C1=5.00, d to C2=0.00 => Assigned to Cluster 2
#    P3(8,4):  d to C1=8.49, d to C2=6.08 => Assigned to Cluster 2
#    P4(5,8):  d to C1=3.61, d to C2=4.24 => Assigned to Cluster 1
#    P5(7,5):  d to C1=7.07, d to C2=5.00 => Assigned to Cluster 2
#    P6(6,4):  d to C1=7.21, d to C2=4.12 => Assigned to Cluster 2
#    P7(1,2):  d to C1=8.06, d to C2=3.16 => Assigned to Cluster 2
#    P8(4,9):  d to C1=2.24, d to C2=4.47 => Assigned to Cluster 1
#
#    Cluster 1: { P1, P4, P8 }
#    Cluster 2: { P2, P3, P5, P6, P7 }
#
# Step 3: Recalculate Centroids:
#    New C1 = mean( (2,10), (5,8), (4,9) ) = (11/3, 27/3) = (3.67, 9.00)
#    New C2 = mean( (2,5), (8,4), (7,5), (6,4), (1,2) ) = (24/5, 20/5) = (4.80, 4.00)
#
# Step 4: Iteration 2:
#    Recomputing distances with updated C1 and C2 produces the identical cluster
#    memberships. Centroids stabilize -> Convergence achieved!
# ==============================================================================


# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------
# 1. CREATE SAMPLE DATASET
# --------------------------------------------

# Fix the random seed so that the same data
# is generated every time the program runs
np.random.seed(42)

# Create first cluster of students
# Mean = [45, 50]
# Standard deviation = 12
# 100 observations, 2 features
cluster1 = np.random.normal([45, 50], 12, (100, 2))

# Create second cluster of students
# Mean = [70, 65]
cluster2 = np.random.normal([70, 65], 12, (100, 2))

# Create third cluster of students
# Mean = [80, 40]
cluster3 = np.random.normal([80, 40], 12, (100, 2))


# Combine all three clusters into one dataset
data = np.vstack((cluster1, cluster2, cluster3))


# --------------------------------------------
# 2. CREATE DATAFRAME
# --------------------------------------------

# Store the dataset in a Pandas DataFrame
# First column = Math marks
# Second column = Science marks

df = pd.DataFrame(
    data,
    columns=["Math", "Science"]
)


# Display first few rows
print("Original Dataset:")
print(df.head())


# --------------------------------------------
# 3. VISUALIZE ORIGINAL DATASET
# --------------------------------------------

plt.scatter(
    df["Math"],
    df["Science"]
)

plt.xlabel("Math Marks")
plt.ylabel("Science Marks")
plt.title("Original Dataset")

plt.show()


# --------------------------------------------
# 4. SELECT NUMBER OF CLUSTERS
# --------------------------------------------

# We want to divide the data into 3 clusters

K = 3


# --------------------------------------------
# 5. INITIALIZE CENTROIDS
# --------------------------------------------

# Randomly select K data points as initial centroids

centroids = df.sample(
    K,
    random_state=42
).values

print("\nInitial Centroids:")
print(centroids)


# Maximum number of iterations
max_iterations = 100


# --------------------------------------------
# 6. K-MEANS ITERATION
# --------------------------------------------

for iteration in range(max_iterations):

    # List to store cluster assignment
    # for every data point
    clusters = []


    # ----------------------------------------
    # 6.1 ASSIGN EACH POINT TO A CLUSTER
    # ----------------------------------------

    # Loop through every data point
    for point in df.values:

        # Store distances from the current
        # point to every centroid
        distances = []


        # Calculate distance from each centroid
        for centroid in centroids:

            # Euclidean distance formula:
            #
            # distance =
            # sqrt((x1-c1)^2 + (x2-c2)^2)

            distance = np.sqrt(
                np.sum(
                    (point - centroid) ** 2
                )
            )

            distances.append(distance)


        # Find the centroid having
        # minimum distance

        cluster = np.argmin(distances)


        # Store cluster number
        clusters.append(cluster)


    # Convert list into NumPy array
    clusters = np.array(clusters)


    # ----------------------------------------
    # 6.2 STORE OLD CENTROIDS
    # ----------------------------------------

    # Make a copy of the current centroids
    # so that we can check whether they changed

    old_centroids = centroids.copy()


    # ----------------------------------------
    # 6.3 UPDATE CENTROIDS
    # ----------------------------------------

    # Calculate a new centroid for every cluster

    for i in range(K):

        # Select all points belonging
        # to cluster i

        points = df.values[
            clusters == i
        ]


        # Make sure the cluster is not empty

        if len(points) > 0:

            # Calculate the mean of all points
            # This becomes the new centroid

            centroids[i] = np.mean(
                points,
                axis=0
            )


    # ----------------------------------------
    # 6.4 CHECK FOR CONVERGENCE
    # ----------------------------------------

    # If old and new centroids are almost
    # identical, the algorithm has converged

    if np.allclose(
        old_centroids,
        centroids
    ):

        print(
            "\nConverged after",
            iteration + 1,
            "iterations."
        )

        break


# --------------------------------------------
# 7. STORE FINAL CLUSTER ASSIGNMENTS
# --------------------------------------------

# Add cluster number to the DataFrame

df["Cluster"] = clusters


# --------------------------------------------
# 8. DISPLAY FINAL CENTROIDS
# --------------------------------------------

print("\nFinal Centroids:")
print(centroids)


# --------------------------------------------
# 9. DISPLAY CLUSTERED DATA
# --------------------------------------------

print("\nClustered Data:")
print(df)


# --------------------------------------------
# 10. VISUALIZE FINAL CLUSTERS
# --------------------------------------------

for i in range(K):

    # Select points belonging to cluster i

    points = df[
        df["Cluster"] == i
    ]


    # Plot points of this cluster

    plt.scatter(
        points["Math"],
        points["Science"],
        label=f"Cluster {i + 1}"
    )


# --------------------------------------------
# 11. PLOT CENTROIDS
# --------------------------------------------

# Plot the final centroid positions

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)


# --------------------------------------------
# 12. ADD GRAPH LABELS
# --------------------------------------------

plt.xlabel("Math Marks")
plt.ylabel("Science Marks")

plt.title("K-Means Clustering")

plt.legend()

plt.show()