# --------------------------------------------
# EXPERIMENT 5 - K-MEANS CLUSTERING
# --------------------------------------------

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