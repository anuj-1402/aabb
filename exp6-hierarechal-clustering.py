# ============================================================
# EXPERIMENT 06 - HIERARCHICAL CLUSTERING
# Agglomerative Hierarchical Clustering
# Single Linkage + Euclidean Distance
# ============================================================

import math


# ============================================================
# 1. DATASET
# ============================================================

# Dataset format:
# [Age, Spending Score]

data = [
    [18, 85],
    [20, 90],
    [22, 80],
    [19, 95],

    [55, 20],
    [60, 15],
    [50, 25],
    [65, 10]
]


# ============================================================
# 2. EUCLIDEAN DISTANCE
# ============================================================

def euclidean_distance(point1, point2):

    """
    Calculate Euclidean distance between
    two data points.

    Formula:

    d = sqrt(
        (x1 - x2)^2 +
        (y1 - y2)^2
    )
    """

    distance = math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2
    )

    return distance


# ============================================================
# 3. SINGLE LINKAGE DISTANCE
# ============================================================

def cluster_distance(cluster1, cluster2):

    """
    Calculate the distance between two clusters
    using SINGLE LINKAGE.

    Single linkage means:

    Distance between clusters =
    minimum distance between any two points
    belonging to the two clusters.
    """

    # Start with infinity

    min_distance = float('inf')


    # Compare every point in cluster 1
    # with every point in cluster 2

    for point1 in cluster1:

        for point2 in cluster2:

            # Calculate distance between
            # the two points

            distance = euclidean_distance(
                point1,
                point2
            )


            # Keep the minimum distance

            if distance < min_distance:

                min_distance = distance


    return min_distance


# ============================================================
# 4. HIERARCHICAL CLUSTERING
# ============================================================

def hierarchical_clustering(data, target_k):

    """
    Agglomerative Hierarchical Clustering.

    Parameters:
        data      -> input dataset
        target_k  -> required number of clusters

    Returns:
        final clusters
    """


    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    # Initially every data point is treated
    # as an individual cluster.

    clusters = [
        [point]
        for point in data
    ]


    print("\nInitial Clusters:")

    for i, cluster in enumerate(clusters):

        print(
            f"C{i + 1}: {cluster}"
        )


    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------
    # Continue merging until the desired
    # number of clusters is reached.

    while len(clusters) > target_k:


        # Store minimum distance found

        min_distance = float('inf')


        # Store indices of clusters
        # that should be merged

        merge_pair = (0, 1)


        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------
        # Compare every pair of clusters

        for i in range(len(clusters)):

            for j in range(
                i + 1,
                len(clusters)
            ):

                # Calculate single-linkage distance

                distance = cluster_distance(
                    clusters[i],
                    clusters[j]
                )


                # Check whether this is
                # the closest pair

                if distance < min_distance:

                    min_distance = distance

                    merge_pair = (i, j)


        # Get indices of clusters to merge

        i, j = merge_pair


        print(
            f"\nMerging clusters "
            f"{i + 1} and {j + 1}"
        )

        print(
            f"Distance = "
            f"{min_distance:.2f}"
        )


        # ----------------------------------------------------
        # STEP 4
        # ----------------------------------------------------
        # Merge the two clusters

        clusters[i].extend(
            clusters[j]
        )


        # Remove the second cluster

        clusters.pop(j)


        # Display current clusters

        print("Current clusters:")

        for index, cluster in enumerate(clusters):

            print(
                f"C{index + 1}: {cluster}"
            )


    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    return clusters


# ============================================================
# 5. MAIN PROGRAM
# ============================================================

target_clusters = 2


print("=" * 60)

print(
    "EXPERIMENT 06: "
    "HIERARCHICAL CLUSTERING"
)

print("=" * 60)


# Perform hierarchical clustering

final_clusters = hierarchical_clustering(
    data,
    target_clusters
)


# ============================================================
# 6. DISPLAY FINAL RESULT
# ============================================================

print("\n")
print("=" * 60)

print("FINAL CLUSTERS")

print("=" * 60)


for i, cluster in enumerate(
    final_clusters
):

    print(
        f"\nCluster {i + 1}:"
    )


    for point in cluster:

        print(
            f"Age: {point[0]}, "
            f"Spending Score: {point[1]}"
        )


print("\n" + "=" * 60)