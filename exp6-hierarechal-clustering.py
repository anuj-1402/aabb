# ==============================================================================
# EXPERIMENT 06: HIERARCHICAL CLUSTERING (AGGLOMERATIVE SINGLE LINKAGE)
# ==============================================================================
#
# AIM:
# To implement and understand Agglomerative (bottom-up) Hierarchical Clustering
# using Euclidean Distance and Single Linkage, and trace the cluster merge hierarchy.
#
# ------------------------------------------------------------------------------
# DETAILED THEORY:
# ------------------------------------------------------------------------------
# 1. HIERARCHICAL CLUSTERING:
#    Hierarchical clustering builds a hierarchy of nested clusters represented
#    as a tree diagram called a Dendrogram.
#    Unlike K-Means:
#    - Does not require pre-specifying the number of clusters (K) up front.
#    - Is deterministic (no random centroid initialization dependencies).
#
# 2. TWO MAIN PARADIGMS:
#    - Agglomerative (Bottom-Up): Starts with every point as an individual
#      cluster. Iteratively merges the two closest clusters until all points
#      unify or target K is met.
#    - Divisive (Top-Down): Starts with all points in one master cluster and
#      recursively splits it down.
#
# 3. DISTANCE & LINKAGE MEASURES:
#    Euclidean Distance between points P(x1, y1) and Q(x2, y2):
#        d(P, Q) = sqrt( (x1 - x2)^2 + (y1 - y2)^2 )
#
#    Linkage Criteria between Cluster A and Cluster B:
#    - Single Linkage (Minimum Distance):
#          d(A, B) = min { dist(p, q) : p in A, q in B }
#    - Complete Linkage (Maximum Distance):
#          d(A, B) = max { dist(p, q) : p in A, q in B }
#    - Average Linkage:
#          d(A, B) = (1 / (|A|*|B|)) * SUM [ dist(p, q) ]
#    - Ward's Method:
#          Minimizes the increase in total within-cluster variance.
#
# 4. DENDROGRAM:
#    A tree visual where leaf nodes are data points and the vertical branch
#    height reflects the distance threshold at which clusters were merged.
#
# ------------------------------------------------------------------------------
# WORKED EXAMPLE (CUSTOMER AGE VS SPENDING SCORE):
# ------------------------------------------------------------------------------
# Target: K = 2 clusters.
# Points: P1[18,85], P2[20,90], P3[22,80], P4[55,20], P5[60,15]
#
# Step 1: Initial Clusters:
#    C1={P1}, C2={P2}, C3={P3}, C4={P4}, C5={P5}
#
# Step 2: Proximity Matrix (Euclidean Distances):
#    dist(P1, P2) = sqrt((18-20)^2 + (85-90)^2) = sqrt(4+25)  = 5.39  <-- MINIMUM!
#    dist(P1, P3) = sqrt((18-22)^2 + (85-80)^2) = sqrt(16+25) = 6.40
#    dist(P2, P3) = sqrt((20-22)^2 + (90-80)^2) = sqrt(4+100) = 10.20
#    dist(P4, P5) = sqrt((55-60)^2 + (20-15)^2) = sqrt(25+25) = 7.07
#    Cross-group distances (e.g. dist(P3, P4)) are large (~68.5).
#
# Step 3: First Merge:
#    Smallest distance is 5.39 between P1 and P2.
#    Merge P1 and P2 -> New Cluster {P1, P2}.
#    Remaining: {P1, P2}, {P3}, {P4}, {P5}.
#
# Step 4: Second Merge (Single Linkage):
#    dist({P1, P2}, P3) = min(dist(P1,P3), dist(P2,P3)) = min(6.40, 10.20) = 6.40.
#    Next minimum is 6.40 -> Merge {P1, P2} with P3 -> {P1, P2, P3}.
#    Remaining: {P1, P2, P3}, {P4}, {P5}.
#
# Step 5: Third Merge:
#    dist(P4, P5) = 7.07.
#    dist({P1, P2, P3}, {P4}) = 68.48.
#    Smallest is 7.07 -> Merge P4 and P5 -> {P4, P5}.
#    Remaining: {P1, P2, P3} and {P4, P5}.
#
# Step 6: Target K = 2 Reached:
#    - Cluster 1: { P1, P2, P3 } (Young, high spending score)
#    - Cluster 2: { P4, P5 }     (Senior, low spending score)
# ==============================================================================


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