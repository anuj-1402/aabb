# ============================================================
# EXPERIMENT 07 - APRIORI ALGORITHM
# MARKET BASKET ANALYSIS
# WITHOUT ANY EXTERNAL LIBRARIES
# ============================================================


# ------------------------------------------------------------
# 1. TRANSACTION DATABASE
# ------------------------------------------------------------

transactions = [
    ["I1", "I2", "I5"],          # T1
    ["I2", "I4"],               # T2
    ["I2", "I3"],               # T3
    ["I1", "I2", "I4"],         # T4
    ["I1", "I3"],               # T5
    ["I2", "I3"],               # T6
    ["I1", "I3"],               # T7
    ["I1", "I2", "I3", "I5"],   # T8
    ["I1", "I2", "I3"]          # T9
]


# Minimum number of transactions in which
# an itemset must occur.

MIN_SUPPORT = 2


# ------------------------------------------------------------
# 2. FUNCTION TO COUNT SUPPORT
# ------------------------------------------------------------

def support_count(itemset):

    """
    Count how many transactions contain
    all items of the given itemset.
    """

    count = 0

    for transaction in transactions:

        found = True

        # Check every item in the itemset

        for item in itemset:

            if item not in transaction:

                found = False
                break

        # If all items were found,
        # increase the support count.

        if found:
            count += 1

    return count


# ------------------------------------------------------------
# 3. GENERATE 1-ITEM CANDIDATES
# ------------------------------------------------------------

def generate_C1():

    """
    Generate all unique 1-item candidates.
    """

    items = []

    for transaction in transactions:

        for item in transaction:

            if item not in items:

                items.append(item)


    # Sort items for consistent output

    items.sort()


    # Convert each item into a list
    # representing a 1-itemset.

    C1 = []

    for item in items:

        C1.append([item])

    return C1


# ------------------------------------------------------------
# 4. FIND FREQUENT ITEMSETS
# ------------------------------------------------------------

def get_frequent_itemsets(candidates):

    """
    Calculate support for every candidate
    and keep candidates satisfying
    minimum support.
    """

    frequent = []

    for candidate in candidates:

        count = support_count(candidate)

        if count >= MIN_SUPPORT:

            # Store:
            # [itemset, support count]

            frequent.append(
                [candidate, count]
            )

    return frequent


# ------------------------------------------------------------
# 5. CHECK WHETHER A SUBSET IS FREQUENT
# ------------------------------------------------------------

def is_frequent_subset(subset, previous_frequent):

    """
    Check whether a subset exists
    in the previous frequent itemsets.
    """

    for itemset, count in previous_frequent:

        if subset == itemset:

            return True

    return False


# ------------------------------------------------------------
# 6. GENERATE CANDIDATES USING JOIN + PRUNE
# ------------------------------------------------------------

def generate_candidates(previous_frequent, k):

    """
    Generate k-item candidates from
    frequent (k-1)-itemsets.

    This performs:

        JOIN
          +
        PRUNE
    """

    candidates = []


    # Extract only the itemsets

    previous = []

    for itemset, count in previous_frequent:

        previous.append(itemset)


    # --------------------------------------------------------
    # JOIN STEP
    # --------------------------------------------------------

    for i in range(len(previous)):

        for j in range(i + 1, len(previous)):

            set1 = previous[i]
            set2 = previous[j]


            # Combine both itemsets

            combined = []

            for item in set1:

                if item not in combined:

                    combined.append(item)


            for item in set2:

                if item not in combined:

                    combined.append(item)


            # Sort the candidate

            combined.sort()


            # Candidate must contain exactly k items

            if len(combined) != k:

                continue


            # ------------------------------------------------
            # PRUNE STEP
            # ------------------------------------------------

            valid = True


            # Generate every (k-1)-item subset
            # manually.

            for remove_index in range(k):

                subset = []

                for index in range(k):

                    if index != remove_index:

                        subset.append(
                            combined[index]
                        )


                # If any subset is not frequent,
                # remove this candidate.

                if not is_frequent_subset(
                    subset,
                    previous_frequent
                ):

                    valid = False

                    break


            # Add candidate if it passes pruning

            if valid and combined not in candidates:

                candidates.append(combined)


    return candidates


# ------------------------------------------------------------
# 7. APRIORI ALGORITHM
# ------------------------------------------------------------

def apriori():

    """
    Main Apriori algorithm.
    """

    all_frequent = {}


    # --------------------------------------------------------
    # STEP 1: Generate C1
    # --------------------------------------------------------

    C1 = generate_C1()


    # --------------------------------------------------------
    # STEP 2: Find L1
    # --------------------------------------------------------

    L = get_frequent_itemsets(C1)


    if len(L) == 0:

        return all_frequent


    all_frequent[1] = L


    # Start generating 2-itemsets

    k = 2


    # --------------------------------------------------------
    # STEP 3: Generate L2, L3, ...
    # --------------------------------------------------------

    while True:

        # Generate candidate k-itemsets

        C = generate_candidates(
            L,
            k
        )


        # Stop if no candidates exist

        if len(C) == 0:

            break


        # Find frequent candidates

        L = get_frequent_itemsets(C)


        # Stop if no frequent itemsets remain

        if len(L) == 0:

            break


        # Store frequent itemsets

        all_frequent[k] = L


        # Move to next level

        k += 1


    return all_frequent


# ------------------------------------------------------------
# 8. DISPLAY TRANSACTIONS
# ------------------------------------------------------------

print("=" * 60)

print("        APRIORI ALGORITHM")
print("        MARKET BASKET ANALYSIS")

print("=" * 60)


print("\nTRANSACTION DATABASE")
print("-" * 40)


for i in range(len(transactions)):

    print(
        "T" + str(i + 1),
        "->",
        transactions[i]
    )


# ------------------------------------------------------------
# 9. RUN APRIORI
# ------------------------------------------------------------

frequent_itemsets = apriori()


# ------------------------------------------------------------
# 10. DISPLAY FREQUENT ITEMSETS
# ------------------------------------------------------------

print("\n")

print("=" * 60)

print("FREQUENT ITEMSETS")

print("=" * 60)


for k in frequent_itemsets:

    print(
        "\nL" + str(k) +
        " FREQUENT " +
        str(k) +
        "-ITEMSETS"
    )

    print("-" * 40)


    for itemset, count in frequent_itemsets[k]:

        print(
            str(itemset),
            "-> Support Count =",
            count
        )


# ------------------------------------------------------------
# 11. DISPLAY SUPPORT PERCENTAGE
# ------------------------------------------------------------

print("\n")

print("=" * 60)

print("SUPPORT PERCENTAGE")

print("=" * 60)


total_transactions = len(transactions)


for k in frequent_itemsets:

    for itemset, count in frequent_itemsets[k]:

        percentage = (
            count /
            total_transactions
        ) * 100


        print(
            str(itemset),
            "->",
            round(percentage, 2),
            "%"
        )


# ------------------------------------------------------------
# 12. SIMPLE ASSOCIATION RULE
# ------------------------------------------------------------

print("\n")

print("=" * 60)

print("ASSOCIATION RULE")
print("=" * 60)


# Example rule:

# {I1} -> {I2}

antecedent = ["I1"]

consequent = ["I2"]


# Count support of antecedent

support_antecedent = support_count(
    antecedent
)


# Create combined itemset

combined = ["I1", "I2"]


# Count support of combined itemset

support_combined = support_count(
    combined
)


# Calculate confidence

confidence = (
    support_combined /
    support_antecedent
) * 100


print(
    "{I1} -> {I2}"
)

print(
    "Support Count =",
    support_combined
)

print(
    "Confidence =",
    round(confidence, 2),
    "%"
)


print("\n")

print("=" * 60)

print("PROGRAM COMPLETED")

print("=" * 60)