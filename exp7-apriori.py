

from itertools import combinations


# ============================================================
# 1. TRANSACTION DATASET
# ============================================================

# Each transaction represents the items purchased together.

transactions = [

    {"I1", "I2", "I5"},          # T1
    {"I2", "I4"},               # T2
    {"I2", "I3"},               # T3
    {"I1", "I2", "I4"},         # T4
    {"I1", "I3"},               # T5
    {"I2", "I3"},               # T6
    {"I1", "I3"},               # T7
    {"I1", "I2", "I3", "I5"},   # T8
    {"I1", "I2", "I3"}          # T9
]


# ============================================================
# 2. MINIMUM SUPPORT
# ============================================================

# Minimum number of transactions in which
# an itemset must appear.

MIN_SUPPORT_COUNT = 2


# ============================================================
# 3. CALCULATE SUPPORT COUNT
# ============================================================

def get_support_count(itemset, transactions):

    """
    Count the number of transactions
    containing the complete itemset.
    """

    count = 0

    for transaction in transactions:

        # issubset() checks whether every item
        # in the candidate exists in the transaction.

        if itemset.issubset(transaction):

            count += 1

    return count


# ============================================================
# 4. GENERATE CANDIDATE 1-ITEMSETS
# ============================================================

def generate_C1(transactions):

    """
    Generate all unique 1-item candidates.
    """

    items = set()

    for transaction in transactions:

        for item in transaction:

            items.add(item)

    # Convert every item into a frozenset
    # so that it can be used as an itemset.

    C1 = {
        frozenset([item])
        for item in items
    }

    return C1


# ============================================================
# 5. GENERATE NEXT CANDIDATES
# ============================================================

def generate_candidates(previous_frequent, k):

    """
    Generate candidate k-itemsets from
    the previous frequent itemsets.

    This performs:

        JOIN
        +
        PRUNE
    """

    candidates = set()


    # --------------------------------------------------------
    # JOIN STEP
    # --------------------------------------------------------

    previous_list = list(previous_frequent)

    for i in range(len(previous_list)):

        for j in range(i + 1, len(previous_list)):

            union_set = (
                previous_list[i] |
                previous_list[j]
            )


            # Candidate should contain exactly
            # k different items.

            if len(union_set) != k:

                continue


            # ------------------------------------------------
            # PRUNE STEP
            # ------------------------------------------------

            # Generate every (k-1) subset.

            subsets = combinations(
                union_set,
                k - 1
            )


            valid = True

            for subset in subsets:

                subset = frozenset(subset)


                # Every subset must be frequent.

                if subset not in previous_frequent:

                    valid = False

                    break


            # Add candidate only if all subsets
            # are frequent.

            if valid:

                candidates.add(
                    frozenset(union_set)
                )


    return candidates


# ============================================================
# 6. APRIORI ALGORITHM
# ============================================================

def apriori(transactions, min_support_count):

    """
    Main Apriori algorithm.

    Returns a dictionary containing
    all frequent itemsets.
    """

    # Dictionary to store:
    #
    # L1
    # L2
    # L3
    # ...

    frequent_itemsets = {}


    # --------------------------------------------------------
    # STEP 1: Generate C1
    # --------------------------------------------------------

    candidates = generate_C1(transactions)


    # --------------------------------------------------------
    # STEP 2: Find L1
    # --------------------------------------------------------

    current_frequent = {}

    for candidate in candidates:

        support = get_support_count(
            candidate,
            transactions
        )


        # Keep candidate if support
        # satisfies minimum support.

        if support >= min_support_count:

            current_frequent[candidate] = support


    frequent_itemsets[1] = current_frequent


    # --------------------------------------------------------
    # STEP 3: Generate L2, L3, ...
    # --------------------------------------------------------

    k = 2

    while frequent_itemsets[k - 1]:

        previous_frequent = set(
            frequent_itemsets[k - 1].keys()
        )


        # Generate candidate k-itemsets

        candidates = generate_candidates(
            previous_frequent,
            k
        )


        current_frequent = {}


        # ----------------------------------------------------
        # Count support of every candidate
        # ----------------------------------------------------

        for candidate in candidates:

            support = get_support_count(
                candidate,
                transactions
            )


            if support >= min_support_count:

                current_frequent[candidate] = support


        # Stop if no frequent itemsets exist

        if not current_frequent:

            break


        # Store current frequent itemsets

        frequent_itemsets[k] = current_frequent


        k += 1


    return frequent_itemsets


# ============================================================
# 7. RUN APRIORI
# ============================================================

frequent_itemsets = apriori(
    transactions,
    MIN_SUPPORT_COUNT
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("=" * 60)

print(
    "EXPERIMENT 07: APRIORI ALGORITHM "
    "FOR MARKET BASKET ANALYSIS"
)

print("=" * 60)


for k, itemsets in frequent_itemsets.items():

    print(
        f"\nFREQUENT {k}-ITEMSETS"
    )

    print("-" * 45)


    # Sort itemsets for cleaner output

    sorted_itemsets = sorted(
        itemsets.items(),
        key=lambda x: sorted(x[0])
    )


    for itemset, support in sorted_itemsets:

        items = ", ".join(
            sorted(itemset)
        )

        print(
            f"{{{items}}}"
            f"  -> Support Count = {support}"
        )


# ============================================================
# 9. SUPPORT PERCENTAGE
# ============================================================

print("\n" + "=" * 60)

print("SUPPORT PERCENTAGES")

print("=" * 60)


total_transactions = len(
    transactions
)


for k, itemsets in frequent_itemsets.items():

    for itemset, support in itemsets.items():

        support_percentage = (
            support /
            total_transactions
        ) * 100


        items = ", ".join(
            sorted(itemset)
        )


        print(
            f"{{{items}}}"
            f" -> {support_percentage:.2f}%"
        )


# ============================================================
# 10. ASSOCIATION RULE
# ============================================================

def calculate_confidence(
    antecedent,
    consequent,
    transactions
):

    """
    Calculate:

    Confidence(A -> B)

              Support(A U B)
    = ---------------------------
                Support(A)
    """

    union = (
        antecedent |
        consequent
    )


    support_union = get_support_count(
        union,
        transactions
    )


    support_antecedent = get_support_count(
        antecedent,
        transactions
    )


    if support_antecedent == 0:

        return 0


    return (
        support_union /
        support_antecedent
    )


# Example:

antecedent = frozenset({"I1"})

consequent = frozenset({"I2"})


confidence = calculate_confidence(
    antecedent,
    consequent,
    transactions
)


print("\n" + "=" * 60)

print("ASSOCIATION RULE")

print("=" * 60)


print(
    "{I1} -> {I2}"
)

print(
    f"Confidence = "
    f"{confidence * 100:.2f}%"
)


print("=" * 60)