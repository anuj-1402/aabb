# ==============================================================================
# EXPERIMENT 03: DECISION TREE CLASSIFICATION (ID3 ALGORITHM)
# ==============================================================================
#
# AIM:
# To understand and implement a Decision Tree Classification model using the ID3
# (Iterative Dichotomiser 3) algorithm based on Information Gain and Entropy.
#
# ------------------------------------------------------------------------------
# DETAILED THEORY:
# ------------------------------------------------------------------------------
# 1. CLASSIFICATION:
#    Classification is a supervised machine learning technique where a model
#    learns from labeled training records to predict discrete categorical class
#    labels (e.g., buys_computer: yes/no) for new, unseen data observations.
#
# 2. DECISION TREE ANATOMY:
#    - Root Node: The topmost attribute test that best splits the dataset.
#    - Internal Nodes: Intermediate decision checks on specific attributes.
#    - Branches: Possible values/outcomes of that attribute check.
#    - Leaf Nodes: Terminal nodes representing the final classification label.
#
# 3. ID3 ALGORITHM & CORE METRICS:
#    ID3 builds the tree top-down greedily by choosing at each node the attribute
#    that yields the highest Information Gain (i.e., greatest entropy reduction).
#
#    A. ENTROPY:
#       Measures the disorder, impurity, or uncertainty in a sample set S:
#           Entropy(S) = - SUM [ p_i * log2(p_i) ]
#       where p_i is the proportion of records belonging to class i.
#       - If all instances belong to the exact same class (pure), Entropy = 0.
#       - If classes are equally divided (50% yes, 50% no), Entropy = 1.0.
#
#    B. INFORMATION GAIN:
#       The expected reduction in entropy achieved by partitioning dataset S
#       on attribute A:
#           Gain(S, A) = Entropy(S) - SUM [ (|S_v| / |S|) * Entropy(S_v) ]
#       where v ranges over all possible values of attribute A, and S_v is the
#       subset of S where attribute A takes value v.
#
# 4. STOPPING CRITERIA:
#    1. All records at the current node belong to the identical class (pure leaf).
#    2. No remaining attributes left to test (assign majority class).
#    3. Subset of records is empty (assign parent's majority class).
#
# ------------------------------------------------------------------------------
# WORKED EXAMPLE (ALLELECTRONICS DATASET):
# ------------------------------------------------------------------------------
# Consider 14 customer records with target 'buys_computer':
# Total records: 14 (9 'yes', 5 'no')
#
# Step 1: Calculate Total Dataset Entropy:
#    Entropy(S) = - (9/14)*log2(9/14) - (5/14)*log2(5/14) = 0.9403
#
# Step 2: Calculate Information Gain for Candidate Attributes:
#    - Attribute 'Age' (youth: 2 yes/3 no; middle_aged: 4 yes/0 no; senior: 3 yes/2 no):
#      Entropy(youth) = 0.9710, Entropy(middle_aged) = 0.0000, Entropy(senior) = 0.9710
#      Expected Entropy E(Age) = (5/14)*0.9710 + (4/14)*0 + (5/14)*0.9710 = 0.6936
#      Gain(S, Age) = 0.9403 - 0.6936 = 0.2467  <--- MAXIMUM GAIN!
#
#    - Attribute 'Student' (no: 3 yes/4 no; yes: 6 yes/1 no):
#      E(Student) = 0.7885 => Gain(S, Student) = 0.1518
#
#    - Attribute 'Credit_Rating' (fair: 6 yes/2 no; excellent: 3 yes/3 no):
#      E(Credit_Rating) = 0.8922 => Gain(S, Credit_Rating) = 0.0481
#
# Step 3: Root Selection & Branching:
#    - Since Age has the maximum Information Gain (0.2467), 'Age' is the ROOT NODE.
#    - Branch 'middle_aged': 100% 'yes' -> Leaf Node = 'yes'.
#    - Branch 'youth': Split next on 'Student' (student='no' -> 'no', student='yes' -> 'yes').
#    - Branch 'senior': Split next on 'Credit_Rating' (fair -> 'yes', excellent -> 'no').
#
# Step 4: Prediction on Unseen Sample:
#    Test Sample: {'age': 'youth', 'student': 'yes', 'credit_rating': 'fair'}
#    Classification Path: Age('youth') -> Student('yes') => Prediction = "yes".
# ==============================================================================

import math


# ============================================================
# 1. DATASET
# ============================================================

# AllElectronics dataset
# Target attribute = buys_computer

dataset = [
    {'age': 'youth', 'student': 'no',
     'credit_rating': 'fair', 'buys_computer': 'no'},

    {'age': 'youth', 'student': 'no',
     'credit_rating': 'excellent', 'buys_computer': 'no'},

    {'age': 'middle_aged', 'student': 'no',
     'credit_rating': 'fair', 'buys_computer': 'yes'},

    {'age': 'senior', 'student': 'no',
     'credit_rating': 'fair', 'buys_computer': 'yes'},

    {'age': 'senior', 'student': 'no',
     'credit_rating': 'fair', 'buys_computer': 'yes'},

    {'age': 'senior', 'student': 'yes',
     'credit_rating': 'excellent', 'buys_computer': 'no'},

    {'age': 'middle_aged', 'student': 'yes',
     'credit_rating': 'excellent', 'buys_computer': 'yes'},

    {'age': 'youth', 'student': 'no',
     'credit_rating': 'fair', 'buys_computer': 'no'},

    {'age': 'youth', 'student': 'yes',
     'credit_rating': 'fair', 'buys_computer': 'yes'},

    {'age': 'senior', 'student': 'yes',
     'credit_rating': 'fair', 'buys_computer': 'yes'},

    {'age': 'youth', 'student': 'yes',
     'credit_rating': 'excellent', 'buys_computer': 'yes'},

    {'age': 'middle_aged', 'student': 'no',
     'credit_rating': 'excellent', 'buys_computer': 'yes'},

    {'age': 'middle_aged', 'student': 'yes',
     'credit_rating': 'fair', 'buys_computer': 'yes'},

    {'age': 'senior', 'student': 'no',
     'credit_rating': 'excellent', 'buys_computer': 'no'}
]


# ============================================================
# 2. CALCULATE ENTROPY
# ============================================================

def calculate_entropy(data):

    # If dataset is empty
    if len(data) == 0:
        return 0

    # Count occurrences of each class
    counts = {}

    for row in data:

        label = row['buys_computer']

        counts[label] = counts.get(label, 0) + 1

    # Total number of records
    total = len(data)

    entropy = 0

    # Calculate entropy
    for count in counts.values():

        probability = count / total

        entropy -= probability * math.log2(probability)

    return entropy


# ============================================================
# 3. CALCULATE INFORMATION GAIN
# ============================================================

def calculate_information_gain(data, attribute):

    # Entropy before splitting
    total_entropy = calculate_entropy(data)

    # Create groups based on attribute values
    subsets = {}

    for row in data:

        value = row[attribute]

        if value not in subsets:
            subsets[value] = []

        subsets[value].append(row)

    # Calculate weighted entropy
    weighted_entropy = 0

    total = len(data)

    for subset in subsets.values():

        probability = len(subset) / total

        weighted_entropy += (
            probability *
            calculate_entropy(subset)
        )

    # Information Gain
    information_gain = (
        total_entropy -
        weighted_entropy
    )

    return information_gain


# ============================================================
# 4. BUILD DECISION TREE
# ============================================================

def build_tree(data, attributes):

    # Get all target labels
    labels = [
        row['buys_computer']
        for row in data
    ]

    # --------------------------------------------------------
    # BASE CASE 1:
    # All records have the same class
    # --------------------------------------------------------

    if labels.count(labels[0]) == len(labels):

        return labels[0]


    # --------------------------------------------------------
    # BASE CASE 2:
    # No attributes remaining
    # Return majority class
    # --------------------------------------------------------

    if len(attributes) == 0:

        return max(
            set(labels),
            key=labels.count
        )


    # --------------------------------------------------------
    # CALCULATE INFORMATION GAIN
    # FOR EVERY ATTRIBUTE
    # --------------------------------------------------------

    gains = {}

    for attribute in attributes:

        gains[attribute] = (
            calculate_information_gain(
                data,
                attribute
            )
        )


    # Select attribute with maximum gain

    best_attribute = max(
        gains,
        key=gains.get
    )


    # Create tree node

    tree = {
        best_attribute: {}
    }


    # Get possible values of best attribute

    values = set(
        row[best_attribute]
        for row in data
    )


    # Remaining attributes

    remaining_attributes = [
        attribute
        for attribute in attributes
        if attribute != best_attribute
    ]


    # --------------------------------------------------------
    # RECURSIVELY BUILD SUBTREES
    # --------------------------------------------------------

    for value in values:

        # Select records having this value

        subset = [
            row
            for row in data
            if row[best_attribute] == value
        ]


        # Recursively create subtree

        tree[best_attribute][value] = build_tree(
            subset,
            remaining_attributes
        )


    return tree


# ============================================================
# 5. PREDICT CLASS FOR A NEW CUSTOMER
# ============================================================

def predict(tree, sample):

    # If tree is already a class label
    if not isinstance(tree, dict):

        return tree


    # Get root attribute
    root = list(tree.keys())[0]

    # Get value of that attribute
    value = sample[root]


    # Follow corresponding branch

    if value in tree[root]:

        return predict(
            tree[root][value],
            sample
        )

    else:

        return "Unknown"


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

attributes = [
    'age',
    'student',
    'credit_rating'
]


# Build decision tree

decision_tree = build_tree(
    dataset,
    attributes
)


# Display decision tree

print("=" * 60)
print("EXPERIMENT 03: DECISION TREE CLASSIFIER (ID3)")
print("=" * 60)

print("\nGenerated Decision Tree:")
print(decision_tree)


# ============================================================
# 7. TEST SAMPLE 1
# ============================================================

test1 = {
    'age': 'youth',
    'student': 'yes',
    'credit_rating': 'fair'
}

print("\nTest Sample 1:")
print(test1)

prediction1 = predict(
    decision_tree,
    test1
)

print("Prediction:", prediction1)


# ============================================================
# 8. TEST SAMPLE 2
# ============================================================

test2 = {
    'age': 'senior',
    'student': 'no',
    'credit_rating': 'excellent'
}

print("\nTest Sample 2:")
print(test2)

prediction2 = predict(
    decision_tree,
    test2
)

print("Prediction:", prediction2)


# ============================================================
# 9. DISPLAY INFORMATION GAIN
# ============================================================

print("\nInformation Gain:")

for attribute in attributes:

    gain = calculate_information_gain(
        dataset,
        attribute
    )

    print(
        attribute,
        "->",
        round(gain, 4)
    )


print("\n" + "=" * 60)