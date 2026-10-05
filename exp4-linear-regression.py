# ============================================
# EXPERIMENT 04 - LINEAR REGRESSION
# ============================================

# Dataset
# X = Experience in years
# Y = Salary in thousands

X = [2.5, 4.0, 6.5, 11.0, 12.5]
Y = [55, 225, 300, 350, 475]


# ============================================
# 1. CALCULATE LINEAR REGRESSION
# ============================================

def linear_regression(X, Y):

    # Number of observations
    n = len(X)

    # Calculate mean of X
    mean_x = sum(X) / n

    # Calculate mean of Y
    mean_y = sum(Y) / n

    # Calculate numerator and denominator
    numerator = 0
    denominator = 0

    for i in range(n):

        # Numerator:
        # (Xi - meanX) * (Yi - meanY)

        numerator += (
            (X[i] - mean_x) *
            (Y[i] - mean_y)
        )

        # Denominator:
        # (Xi - meanX)^2

        denominator += (
            (X[i] - mean_x) ** 2
        )

    # Calculate slope
    beta = numerator / denominator

    # Calculate intercept
    alpha = mean_y - (
        beta * mean_x
    )

    return alpha, beta


# ============================================
# 2. PREDICTION FUNCTION
# ============================================

def predict(alpha, beta, x):

    # Regression equation:
    #
    # Y = alpha + beta * X

    return alpha + beta * x


# ============================================
# 3. MAIN PROGRAM
# ============================================

alpha, beta = linear_regression(X, Y)


print("=" * 55)
print("      EXPERIMENT 04: LINEAR REGRESSION")
print("=" * 55)


# Display coefficients

print("\n1. MODEL COEFFICIENTS:")

print(f"Slope (β): {beta:.4f}")

print(f"Intercept (α): {alpha:.4f}")

print(
    f"Model Equation: Y = "
    f"{alpha:.4f} + {beta:.4f}X"
)


# ============================================
# 4. ORIGINAL VS PREDICTED VALUES
# ============================================

print("\n2. ORIGINAL DATA VS PREDICTED VALUES:")

print("-" * 55)

print(
    f"{'X(Input)':<12}"
    f"{'Y(Actual)':<15}"
    f"{'Y(Predicted)':<15}"
)

print("-" * 55)


for i in range(len(X)):

    # Predict Y for existing X

    y_pred = predict(
        alpha,
        beta,
        X[i]
    )

    print(
        f"{X[i]:<12.2f}"
        f"{Y[i]:<15.2f}"
        f"{y_pred:<15.2f}"
    )


# ============================================
# 5. PREDICTION FOR NEW DATA
# ============================================

print("\n3. PREDICTION FOR NEW INPUT DATA:")

print("-" * 55)

# New experience value

new_x = 8.0


# Predict salary

new_y = predict(
    alpha,
    beta,
    new_x
)


print(
    f"Predicting Y for X = {new_x}:"
)

print(
    f"Resulting Y = {new_y:.2f}"
)

print("=" * 55)