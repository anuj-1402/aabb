# ==============================================================================
# EXPERIMENT 04: LINEAR REGRESSION
# ==============================================================================
#
# AIM:
# To implement Simple Linear Regression using the Ordinary Least Squares (OLS)
# method to model the relationship between an independent variable and a
# dependent variable, and perform continuous numerical predictions.
#
# ------------------------------------------------------------------------------
# DETAILED THEORY:
# ------------------------------------------------------------------------------
# 1. LINEAR REGRESSION:
#    Linear Regression is a fundamental supervised learning and statistical
#    technique used to model the linear relationship between a dependent (target)
#    variable Y and one or more independent (explanatory) variables X.
#
# 2. SIMPLE LINEAR REGRESSION MODEL EQUATION:
#        Y = alpha + beta * X
#    where:
#    - Y is the predicted dependent variable (e.g., Salary).
#    - X is the independent variable (e.g., Experience in years).
#    - beta is the Slope (regression coefficient): the change in Y per unit change in X.
#    - alpha is the Intercept: the value of Y when X = 0 (Y-axis crossing point).
#
# 3. ORDINARY LEAST SQUARES (OLS) METHOD:
#    In real-world data, observed points rarely lie exactly on a single line.
#    The vertical deviation between an actual point Y_i and the fitted line Y_hat_i
#    is the error or residual: e_i = Y_i - Y_hat_i.
#    OLS minimizes the Sum of Squared Residuals (SSE):
#        Minimize: SSE = SUM [ (Y_i - (alpha + beta * X_i))^2 ]
#
# 4. MATHEMATICAL FORMULAS:
#    Let x_bar be the mean of X, and y_bar be the mean of Y.
#        beta  = SUM [ (X_i - x_bar) * (Y_i - y_bar) ] / SUM [ (X_i - x_bar)^2 ]
#        alpha = y_bar - (beta * x_bar)
#
# ------------------------------------------------------------------------------
# WORKED EXAMPLE (EXPERIENCE VS SALARY):
# ------------------------------------------------------------------------------
# Given 5 observations:
#    X (Years of Experience): [2.5, 4.0, 6.5, 11.0, 12.5]
#    Y (Salary in thousands): [55,  225, 300, 350,  475]
#
# Step 1: Compute Means:
#    x_bar = (2.5 + 4.0 + 6.5 + 11.0 + 12.5) / 5 = 36.5 / 5 = 7.30
#    y_bar = (55 + 225 + 300 + 350 + 475) / 5    = 1405 / 5 = 281.00
#
# Step 2: Compute Sum of Deviations:
#    SUM [ (X_i - x_bar)^2 ] = 23.04 + 10.89 + 0.64 + 13.69 + 27.04 = 75.30
#    SUM [ (X_i - x_bar)*(Y_i - y_bar) ] = 1084.80 + 184.80 - 15.20 + 255.30 + 1008.80
#                                        = 2518.50
#
# Step 3: Compute Coefficients:
#    beta  = 2518.50 / 75.30 = 33.4462
#    alpha = 281.00 - (33.4462 * 7.30) = 36.8427
#    Fitted Line: Salary = 36.8427 + 33.4462 * Experience
#
# Step 4: Prediction for New Experience (X = 8.0 years):
#    Predicted Salary = 36.8427 + 33.4462 * (8.0) = 304.41 thousand rupees (~₹3,04,412).
# ==============================================================================


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