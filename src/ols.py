from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression


def ols(X, y):
    # Create a linear regression object
    model = LinearRegression()
    X = X
    y = y

    # Fit the linear regression
    model.fit(X, y)

    # Print the coefficients
    print("Interception", model.intercept_)
    print("Coefficents", model.coef_)
    print("Model score", model.score(X, y))

    return model, X, y


def metrics(model, X, y):
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    # Fit the model on the training set
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)
    n, p = X.shape
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    print("Adjusted R^2:", adjusted_r2)



def get_P_Value(X, y):
    # Get the P-values
    f_values, p_values = f_regression(X, y)

    # Print the P-values
    print(p_values)

    # This will return an array of P-values, one for each column in X.
    # A low P-value ( < 0.05) indicates that there is strong evidence
    # against the null hypothesis that the coefficient is equal to zero, meaning that the corresponding independent
    # variable is important.


def get_LassoCV_RidgeCV(X, y):
    lasso = LassoCV()
    lasso.fit(X, y)
    lasso_coef = lasso.coef_
    print("This is the LassoCV", lasso_coef)

    # RidgeCV
    ridge = RidgeCV()
    ridge.fit(X, y)
    ridge_coef = ridge.coef_
    print("This is the RidgeCV", ridge_coef)


def get_RFE(df):
    # Create a linear regression object
    lm = LinearRegression()

    X = df[['P4 Average',  'P2 Average']]
    y = df['Vulnearbilities Rewarded']
    # Create the RFE object and rank each pixel
    rfe = RFE(lm)
    rfe = rfe.fit(X, y)
    print(rfe.ranking_)

    # Print the names and their ranking
    print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), X.columns)))


def get_SelectKBest(X, y):
    # Create the SelectKBest object
    fvalue_selector = SelectKBest(f_regression, k=5)

    # Apply the SelectKBest object to the features and target
    X_kbest = fvalue_selector.fit_transform(X, y)

    # Show results
    print('Original number of features:', X.shape[1])
    print('Reduced number of features:', X_kbest.shape[1])


def visualize(model, df: pd.DataFrame, X):
    predicated_vulnearbilities_rewarded = model.predict(X)
    sns.regplot(x='Vulnearbilities Rewarded',
                y=predicated_vulnearbilities_rewarded, data=df)
    plt.xlabel('Vulnearbilities Rewarded')
    plt.ylabel('Predicted Vulnearbilities rewarded')
    plt.show()


def main():
    df = pd.read_excel(
        '../used_data/cleaned-dataset-log.xlsx')

    X = df[[  'Average Payout',
       'Hall of Famers', 'Number People', 'annocument_count',
        'Reward Range Average', 'Validation Within Hours',
       'P1 Average', 'P2 Average', 'P3 Average', 'P4 Average']]
    y = df['Vulnearbilities Rewarded']

    results, X, y = ols(X, y)
    metrics(results, X, y)
    # visualize(results, df, X)

    # help functions to get more information about the different parameters
    get_P_Value(X, y)
    get_LassoCV_RidgeCV(X, y)
    get_RFE(df)
    get_SelectKBest(X, y)

    # # looking at the smaller programs
    # output: vulnearbilities_rewarded = 0.8564153381222468 + 0.246376072 * average_payout - 0.0000141271570 * number_people + 0.0000179166772 * hall_of_famers
if __name__ == '__main__':
    main()
