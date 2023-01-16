from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression


def ols(X, y):
    # Create a linear regression object
    model = LinearRegression()
    X = X
    y = y

    # model formula looks as followed:
    # prediction = intercept + coefficient_1 * Number_people + coefficient_2 * Hall_of_famers + coefficient_3 * Average_payout + coefficent_4 * Validation Within

    # scaling
    scaler = StandardScaler()

    # Fit the scaler to the data
    scaler.fit(X)

    # Transform the data using the scaler
    X = scaler.transform(X)

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

    # scaling to not get overfitting
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

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


def get_RFE(X, y):
    # Create a linear regression object
    lm = LinearRegression()

    # Create the RFE object and rank each pixel
    rfe = RFE(lm, 4)
    rfe = rfe.fit(X, y)

    # Print the names and their ranking
    print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), X.columns)))


def get_SelectKBest(X, y):
    # Create the SelectKBest object
    fvalue_selector = SelectKBest(f_regression, k=4)

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
        '/Users/patrickahrend/Developer/data-analytics-bugcrowd/used_data/final-data-16-01.xlsx')
    print(df.columns)
    print(df.dtypes)
    columns = ['Average Payout', 'Hall of Famers', 'Number People', 'Maximum Reword', 'annocument_count',
               'Reward Range Average', 'Validation Within Hours', 'P4 Average', 'P3 Average', 'P2 Average', 'P1 Average']

    for column in columns:
        df[column] = df[column].replace('0', np.nan)

    # dropps 0s and Nan
    df = df.dropna(subset=columns)

    print("Amount of rows", df.shape[0])

    X = df[['Average Payout', 'Hall of Famers', 'Number People', 'Maximum Reword', 'annocument_count',
            'Reward Range Average', 'Validation Within Hours', 'P4 Average', 'P3 Average', 'P2 Average', 'P1 Average']]
    y = df['Vulnearbilities Rewarded']
    results, X, y = ols(X, y)
    metrics(results, X, y)
    # visualize(results, df, X)

    # # looking at the smaller programs
    # df = df.where(df['Vulnearbilities Rewarded'] <= 500)
    # output: vulnearbilities_rewarded = 0.8564153381222468 + 0.246376072 * average_payout - 0.0000141271570 * number_people + 0.0000179166772 * hall_of_famers


if __name__ == '__main__':
    main()
