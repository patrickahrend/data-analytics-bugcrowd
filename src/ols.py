from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def cleansing_Number_People(df: pd.DataFrame):

    # Read in the data
    df = pd.read_excel(
        '/Users/patrickahrend/Developer/data-analytics-bugcrowd/data-octa/Bugcrowd_industry-asc_24.12.xlsx')

    df = df.dropna(subset=['Number_people'])

    df["Number_people"] = df["Number_people"].str.replace('total', '')
    # Print the first few rows
    df['Number_people'] = df['Number_people'].astype(int)

    return df


def clean_Hall_of_Fame(df: pd.DataFrame):
    df = df.dropna(subset=['Hall_of_famers'])

    df["Hall_of_famers"] = df["Hall_of_famers"].str.replace(
        'View the hall', '')
    df["Hall_of_famers"] = df["Hall_of_famers"].str.replace('View all ', '')
    df["Hall_of_famers"] = df["Hall_of_famers"].str.replace('', '0')
    df["Hall_of_famers"] = df["Hall_of_famers"].astype(int)

    return df

# TODO: fix this methode to clean the data and use in OLS


def cleasne_vulnearbilities_rewarded(df: pd.DataFrame):
    df['Vulnearbilities_rewarded'] = df['Vulnearbilities_rewarded'].astype(int)
    return df


def cleasne_validation_within(df: pd.DataFrame):

    df["Validation_within"] = df["Validation_within"].str.replace(
        '\n', '').str.replace('days', '')

    for ele in df["Validation_within"]:
        if "$" in ele:
            ele = ele.replace(ele, "0")
            ele = ele.strip()
        if "days" in ele:
            ele = ele.replace("days", "")
            ele = ele.strip()
            ele = int(ele)
            ele = ele * 24
            ele = str(ele) + " hours"
        if "day" in ele:
            ele = ele.replace("day", "")
            ele = ele.strip()
            ele = int(ele)
            ele = ele * 24
            ele = str(ele) + " hours"
        if "about" in ele:
            ele = ele.replace("about", "")
            ele = ele.strip()
        if "month" in ele:
            ele = ele.replace("month", "")
            ele = ele.strip()
            ele = int(ele)
            ele = ele * 24 * 30
            ele = str(ele) + " hours"
        if "minutes" in ele:
            ele = ele.replace("minutes", "")
            ele = ele.strip()
            ele = int(ele)
            ele = ele * 60
            ele = str(ele) + " hours"
        if "hours" in ele:
            ele = ele.replace("hours", "")
            ele = ele.strip()
        if "hour" in ele:
            ele = ele.replace("hour", "")
            ele = ele.strip()

        df_final = df_final.append(
            {'Validation_within': ele}, ignore_index=True)
    return df


def clean_average_payout(df: pd.DataFrame):

    df = df.dropna(subset=['Average_payout'])
    df["Average_payout"] = df["Average_payout"].str.replace(
        '$', '').str.replace(',', '').str.replace(' ', '')
    df["Average_payout"] = df["Average_payout"].str.replace("", "0")
    df["Average_payout"] = df["Average_payout"].astype(float)
    return df


def ols(df: pd.DataFrame):
    # Create a linear regression object
    model = LinearRegression()
    X = df.loc[:, ['Number People', 'Hall of Famers',
                   'Average Payout', 'Validation Within']]

    y = df['Vulnearbilities Rewarded']

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


def metrics(X, y, model):
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


def visualize(model, df: pd.DataFrame, X):
    predicated_vulnearbilities_rewarded = model.predict(X)
    sns.regplot(x='Vulnearbilities Rewarded',
                y=predicated_vulnearbilities_rewarded, data=df)
    plt.xlabel('Vulnearbilities Rewarded')
    plt.ylabel('Predicted Vulnearbilities rewarded')
    plt.show()


def main():
    df = pd.read_excel(
        '/Users/patrickahrend/Developer/data-analytics-bugcrowd/data-octa/bugcrowd_modified_24.12.xlsx')
    print(df.columns)

    # replacing placeholders with np values againa
    df["Average Payout"] = df["Average Payout"].replace(0, np.nan)
    df["Number People"] = df["Number People"].replace(0, np.nan)
    df["Hall of Famers"] = df["Hall of Famers"].replace(0, np.nan)

    df = df.dropna(subset=['Hall of Famers', 'Number People',
                           'Validation Within', 'Average Payout', 'Vulnearbilities Rewarded'])
    df['Validation Within'] = df['Validation Within'].astype(int)
    df['Vulnearbilities Rewarded'] = df["Vulnearbilities Rewarded"].astype(
        int)

    # dependent variable is vulnearbilities_rewarded
    # independent variable is average_payout, number_people, hall_of_famers,

    results, X, y = ols(df)
    metrics(X, y, results)

    visualize(results, df, X)

    # output: vulnearbilities_rewarded = 0.8564153381222468 + 0.246376072 * average_payout - 0.0000141271570 * number_people + 0.0000179166772 * hall_of_famers


if __name__ == '__main__':
    main()
