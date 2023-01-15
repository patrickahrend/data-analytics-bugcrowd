from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns

##############
# erstes OLS mit Evas File Sources##
##############

def cleansing_Number_People(df: pd.DataFrame):

    # Read in the data
    df = pd.read_excel(
        r"C:\Users\eva-k\Documents\Studium\TUM\Master MMT\5. Semester\Cybercrime\data-analytics-bugcrowd\data-octa\Bugcrowd_industry-asc_24.12..xlsx")

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
    X = df.loc[:, ['Number_people', 'Hall_of_famers', 'Average_payout']]
    y = df['Vulnearbilities_rewarded']

    # model formula looks as followed:
    # prediction = intercept + coefficient_1 * Number_people + coefficient_2 * Hall_of_famers + coefficient_3 * Average_payout

    # Fit the linear regression
    model.fit(X, y)

    # Print the coefficients
    print(model.intercept_)
    print(model.coef_)
    print(model.score(X, y))

    return model, X, y


def visualize(model, df: pd.DataFrame, X):
    predicated_vulnearbilities_rewarded = model.predict(X)
    sns.scatterplot(x='Vulnearbilities_rewarded',
                    y=predicated_vulnearbilities_rewarded, data=df)
    plt.xlabel('Vulnearbilities_rewarded')
    plt.ylabel('Predicted Vulnearbilities_rewarded')
    plt.show()


def main():
    df = pd.read_excel(
        r"C:\Users\eva-k\Documents\Studium\TUM\Master MMT\5. Semester\Cybercrime\data-analytics-bugcrowd\data-octa\Bugcrowd_industry-asc_24.12..xlsx")
    df = df.dropna(subset=['Number_people', 'Hall_of_famers',
                           'Vulnearbilities_rewarded', 'Validation_within', 'Average_payout'])
    df = cleansing_Number_People(df)
    df = clean_Hall_of_Fame(df)
    df = cleasne_vulnearbilities_rewarded(df)
    df = clean_average_payout(df)
    # dependent variable is vulnearbilities_rewarded
    # independent variable is average_payout, number_people, hall_of_famers,
    results, X, y = ols(df)
    visualize(results, df, X)

    # output: vulnearbilities_rewarded = 0.8564153381222468 + 0.246376072 * average_payout - 0.0000141271570 * number_people + 0.0000179166772 * hall_of_famers


if __name__ == '__main__':
    main()