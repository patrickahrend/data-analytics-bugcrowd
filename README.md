# Seminar - Data Analytics for Cybercrime and Undesirable Online Behaviors


## Approach 


### 1.Data Collection
We used Octoparse to crawl the [programs](https://bugcrowd.com/programs)  information of BugCrowd. In the first loop it crawls the top layer information, then clinks on the link and crawls the program specific information like vunerbility rewarded or average payout and then paginates through all programs to do this for all programs. The data is stored in a csv and excel file.
To do thi in octoparse, a workflow can be created that looks as followed:
<img width="270" alt="image" src="https://user-images.githubusercontent.com/57754014/211603607-6c720a9b-8130-459a-8ccc-93af18b7b305.png">


We also crawled all the data from each tab of the [discovery page](https://bugcrowd.com/programs/discovery/featured) to get categories like industries, technologies,staff pick or highest reward range.

### 2.Data Cleaning
In the first step for each column placeholder dummy values were inserted to still keep the information of the columnm, that will be dropped. Then column dtypes were changed and columns were standarized like validation within was calculated to all hours for instance. Further we encoded the column is_safe_habour to 0 = not safe habour, 1 = partialy safe habour and 2= safe habor. Then the data was merged from the categorie and the columns were dropped that were not needed. The data was then exported to a csv file.


### 3.Data Analysis
With the data plots were created to get a better understand which are stored under [diagrams](https://github.com/patrickahrend/data-analytics-bugcrowd/tree/main/diagrams).
Later we created a model to predict the average payout of a program. The model was created with the help of the sklearn library. The model was trained with the data from the discovery page and the data from the programs page. The model was trained with the following parameters: 
```python
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
```


### Heatmap 

<img width="357" alt="image" src="https://user-images.githubusercontent.com/57754014/215579279-8c26999d-68b6-42fa-a3b0-03f084162219.png">




