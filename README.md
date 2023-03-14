# Seminar - Data Analytics for Cybercrime and Undesirable Online Behaviors


## Approach 


### 1.Data Collection
We used Octoparse to crawl the [programs](https://bugcrowd.com/programs)  information of BugCrowd. In the first loop it crawls the top layer information, then clinks on the link and crawls the program specific information like vunerbility rewarded or average payout and then paginates through all programs to do this for all programs. The data is stored in a csv and excel file.
To do thi in octoparse, a workflow can be created that looks as followed:
<img width="270" alt="image" src="https://user-images.githubusercontent.com/57754014/211603607-6c720a9b-8130-459a-8ccc-93af18b7b305.png">


We also crawled all the data from each tab of the [discovery page](https://bugcrowd.com/programs/discovery/featured) to get categories like industries, technologies,staff pick or highest reward range.

### 2.Data Cleaning
In the first step for each column placeholder dummy values were inserted to still keep the information of the columnm, that will be dropped. Then column dtypes were changed and columns were standarized like validtion within was calculated to all hours for instance. Further did we calculated the average for range variables Then the data was merged from the categorie and the columns were dropped that were not needed.

To use linear regression we had to check the assumptions of linear regression. The assumptions are:
- Linearity
- Homoscedasticity
- Multivariate normality
- No autocorrelation
- Lack of multicollinearity

We detected multicolinarity in the columns P1,P2,P3 removed them. Further did we detect heteroscedasticity. Therefore we log-transformed the data. 

The used data can be found in the [data folder](https://github.com/patrickahrend/data-analytics-bugcrowd/tree/main/used_data). 

### 3.Data Analysis
We used linear regression to interprete the data. The models were created with the sktlearn and statsmodels libraries. The code for the regression looks as followed: 
```python
    # Create a linear regression model object
    model = LinearRegression()

    # Define the input features and the target variable
    X = df[[ 'Average Payout',
       'Hall of Famers', 'Number People', 'annocument_count',
        'Reward Range Average', 'Validation Within Hours',
       'P1 Average', 'P2 Average', 'P3 Average', 'P4 Average']]
    y = df['Vulnearbilities Rewarded']  

    # Fit the linear regression
    model.fit(X, y)

    # Add constant column to input features
    X = sm.add_constant(X)  

    # Create an OLS model object
    ols_model = sm.OLS(y, X)

    # Fit the OLS model to the data
    ols_results = ols_model.fit()

    # Get p-values for coefficients
    p_values = ols_results.pvalues[1:]

    # Set significance level
    alpha = 0.05
    # Extract significant variables
    significant_vars = p_values[p_values < alpha]
    # Print variable names and p-values for significant variables
    for var, p_val in significant_vars.items():
        print(f"{var}: {p_val}")
```

To valid our model we used an online statistics software called [datatab](https://datatab.net/). 

### Heatmap 
We generated a heatmap to see the correlation between the columns.
![image](https://user-images.githubusercontent.com/57754014/215579467-732c9c7d-bae6-4d6d-8ca6-37f153c98fce.png)



### Code Structure 
In the [code folder](https://github.com/patrickahrend/data-analytics-bugcrowd/tree/main/src) the following structure can be found:
- analysis: contains the code for the data cleaning and generation of simple plots
- obversation: contains the code for the obversation and generation of the heatmap 
- regression: contains an example of how we used the library to conduct a linear regression 
- ols: is an early on explanotory approach to linear regression 


