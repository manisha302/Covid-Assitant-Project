# Covid-Assitant-Project
## AIM ##
Project aim was to create a prediction algorithm which can detect the posibility of a person getting infected based on his/her sorrounding condition which is based on several different different attributes like hygene condition, new cases,existing cases, death ratio, disease of a person etc.
## Dataset
dataset was taken from this [git repo](https://github.com/owid/covid-19-data/tree/master/public/data) which is further a collection of data from [our world data](https://ourworldindata.org/coronavirus)
i have attached a image od dataset description from the github repository whose link i have been given above
![owid-covid-data](https://user-images.githubusercontent.com/70663378/139447664-2aa70603-cb45-4a8c-8fe3-3fcdbec6018b.png)

# Datasets has more than 80% null values in more than 15 attributes so, this was the biggest challenge to deal with null values
## EDA 
### Seaborn ,Matplotlib and StatsModel
 Detailed EDA for visualising all missing values, outliers, confidence intervals of particular attributes using:
1.Displot with kde
2.Box plot - Numerical feaures
3.Violin Plot- Categorical features
4.lmplot
5.Pie chart
6.Strip plot
7.regplot

# Treating NULL VALUES 
## Feature Engineering
### Droping variables having null values greater than 70%
### Outlier treatment using interquartile range
### P-Test - statistical features dependence test for keeping only relevant features
### statsmodel for plotting linear regression fit 

## Modelling
1.Linear Regression 
2.Ridge, Lasso Regression 
3.knn 
4.decision tree
5.Random forest
6.Bagging
7.Elastic Net
8.Gradient Boosting
9.Histogram-Based Gradient Boosting Regressor
10.Gradient Boosting With xgboost Regressor 
11.Gradient Boosting With Lightbgm Regressor

### Decision Tree gave the best results 



https://user-images.githubusercontent.com/70663378/139467059-b8290b77-829f-41c4-9572-99268cfa418f.mp4


