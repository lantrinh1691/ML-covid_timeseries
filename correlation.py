import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sn
import random

US = pd.read_csv('Data_covid_US.csv', encoding="unicode escape")
US = US.set_index("State")
print(US.head())

human = ['Total Cases', 'Total Death', 'Population', 'Male', 'Female' ]
social_welfare = ['Total Cases', 'Total Death','Number of CoCs', 'Homeless', 'Health', "Day lockdown" ] 
econ = ['Total Cases', 'Total Death','GDPs', '% of USA', 'GDP Growth in 2018', 'Day lockdown', "GDP Rank\xa0" ]

def matrix(attribute):
    scatter_matrix(US[attribute], figsize=(12,8))
    plt.tight_layout()
    plt.show()
    
    sn.heatmap(US[attribute].corr(), annot = True)
    plt.show

matrix(social_welfare)
matrix(human)
matrix(econ)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import numpy as np


def poly_reg(X,y):
    print("POLYNOMIAL PREDICTION")
    lin_reg = LinearRegression()
    poly = PolynomialFeatures(degree = 2)
    #splitting the data
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,y, random_state=1)
    train_poly, test_poly = poly.fit_transform(X_train), poly.fit_transform(X_test)

    lin_reg.fit(train_poly,Y_train)
    print('Intercept:\n',lin_reg.intercept_)
    print("Coefficient:\n",lin_reg.coef_)
    
    predict_test_linreg = lin_reg.predict(test_poly)
    print("RMSE:",np.sqrt(metrics.mean_squared_error(Y_test, predict_test_linreg)))
    
    #Choosing a random value from the test set to calculate.
    value = random.choice(X_test.index)
    print(value)
    
    input_var = X_test.loc[value,:]
    
    test_data=poly.fit_transform(np.array(input_var).reshape(-1,1))
    predict = lin_reg.predict(test_data)

    return (f"PREDICTION:\n{predict}")

def linear_reg(X,y):
    print("LINEAR REGRESSION")
    lin_reg = LinearRegression()
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,y, random_state=1)
    
    lin_reg.fit(X_train,Y_train)
    print('Intercept:\n',lin_reg.intercept_)
    
    print("Coefficient:\n", lin_reg.coef_)
    for i in lin_reg.coef_:
        print (float(i))
    
    predict_test_linreg = lin_reg.predict(X_test)
    print("RMSE:",np.sqrt(metrics.mean_squared_error(Y_test, predict_test_linreg)))
    
    #Choosing a random value from the test set to calculate.
    value = random.choice(X_test.index)
    print(value)
    input_var = X_test.loc[value,:]
    
    predict = lin_reg.predict(np.array(input_var).reshape(1,-1))

    return (f"PREDICTION:\n{predict}") 

print(f"social welfare:\n{linear_reg(US[['Homeless','Health']], US['Total Death'])}\n")

print(f"economic vs covid\n{linear_reg(US[['Population','Total Cases', 'Day lockdown']], US['GDPs'])}\n")    
   
print(f"lockdown\n{linear_reg(US[['Total Cases','Health','GDPs']],US['Day lockdown'])}\n") 

#LINEAR MODEL FOR LOCKDOWN AND HEALTH
train_lreg,valid_lreg = train_test_split(US, test_size=0.2, random_state=1)

lin_reg = LinearRegression(normalize=(True))
lin_reg.fit(np.array(train_lreg["Health"]).reshape(-1,1), np.array(train_lreg["Day lockdown"]).reshape(-1,1))
prediction_valid_linreg = lin_reg.predict(np.array(valid_lreg["Health"]).reshape(-1,1))

plt.figure(figsize=(11,6))
prediction_linreg = lin_reg.predict(np.array(US["Health"]).reshape(-1,1))
linreg_output=[]
for i in range(prediction_linreg.shape[0]):
    linreg_output.append(prediction_linreg[i][0])
    
#calculating the error made from prediction using the testing data    
print("Root MSE for LinearRegression for testing:\n",np.sqrt(metrics.mean_squared_error(valid_lreg["Day lockdown"],prediction_valid_linreg)))   
#calculating the error for prediction of the complete series
print("RMSE for complete prediction\n",np.sqrt(metrics.mean_squared_error(US["Day lockdown"], linreg_output)))
#determining if the prediction line is a good fit
print("R^2",metrics.r2_score(US["Day lockdown"],linreg_output))

US.plot(kind='scatter',
             x="Health",
             y='Day lockdown')
plt.tight_layout()
plt.savefig('scatter-plot.png', dpi=600)
plt.show()

print(f"Intercept: {lin_reg.intercept_}")
print(f"Coefficients: {lin_reg.coef_}\n")

US.plot(kind='scatter', 
        x="Health",
        y='Day lockdown',
        figsize=(5,3))

plt.xlabel("Health index")
Xaxis=np.linspace(0, 50,10)
plt.plot(Xaxis, lin_reg.intercept_ + lin_reg.coef_[0]*Xaxis, "--", color= 'blue')

plt.tight_layout()
plt.savefig('scatter-plot-linear-line.png', dpi=600)
plt.show()

#POLYNOMIAL MODEL FOR LOCKDOWN AND HEALTH
train_preg,valid_preg = train_test_split(US, test_size=0.2, random_state=1)
poly = PolynomialFeatures(degree = 20)

train_poly = poly.fit_transform(np.array(train_preg["Health"]).reshape(-1,1))
valid_poly = poly.fit_transform(np.array(valid_preg["Health"]).reshape(-1,1))
y = train_preg["Day lockdown"]

linreg = LinearRegression(normalize = True)
linreg.fit(train_poly,y)

prediction_poly=linreg.predict(valid_poly)
rmse_poly = np.sqrt(metrics.mean_squared_error(valid_preg["Day lockdown"], prediction_poly))

comp_data = poly.fit_transform(np.array(US["Health"]).reshape(-1,1))
plt.figure(figsize=(11,6))
predictions_poly=linreg.predict(comp_data)

print("Root MSE for PolynomialRegression:\n", rmse_poly)
print("RMSE for complete prediction\n",np.sqrt(metrics.mean_squared_error(US["Day lockdown"], predictions_poly)))
print("R^2",metrics.r2_score(US["Day lockdown"],predictions_poly))

print(f"Intercept: {linreg.intercept_}\n")
#print(f"Coefficients: {linreg.coef_}")

US.plot(kind='scatter', 
        x="Health",
        y='Day lockdown',
        figsize=(5,3))

plt.xlabel("Health index")
plt.plot(predictions_poly, "--", color= 'blue')

plt.tight_layout()
plt.savefig('scatter-plot-polynomial-line.png', dpi=600)
plt.show()

