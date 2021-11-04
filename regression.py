import DataFrame as df
#printing information and graph from the DataFrame module
df.to_print()
df.fig()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from DataFrame import datewise
from plotly.offline import plot
import plotly.graph_objects as go

datewise["Days Since"] = datewise.index - datewise.index[0]
datewise["Days Since"] = datewise["Days Since"].dt.days

#LINEAR REGRESSION FOR CONFIRMED CASE PREDICTION
from sklearn.linear_model import LinearRegression

train_lreg = datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_lreg = datewise.iloc[int(datewise.shape[0]*0.95):]

lin_reg = LinearRegression(normalize=(True))
lin_reg.fit(np.array(train_lreg["Days Since"]).reshape(-1,1), np.array(train_lreg["Deaths"]).reshape(-1,1))
prediction_valid_linreg = lin_reg.predict(np.array(valid_lreg["Days Since"]).reshape(-1,1))

plt.figure(figsize=(11,6))
prediction_linreg = lin_reg.predict(np.array(datewise["Days Since"]).reshape(-1,1))
linreg_output=[]
for i in range(prediction_linreg.shape[0]):
    linreg_output.append(prediction_linreg[i][0])
    
#calculating the error made from prediction using the testing data    
print("Root MSE for LinearRegression for testing:\n",np.sqrt(mean_squared_error(valid_lreg["Deaths"],prediction_valid_linreg)))   
#calculating the error for prediction of the complete series
print("RMSE for complete prediction\n",np.sqrt(mean_squared_error(datewise["Deaths"], linreg_output)))
#determining if the prediction line is a good fit
print("R^2",r2_score(datewise["Deaths"],linreg_output))
print("\n")

#GRAPH FOR LINEAR REGRESSION
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=datewise.index,
    y=datewise["Deaths"],
    mode ="lines+markers",
    name="Train data for confirmed cases"))

fig2.add_trace(go.Scatter(
    x=datewise.index,
    y=linreg_output,
    mode = "lines",
    name="Linear Regression line",
    line = dict(color="black", dash="dot")))

fig2.update_layout(
    title="Confirmed Cases Linear Regression Prediction",
    xaxis_title="Date",
    yaxis_title="Confirmed Cases",
    legend=dict(x=0, y=1, traceorder ="normal"))
plot(fig2, filename = "Linreg_confirmed.html", auto_open=(True))

#POLYNOMIAL REGRESSION MODEL FOR CONFIRMED CASES GLOBAL
from sklearn.preprocessing import PolynomialFeatures

train_preg = datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_preg = datewise.iloc[int(datewise.shape[0]*0.95):]


poly = PolynomialFeatures(degree = 8)

train_poly = poly.fit_transform(np.array(train_preg["Days Since"]).reshape(-1,1))
valid_poly = poly.fit_transform(np.array(valid_preg["Days Since"]).reshape(-1,1))
y = train_preg["Deaths"]

linreg = LinearRegression(normalize = True)
linreg.fit(train_poly,y)

prediction_poly=linreg.predict(valid_poly)
rmse_poly = np.sqrt(mean_squared_error(valid_preg["Deaths"], prediction_poly))

comp_data = poly.fit_transform(np.array(datewise["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
predictions_poly=linreg.predict(comp_data)

print("Root MSE for PolynomialRegression:\n", rmse_poly)
print("RMSE for complete prediction\n",np.sqrt(mean_squared_error(datewise["Deaths"], predictions_poly)))
print("R^2",r2_score(datewise["Deaths"],predictions_poly))
print("\n")

#TIME SERIES PREDICTIONS
from datetime import timedelta,date

#calculating the number of day different from the last registered day and present time 
in_ = date.today()
last = datewise.index.max().date()
i = in_ - last
i = i.days  

new_prediction_poly=[]
#can alter the range depending on how far ahead you want to predict
for n in range (0,5):
    new_date_poly = poly.fit_transform(np.array(datewise["Days Since"].max()+n).reshape(-1,1))
    new_prediction_poly.append(linreg.predict(new_date_poly)[0])

x = 1    
for i in new_prediction_poly:
    print (f"Prediction for day {x}th",i)
    x+=1
print("\n")

#GRAPH FOR POLYNOMIAL REGRESSION + FUTURE PREDICTIONS
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=datewise.index,
    y=datewise["Deaths"],                                                   
    mode ="lines+markers",
    name="Train data for confirmed cases"))

fig3.add_trace(go.Scatter(
    x=datewise.index.append(pd.date_range(datewise.index.max(),in_-timedelta(days=1),freq="d")),
    y= np.concatenate((predictions_poly,np.array(new_prediction_poly))),
    mode = "lines",
    name="Polynomial Regression line",
    line = dict(color="black", dash="dot")))

fig3.update_layout(
    title="Times Serie Polynomial Regression Prediction",
    xaxis_title="Date",
    yaxis_title="Confirmed Cases",
    legend=dict(x=0, y=1, traceorder ="normal"))
plot(fig3, filename = "predict_polreg_confirmed.html", auto_open=True)

#COMPARING DIFFERENT DEGREE OF THE POLYNOMIAL LINEAR REGRESSION MODEL
def compare_degree(deg,datewise):
    rmse_compare = []
    train_preg = datewise.iloc[:int(datewise.shape[0]*0.95)]
    valid_preg = datewise.iloc[int(datewise.shape[0]*0.95):]
    for i in deg:
        deg = []
        poly = PolynomialFeatures(degree = i)
        train_poly = poly.fit_transform(np.array(train_preg["Days Since"]).reshape(-1,1))
        valid_poly = poly.fit_transform(np.array(valid_preg["Days Since"]).reshape(-1,1))
        y = train_preg["Deaths"]

        linreg = LinearRegression(normalize = True)
        linreg.fit(train_poly,y)

        prediction_poly=linreg.predict(valid_poly)
        rmse_test = np.sqrt(mean_squared_error(valid_preg["Deaths"], prediction_poly))
        deg.append(rmse_test)
        
        comp_data = poly.fit_transform(np.array(datewise["Days Since"]).reshape(-1,1))
        predictions_poly=linreg.predict(comp_data)
        rmse_all = np.sqrt(mean_squared_error(datewise["Deaths"], predictions_poly))
        deg.append(rmse_all)
        r_sqrt = r2_score(datewise["Deaths"],predictions_poly)
        deg.append(r_sqrt)
        
        rmse_compare.append(deg)
    return rmse_compare

d = [2,5,8,10]
out = compare_degree(d, datewise)
x = 0
for i in out:
    # the order is [testing rmse, serie rmse, r-squared]
    print(f"Performance Measurement of degree {d[x]}\n",i)
    x += 1

#GRAPH COMPARING POLYNOMIAL AND LINEAR REGRESSION
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=datewise.index,
    y=datewise["Deaths"],                                                   
    mode ="lines+markers",
    name="Train data for confirmed cases"))

fig4.add_trace(go.Scatter(
    x=datewise.index,
    y=linreg_output,
    mode = "lines",
    name="Linear Regression line",
    line = dict(color="black", dash="dot")))

fig4.add_trace(go.Scatter(
    x=datewise.index.append(pd.date_range(datewise.index.max(),in_-timedelta(days=1),freq="d")),
    y= np.concatenate((predictions_poly,np.array(new_prediction_poly))),
    mode = "lines",
    name="Polynomial Regression line",
    line = dict(color="red", dash="dot")))

fig4.update_layout(
    title="Times Serie Regression Prediction Comparison",
    xaxis_title="Date",
    yaxis_title="Confirmed Cases",
    legend=dict(x=0, y=1, traceorder ="normal"))
plot(fig4, filename = "compared_confirmed.html", auto_open=True)
    

    
    
     



