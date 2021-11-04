import datetime
#calculating the running time of the algorithms
begin = datetime.datetime.now()
from DataFrame import datewise
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

CONFIRMED = datewise["Confirmed"]
DEATHS = datewise["Deaths"]

#FUNCTION SEPARATING INPUT X AND OUTPUT Y
#Using timestep t as input and t+1 as output
def extract_sample(series):
    X = []
    Y = []
    for i in range(len(series)-1):
        X.append(series[i:i+1])
        Y.append(series[i+1])
    return X, Y
 
X_CONFIRMED, Y_CONFIRMED = extract_sample(CONFIRMED)
X_DEATHS, Y_DEATHS = extract_sample(DEATHS)

#FUNCTION TRAINING THE CHOSEN NETWORK AND GRAPH THE RESULT 
def train_test(neural_network, X, Y, epochs=5000, training_percent_size = 0.95):
    N = len(X)
    # Extract training and testing samples and convert in numpy arrays
    n_training = int(N*training_percent_size)
    X_TRAINING, Y_TRAINING = np.array(X[:n_training]), np.array(Y[:n_training])
    X_TESTING, Y_TESTING = np.array(X[n_training:]), np.array(Y[n_training:])
    # neural_network.fit expect a (samples, input-steps, output-steps) shaped X array
    X_TRAINING = X_TRAINING.reshape((X_TRAINING.shape[0], X_TRAINING.shape[1], 1))
    X_TESTING = X_TESTING.reshape((X_TESTING.shape[0], X_TESTING.shape[1], 1))
    # Training: shuffle=False prevents data shuffling, order is important! 
    for i in range(epochs):
        neural_network.fit(X_TRAINING, Y_TRAINING, epochs = 1, verbose = 0, shuffle = False)
        neural_network.reset_states()
    Y_PREDICTED = []    # we list predictions in this array
    for x in X_TESTING:
        x = x.reshape((1, 1, 1))
        y = neural_network.predict(x, verbose = 0)
        Y_PREDICTED.append(y[0][0])
    # Plot data in read and predictions in blue
    plt.plot(Y, 'r')
    plt.plot(range(n_training, N), Y_PREDICTED, 'b')
    plt.xlabel("Days of covid spread") 
    plt.ylabel("Number of cases")
    plt.show()
    # Test loss
    print("RSME =",np.sqrt(mean_squared_error( Y_TESTING, np.array(Y_PREDICTED))))
    print("R-squares =", r2_score(Y_TESTING, np.array(Y_PREDICTED)))
    print("\n")
    # The curve with prediction instead of testing values is returned
    return np.concatenate((Y_TRAINING, Y_PREDICTED))
 
# Set up the network with one layer of 10 nodes
nn = Sequential()
nn.add(LSTM(10, activation = "relu", input_shape = (1, 1)))
nn.add(Dense(1))   # the layer has a 1-dimensional output (a number!)
nn.compile(loss="mean_squared_error", optimizer="adam")
print("one-layer LSTM") 
prediction = train_test(nn, X_DEATHS, Y_DEATHS)

# Set up the network with 2 layers
nn2 = Sequential()
nn2.add(LSTM(10, activation = "relu", input_shape = (1, 1), return_sequences=True))
nn2.add(LSTM(10, activation = "relu"))
nn2.add(Dense(1))  # the layer has a 1-dimensional output (a number!)
nn2.compile(loss="mean_squared_error", optimizer="adam")
print("two-layer LSTM")
prediction = train_test(nn2, X_DEATHS, Y_DEATHS)

def difference(s):
    return [s[i+1] - s[i] for i in range(len(s)-1)]
 
def cumulated(s0, d):
    # cumulated(s[0], difference(s)) == s
    s = [s0]
    for i in range(len(d)):
        s.append(s[i] + d[i])
    return s
 
D_CONFIRMED = difference(CONFIRMED)
D_DEATHS = difference(DEATHS)
 
fig, axes = plt.subplots(2, 2, figsize=(10, 5))
axes[0,0].plot(D_CONFIRMED, "r+", label="CONFIRMED diff")
axes[0,1].plot(D_DEATHS, "k+", label="Deaths diff")
axes[0,0].legend()
axes[0,1].legend()
axes[1,0].hist(D_CONFIRMED)
axes[1,1].hist(D_DEATHS)

plt.figure(0)
DX_DEATHS, DY_DEATHS = extract_sample(D_DEATHS)
d_prediction = train_test(nn2, DX_DEATHS, DY_DEATHS)
 
#PLOTTING THE COMPLETE SERIE
plt.figure(1)
prediction = cumulated(Y_DEATHS[0], d_prediction)
plt.plot(prediction, 'b')
plt.plot(Y_DEATHS, 'r')
print("RSME (dif) =", np.sqrt(mean_squared_error( Y_DEATHS, prediction)))
print("R-squared (dif)", r2_score(Y_DEATHS,prediction))
print("\n")

#PLOTLY GRAPH OF THE OPTIMAL MODEL 
from plotly.offline import plot
import plotly.graph_objects as go

final = go.Figure()
final.add_trace(go.Scatter(
    x=datewise.index,
    y=Y_DEATHS,
    mode ="lines+markers",
    name ="cases growth" 
))
final.add_trace(go.Scatter(
    x=datewise.index,
    y=prediction,
    mode ="lines+markers",
    name ="prediction",
    line = dict(color="red",dash="dot")
))
final.update_layout(
    title="Time Series Prediction with ANN",
    xaxis_title="Date",
    yaxis_title="Number of cases",
    legend=dict(x=0,y=1, traceorder="normal")
)
plot(final, filename = "time_series_ann.html", auto_open=True)


def scaling(s, a = -1, b = 1):
    m = min(s)
    M = max(s)
    scale = (b - a)/(M - m)
    return a + scale * (s - m)
 
DS_CONFIRMED = scaling(difference(CONFIRMED))
DS_DEATHS = scaling(difference(DEATHS))
 
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(DS_CONFIRMED, "r+", label="CONFIRMED diff+scaled")
axes[1].plot(DS_DEATHS, "k+", label="Deaths diff+scaled")
axes[0].legend()
axes[1].legend()

#2 layer network using sigmoid activation function
nn3 = Sequential()
nn3.add(LSTM(10, input_shape = (1, 1), return_sequences=True))
nn3.add(LSTM(10))
nn3.add(Dense(1))
nn3.compile(loss="mean_squared_error", optimizer="adam")
 
plt.figure(0)
DSX_DEATHS, DSY_DEATHS = extract_sample(DS_DEATHS)
ds_prediction = train_test(nn3, DSX_DEATHS, DSY_DEATHS)

#PLOTTING THE COMPLETE SERIE
plt.figure(1)
#undoing the scaling
d_prediction = scaling(ds_prediction, min(D_DEATHS), max(D_DEATHS))
prediction = cumulated(Y_DEATHS[0], d_prediction)
plt.plot(prediction, 'b')
plt.plot(Y_DEATHS, 'r')
print("RSME (dif + scaled) =", np.sqrt(mean_squared_error( Y_DEATHS, prediction)))
print("R-squared (dif + scaled) =\n", r2_score(Y_DEATHS,prediction))
print("\n")

print(datetime.datetime.now() - begin)