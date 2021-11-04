import pandas as pd
import numpy as np
 
covid = pd.read_csv("covid_19_December.csv")
covid.drop(["SNo"],1,inplace=True)
#converting observation date in to Datetime format
covid["ObservationDate"] = pd.to_datetime(covid['ObservationDate'])

#grouped_country is a dataframe contaning the total number of cases per country per day in term of 
grouped_country = covid.groupby(["Country/Region",'ObservationDate']).agg(
    {"Confirmed":"sum",
     "Recovered":"sum",
     "Deaths":"sum"}
    )

grouped_country["Active"] = grouped_country["Confirmed"] - grouped_country["Recovered"] - grouped_country["Deaths"]

#DATEWISE ANALYSIS
datewise=covid.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
datewise["Days Since"]=datewise.index-datewise.index.min()


datewise["WeekOfYear"] = datewise.index.isocalendar().week
week_num = []
week_confirmed = []
week_recovered = []
week_deaths = []
w=1
for i in list(datewise["WeekOfYear"].unique()):
    week_confirmed.append(datewise[datewise["WeekOfYear"]==i]["Confirmed"].iloc[-1])
    week_recovered.append(datewise[datewise["WeekOfYear"]==i]["Recovered"].iloc[-1])
    week_deaths.append(datewise[datewise["WeekOfYear"]==i]["Deaths"].iloc[-1])
    week_num.append(w)
    w += 1

from plotly.offline import plot
import plotly.graph_objects as go

#STATIC PLOT - interactive, opened on browser


def fig():
    #GRAPH FOR GROWTH WEEKWISE
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=week_num,
        y=week_confirmed,
        mode="lines+markers",
        name="Weekly growth of confirmed cases"))

    fig1.add_trace(go.Scatter(
        x=week_num,
        y=week_recovered,
        mode="lines+markers",
        name="Weekly growth of recovered cases"))

    fig1.add_trace(go.Scatter(
        x=week_num,
        y=week_deaths,
        mode="lines+markers",
        name="Weekly growth of deaths cases"))

    fig1.update_layout(title="Global weekly growth",
                       xaxis_title="Number of week since day 1",
                       yaxis_title="Number of cases",
                       legend=dict(x=0,y=1,traceorder="normal"))
    plot(fig1, filename = "global_growth_weekly.html", auto_open=(True))
    
    #GRAPH FOR GROWTH BY DATE    
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"],
                    mode='lines+markers',
                    name='Confirmed Cases'))
    fig2.add_trace(go.Scatter(x=datewise.index, y=datewise["Recovered"],
                    mode='lines+markers',
                    name='Recovered Cases'))
    fig2.add_trace(go.Scatter(x=datewise.index, y=datewise["Deaths"],
                    mode='lines+markers',
                    name='Death Cases'))
    fig2.update_layout(title="Growth of different types of cases",
                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
    plot(fig2, filename = "global_growth.html", auto_open=(True))
    
    #GRAPH FOR DAILY INCREASE
    fig3=go.Figure()
    fig3.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"].diff().fillna(0),mode='lines+markers',
                              name='Confirmed Cases'))
    fig3.add_trace(go.Scatter(x=datewise.index, y=datewise["Recovered"].diff().fillna(0),mode='lines+markers',
                    name='Recovered Cases'))
    fig3.add_trace(go.Scatter(x=datewise.index, y=datewise["Deaths"].diff().fillna(0),mode='lines+markers',
                    name='Death Cases'))
    fig3.update_layout(title="Daily increase in different types of Cases",
                 xaxis_title="Date",yaxis_title="Number of Cases",legend=dict(x=0,y=1,traceorder="normal"))
    plot(fig3, filename = "global_increase.html", auto_open=(True))

def to_print():
    print (f"size/shape {covid.shape}\n")
    print (f"checking for null values\n {covid.isnull().sum()}\n ")
    print (f"checking for datatype:\n {covid.dtypes}\n")

    print(f"DATEWISE DATAFRAME\n{datewise}\n")
     
    print("BASIC INFORMATION")
    print("Totol number of countries with Disease Spread: ",len(covid["Country/Region"].unique()))
    #the number of cases is cummulative, so using iloc[-1] to access the last row(date) will return the totol value up to that registered date
    print("Total number of Confirmed Cases around the World: ",datewise["Confirmed"].iloc[-1])
    print("Total number of Recovered Cases around the World: ",datewise["Recovered"].iloc[-1])
    print("Total number of Deaths Cases around the World: ",datewise["Deaths"].iloc[-1])
    print("Total number of Active Cases around the World: ",(datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1]))
    print("Total number of Closed Cases around the World: ",datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1])
    #.shape[0] return the number of row/date registered
    print("Approximate number of Confirmed Cases per Day around the World: ",np.round(datewise["Confirmed"].iloc[-1]/datewise.shape[0]))
    print("Approximate number of Recovered Cases per Day around the World: ",np.round(datewise["Recovered"].iloc[-1]/datewise.shape[0]))
    print("Approximate number of Death Cases per Day around the World: ",np.round(datewise["Deaths"].iloc[-1]/datewise.shape[0]))
    
    print("Approximate number of Confirmed Cases per hour around the World: ",np.round(datewise["Confirmed"].iloc[-1]/((datewise.shape[0])*24)))
    print("Approximate number of Recovered Cases per hour around the World: ",np.round(datewise["Recovered"].iloc[-1]/((datewise.shape[0])*24)))
    print("Approximate number of Death Cases per hour around the World: ",np.round(datewise["Deaths"].iloc[-1]/((datewise.shape[0])*24)))
    #using .iloc to access the row, and calculate the difference between the last registered day and the day before
    print("Number of Confirmed Cases in last 24 hours: ",datewise["Confirmed"].iloc[-1]-datewise["Confirmed"].iloc[-2])
    print("Number of Recovered Cases in last 24 hours: ",datewise["Recovered"].iloc[-1]-datewise["Recovered"].iloc[-2])
    print("Number of Death Cases in last 24 hours: ",datewise["Deaths"].iloc[-1]-datewise["Deaths"].iloc[-2])
    print("\n")
    #.diff() calculate the difference between the rows (by default the one before)
    #.fillna(0) fill in the missing value with 0
    #.mean() to find the average
    print("Average increase in number of Confirmed Cases every day: ",np.round(datewise["Confirmed"].diff().fillna(0).mean()))
    print("Average increase in number of Recovered Cases every day: ",np.round(datewise["Recovered"].diff().fillna(0).mean()))
    print("Average increase in number of Deaths Cases every day: ",np.round(datewise["Deaths"].diff().fillna(0).mean()))
    print("\n")
if __name__ == "__main__":
    to_print()
    fig()