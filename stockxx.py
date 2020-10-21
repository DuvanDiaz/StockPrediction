# Import Dependencies

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# App setup

app = dash.Dash()
server = app.server
scaler = MinMaxScaler(feature_range=(0,1))
bynd_df = pd.read_csv("Datasets/BYND.csv")
bynd_df["Date"] = pd.to_datetime(bynd_df.Date, format="%Y-%m-%d")
bynd_df.index = bynd_df['Date']
data = bynd_df.sort_index(ascending=True,axis=0)
new_data = pd.DataFrame(index=range(0,len(bynd_df)), columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data[" Close"][i]=data[" Close"][i]

new_data.index = new_data.Date
new_data.drop("Date",axis=1,inplace=True)
dataset = new_data.values
train = dataset[0:300,:]
valid = dataset[300:,:]
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
x_train,y_train =[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train = np.array(x_train),np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
model = load_model("final_model.h5")
inputs = new_data[len(new_data)-len(valid)-60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []

for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
train = new_data[:300]
valid = new_data[300:]
valid['Predictions'] = closing_price

# Adding Data with the different stocks

shorted = pd.read_csv("Datasets/multiplestock.csv")

app.layout = html.Div([
    html.H1("Price Action Analysis", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label = 'Beyond Meat (BYND) Stock', children = [
            html.Div([
                html.H2("Closing Price", style ={"textAlign": "center"}),
                dcc.Graph(
                    id ="Closing Price",
                    figure ={
                        "data": [
                            go.Scatter(
                                x =train.index,
                                y =valid[" Close"],
                                mode ='markers'
                            )
                        ],
                        "layout": go.Layout(
                            title ='scatter plot',
                            xaxis ={"title": 'Date'},
                            yaxis ={'title': 'Closing Rate'}

                        )
                    }
                ),
                html.H2("LSTM Model Predicted Closing Price", style= {"textAlign": "center"}),
                dcc.Graph(
                    id ="Predicted Closing Price Results",
                    figure ={
                        "data":[
                            go.Scatter(
                                x =valid.index,
                                y =valid["Predictions"],
                                mode ='markers'
                            )
                        ],
                        "layout": go.Layout(
                            title ='scatter plot',
                            xaxis ={"title": 'Date'},
                            yaxis ={'title': 'Closing Rate'}
                        )
                    }
                )
            ])
        ]),
        dcc.Tab(label ='Vanguard Canadian REIT ETF (VNFTF)', children =[
            html.Div([
                html.H1("Vanguard REIT ETF High Vs. Lows (VNFTF)", style ={'textAlign': 'center'}),
                dcc.Dropdown(id ='my-dropdown',
                             options =[{'label': 'Vanguard REIT ETF', 'value': 'VNFTF'},
                                      {'label': 'Tesla','value': 'TSLA'}, 
                                      {'label': 'ROKU', 'value': 'ROKU'}], 
                             multi =True, value =['VNFTF'],
                             style ={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id ='highlow'),
                html.H1("Facebook Market Volume", style ={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options =[{'label': 'Vanguard REIT ETF', 'value': 'VNFTF'},
                                      {'label': 'Tesla','value': 'TSLA'}, 
                                      {'label': 'Roku', 'value': 'ROKU'}], 
                             multi =True, value =['VNFTF'],
                             style ={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id ='volume')
            ], className ="container"),
        ])
    ])
])


# Callback app

@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"VNFTF": "Vanguard REIT ETF", "TSLA": "Tesla","ROKU": "Roku"}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x =shorted[shorted["Symbol"] == stock]["Date"],
                     y =shorted[shorted["Symbol"] == stock][" High"],
                     mode ='lines', opacity=0.7, 
                     name =f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x =shorted[shorted["Symbol"] == stock]["Date"],
                     y =shorted[shorted["Symbol"] == stock][" Low"],
                     mode ='lines', opacity=0.6,
                     name =f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height =600,
            title =f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis ={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis ={"title":"Price (USD)"})}
    return figure
@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])

def update_graph(selected_dropdown_value):
    dropdown = {"VNFTF": "Vanguard REIT ETF", "TSLA": "Tesla", "ROKU": "Roku"}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x =shorted[shorted["Symbol"] == stock]["Date"],
                     y =shorted[shorted["Symbol"] == stock][" Volume"],
                     mode ='lines', opacity =0.7,
                     name =f'Volume {dropdown[stock]}', textposition ='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway =["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height =600,
            title =f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis ={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis ={"title":"Trading Volume"})}
    return figure
if __name__=='__main__':
    app.run_server(debug=True)


