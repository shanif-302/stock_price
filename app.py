import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from flask import flask, render_template,request,send_file

import yfinance as yf
import datetime as df
import os
from sklearn.preprocessing import MinMaxScaler

plt.style.use('fivethirtyeight')

app = Flask(__name__)

#load model

model = load_model('price predictions model.h5')

@app.route('/',methods=['GET','POST'])

def index():
    if request.method == 'POST':
        stock =request.form.get('stock')
        if not stock:
            stock = 'POWERGRID.NS' # Default stock
        
        start = dt.datetime(2000,1,1) 
        end = dt.datetime(2024,11,1)
        
        df = yf.download(stock, start = start, end = end)
        
        # describe data
        data_desc = df.describe()
        
        # Exponetial Moving Averages
        ema_20 =df.Close.ewm(span=20,adjust = False).mean()
        ema_50 =df.Close.ewm(span=50,adjust = False).mean()
        ema_100 =df.Close.ewm(span=100,adjust = False).mean()
        ema_200 = df.Close.ewm(span=200,adjust = False).mean()
        
        data_train = pd.DataFrame(df.Close[0: int(len(df)*0.70)])
        data_test = pd.DataFrame(df.Close[ int(len(df)*0.70) : len(df)])

        # data scaling
        scaler = MinMaxScaler(feature_range = (0,1))
        
        # fit transform
        data_train_scale = scaler.fit_transform(data_train)
        
        # prepare data for prediction
        past_100_days = data_train.tail(100)
        final_df = pd.concat([past_100_days,data_test ],ignore_index = True)
        input_data = scaler.fit_transform(final_df)
        
        x_test,y_test =[],[]
        
        for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i-100:i])
                y_test.append(input_data[i,0])
                
        x_test = np.array(x_test)
        y_test  = np.array(y_test)
                
        y_predict = model.predict(x_test)
        
        # inverse scaling
        scal = scaler.scale_
        scale = 1/scal
        y_predict = y_predict*scale
        y_test= y_test*scale
        

        #plot 1 (closing price vs time (20 and 50 ema))

        fig1,ax1 = plt.subplots(figsize =(12,6))

        ax1.plot(df.close,'y',label = 'Closing price',linewidth=1)
        ax1.plot(ema_20,'g',label = 'EMA 20',linewidth=1)
        ax1.plot(ema_50,'r',label = 'EMA 50',linewidth=1)
        ax1.set_title('Closing Price vs Time (20 and 50  Days EMA)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ema_chart_path = "static/ema_20_50.png"
        fig1.savefig(ema_chart_path)
        plt.close(fig1)
        
        # plot 2 (Closing price vs Time (100 and 200 EMA))
        fig2,ax2 = plt.subplots(figsize =(12,6))
        
        ax2.plot(df.Close,'y',label = 'Closing price',linewidth=1)
        ax2.plot(ema_100,'g',label = 'EMA 100',linewidth=1)
        ax2.plot(ema_200,'r',label = 'EMA 200',linewidth=1)
        ax2.set_title('Closing Price vs Time (100 and 200  Days EMA)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Price')
        ax2.legend()
        ema_chart_path_100_200 = "static/ema_100_200.png"
        fig2.savefig(ema_chart_path_100_200)
        plt.close(fig2)
        
        # plot 3 (Prediction vs Orginal Trend)
        fig3,ax3 = plt.subplots(figsize = (12,6))

        ax3.plot(y_test,'y',label = 'Orginal Price ',linewidth = 1)
        ax3.plot(y_predict,'g',label = 'Predicted Price',linewidth =1)
        ax3.set_title('Prediction vs Orginal Trend')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Price')
        ax3.legend()
        prediction_chart_path = "static/Stock_Prediction.png"
        fig3.savefig(prediction_chart_path)
        plt.close(fig3)
        
        # save dataset are csv
        csv_file_path = f'static/{stock}_dataset.csv'
        df.to_csv(csv_file_path)
        
        return render_template('index.html',
                                plot_path_ema_20_50 =ema_chart_path,
                                plot_path_ema_100_200 = ema_chart_path_100_200,
                                plot_path_prediction = prediction_chart_path,
                                data_desc = data_desc.to_html(classes='table table-bordered'),
                                dataset_link=csv_file_path                    
                               )                                                              
       
    return render_template ('index.html')
@app.route('/download/<filename>')
def download_file(filename):
     return send_file(f'static/{filename}',as_attachment=True)
if __name__ == '__main__':
     app.run(debung=True, use_reloader = False)