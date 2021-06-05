import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler
import pandas.tseries.offsets as offsets
import datetime as dt
import logging
import os
import boto3

def input_data(seq, ws):
    out = []
    L = len(seq)

    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
    
    return out

class Model(nn.Module):


    def __init__(self, input=1, h=50, output=1):
        super().__init__()
        self.hidden_size = h

        self.lstm = nn.LSTM(input, h)
        self.fc = nn.Linear(h, output)

        self.hidden = (
            torch.zeros(1, 1, h),
            torch.zeros(1, 1, h)
        )
    

    def forward(self, seq):

        out, _ = self.lstm(
            seq.view(len(seq), 1, -1),
            self.hidden
        )

        out = self.fc(
            out.view(len(seq), -1)
        )

        return out[-1]



symbols_names = {
    '7201.JP': '日産',
    '4755.JP': '楽天',
    '3197.JP': 'すかいらーくホールディングス',
    # '8905.JP': 'イオンモール',
    # '8591.JP': 'オリックス',
    # '2792.JP': 'ハニーズホールディングス',
    # '9432.JP': '日本電信電話株式会社',
    # '3086.JP': 'J・フロント・リテイリング',
    # '6098.JP': 'リクルートホールディングス',
    # '8233.JP': '高島屋',
    # '5020.JP': 'ENEOS',
    # '8306.JP': '三菱 UFJ フィナンシャルグループ',
    # '9433.JP': 'KDDI',
    # '9005.JP': '東急',
    # '9001.JP': '東武',
    # '9024.JP': '西武',
    # '9434.JP': 'Softbank',
    # '4568.JP': '第一三共 Company',
    # '6367.JP': 'ダイキン Industries',
    # '3329.JP': '東和フード',
    # '9831.JP': 'ヤマダ電機',
    # '8316.JP': '三井住友 Financial Group',
    # '7182.JP': 'ゆうちょ銀行',
    # '8058.JP': '三菱',
    # '6273.JP': 'SMC',
    # '6178.JP': '日本郵政',
    # '2914.JP': 'JT',
    # '8411.JP': 'みずほ Financial Group',
    # '3382.JP': 'セブン & アイ Holdings ',
    # '8031.JP': 'Mitsui',
    # '6503.JP': '三菱電機',
    # '6752.JP': 'パナソニック',
    # '4901.JP': '富士フィルム Holdings',
    # '6702.JP': '富士通'
}

stock_data = {}
error_symbols = []

y = {}
scaler = {}

train_window_size = 11

test_size = 30
extending_seq = {}

epochs = 20

predicted_normalized_labels_list = {}
predicted_normalized_labels_array_1d = {}
predicted_normalized_labels_array_2d = {}
predicted_labels_array_2d = {}

real_last_date_timestamp = {}

future_first_date_timestamp =  {}
future_first_date_series_object = {}
future_first_date_str = {}

future_last_date_timestamp =  {}
future_last_date_series_object = {}
future_last_date_str = {}

furture_period = {}

plot_start_date_timestamp = {}

rates_symbols = {}

for j, key in enumerate(symbols_names):
    logging.critical(j)
    
    try:
        stock_data[j] = data.DataReader(key,'stooq').sort_values('Date', ascending=True)
        stock_data[j] = stock_data[j].drop(['Open', 'High', 'Low', 'Volume'], axis=1)
        y[j] = stock_data[j]['Close'].values

        scaler[j] = MinMaxScaler(feature_range=(-1, 1))
        scaler[j].fit(y[j].reshape(-1, 1))
        y[j] = scaler[j].transform(y[j].reshape(-1, 1))    

        y[j] = torch.FloatTensor(y[j]).view(-1)



        
        train_data = input_data(y[j], train_window_size)

        torch.manual_seed(123)
        model = Model()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


        for epoch in range(epochs):


            for train_window, correct_label in train_data:

                optimizer.zero_grad()

                model.hidden = (
                    torch.zeros(1, 1, model.hidden_size),
                    torch.zeros(1, 1, model.hidden_size)
                )

                train_predicted_label = model.forward(train_window)
                train_loss = criterion(train_predicted_label, correct_label)

                train_loss.backward()
                optimizer.step()

            
            extending_seq[j] = y[j][-test_size:].tolist()


            for i in range(test_size):

                test_window = torch.FloatTensor(extending_seq[j][-test_size:])

                with torch.no_grad():

                    model.hidden = (
                        torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size)
                    )  

                    test_predicted_label = model.forward(test_window)
                    extending_seq[j].append(test_predicted_label.item())
            
        predicted_normalized_labels_list[j] = extending_seq[j][-test_size:]
        predicted_normalized_labels_array_1d[j] = np.array(predicted_normalized_labels_list[j])
        predicted_normalized_labels_array_2d[j] = predicted_normalized_labels_array_1d[j].reshape(-1, 1)
        predicted_labels_array_2d[j] = scaler[j].inverse_transform(predicted_normalized_labels_array_2d[j])
        
        
        
        rate_of_change = (predicted_labels_array_2d[j][1].item() - predicted_labels_array_2d[j][0].item()) / predicted_labels_array_2d[j][0].item()

        rates_symbols[rate_of_change] = key
    except:
        error_symbols.append(key)


tupple_list = sorted(rates_symbols.items(), reverse= True)

try:
    logging.critical('sendEmail')
    client = boto3.client('sns')
    
    SNS_TOPIC = os.environ.get('SNS_TOPIC_NAME')
    subject = os.environ.get('SUBJECT')
    msg = '今日以降の株価予測を送付します。\n'
    for i, tupple in enumerate(tupple_list):
        msg += f'{i+1}位：{symbols_names.get(tupple[1])}({tupple[1]})->{tupple[0]}\n'

    msg += '下記の株価予測に失敗しました。'
    for errorsymbol in error_symbols:
        msg += f'{symbols_names.get(errorsymbol)}({errorsymbol})\n'

    response = client.publish(
        TopicArn = SNS_TOPIC,
        Subject= subject,
        Message= msg
        )


except Exception as e:
    logging.error('---can not send message---')
    logging.error(e)