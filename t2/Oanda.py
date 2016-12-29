import requests
import json
import numpy as np
import pandas as pd
import numpy as np

acct_id = '101-004-4736594-001'
api_key = 'b7c97206c23be8e5ac5b1cfa36feb198-4d119b99ac1dcf4be6ca94e13cf26739'

instruments = 'EUR_USD'
granu = 'D'
count = 5000

headers = {'Authorization': 'Bearer %s' % api_key}
url_str = 'https://api-fxpractice.oanda.com/v1/candles'
params = {'instrument' : instruments, 'accountId' : acct_id , 'granularity' : granu, 'count' : count}


response = requests.get(url=url_str, headers=headers, params = params);

data = json.loads(response.text)

Matrix = np.zeros((len(data['candles']), 5))

for i in range(0, len(data['candles'])):
    Matrix[i, 0] = (data['candles'][i]['closeAsk'] + data['candles'][i]['closeBid']) / 2
    Matrix[i, 1] = data['candles'][i]['highAsk']
    Matrix[i, 2] = data['candles'][i]['highBid']
    Matrix[i, 3] = data['candles'][i]['lowAsk']
    Matrix[i, 4] = data['candles'][i]['lowBid']

inputX_df = pd.DataFrame(Matrix[0:3000, 1:5]).pct_change()
inputY_df = pd.DataFrame(np.vstack(Matrix[0:3000, 0])).pct_change()

inputX_df.loc[0] = 0
inputY_df.loc[0] = 0

inputX = np.zeros((3000, 4))
inputY = np.zeros((3000, 1))

inputX = inputX_df.as_matrix()
inputY = inputY_df.as_matrix()

testX_df = pd.DataFrame(Matrix[3001:5000, 1:5]).pct_change()
testY_df = pd.DataFrame(np.vstack(Matrix[3001:5000, 0])).pct_change()

testX_df.loc[0] = 0
testY_df.loc[0] = 0

test_X = testX_df.as_matrix()
test_Y = testY_df.as_matrix()


'''  ABOVE IS TO IMPORT DATA FROM OANDA  '''

import tflearn




net = tflearn.input_data(shape=[None, 4, 3000])
net = tflearn.lstm(net, 128)
net = tflearn.lstm(net, 128)
net = tflearn.fully_connected(net, 1, activation='tanh')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', name="output1")

model = tflearn.DNN(net, tensorboard_verbose=2)
model.fit(inputX, inputY, n_epoch=1, validation_set=0.1, show_metric=False,snapshot_step=100)



