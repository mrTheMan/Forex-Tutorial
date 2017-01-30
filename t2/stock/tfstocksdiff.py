# Copyright 2016 Stephen Smith

import time
import math
import os
from datetime import date
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pandas_datareader as pdr
from pandas_datareader import data, wb
from six.moves import cPickle as pickle
from yahoo_finance import Share

# Choose amount of historical data to use NHistData
NHistData = 30
TrainDataSetSize = 3000

# Load the Dow 30 stocks from Yahoo into a Pandas datasheet

dow30 = ['AXP', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'DD', 'XOM',
         'GE', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM',
         'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG',
         'TRV', 'UNH', 'UTX', 'VZ', 'V', 'WMT', 'DIS']

num_stocks = len(dow30)

trainData = None
loadNew = False

# If stocks.pickle exists then this contains saved stock data, so use this,
# else use the Pandas DataReader to get the stock data and then pickle it.
stock_filename = 'stocks.pickle'
if os.path.exists(stock_filename):
    try:
        with open(stock_filename, 'rb') as f:
            trainData = pickle.load(f)
    except Exception as e:
      print('Unable to process data from', stock_filename, ':', e)
      raise            
    print('%s already present - Skipping requesting/pickling.' % stock_filename)
else:

    # Get the historical data. Make the date range quite a bit bigger than
    # TrainDataSetSize since there are no quotes for weekends and holidays. This
    # ensures we have enough data.
    
    f = pdr.data.DataReader(dow30, 'yahoo', date.today()-timedelta(days=TrainDataSetSize*2+5), date.today())
    cleanData = f.ix['Adj Close']
    trainData = pd.DataFrame(cleanData)
    trainData.fillna(method='backfill', inplace=True)
    loadNew = True
    print('Pickling %s.' % stock_filename)
    try:
        with open(stock_filename, 'wb') as f:
          pickle.dump(trainData, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', stock_filename, ':', e)

# Normalize the data by dividing each price by the first price for a stock.
# This way all the prices start together at 1.
# Remember the normalizing factors so we can go back to real stock prices
# for our final predictions.
factors = np.ndarray(shape=( num_stocks ), dtype=np.float32)
i = 0

for symbol in dow30:
    factors[i] = trainData[symbol][0]
    trainData[symbol] = trainData[symbol]/trainData[symbol][0]
    i = i + 1    

# Configure how much of the data to use for training, testing and validation.

usableData = len(trainData.index) - NHistData + 1
#numTrainData =  int(0.6 * usableData)
#numValidData =  int(0.2 * usableData
#numTestData = usableData - numTrainData - numValidData - 1
numTrainData = usableData - 1
numValidData = 0
numTestData = 0

train_dataset = np.ndarray(shape=(numTrainData - 1, num_stocks * NHistData), dtype=np.float32)
train_labels = np.ndarray(shape=(numTrainData - 1, num_stocks), dtype=np.float32)
valid_dataset = np.ndarray(shape=(max(0, numValidData - 1), num_stocks * NHistData), dtype=np.float32)
valid_labels = np.ndarray(shape=(max(0, numValidData - 1), num_stocks), dtype=np.float32)
test_dataset = np.ndarray(shape=(max(0, numTestData - 1), num_stocks * NHistData), dtype=np.float32)
test_labels = np.ndarray(shape=(max(0, numTestData - 1), num_stocks), dtype=np.float32)
final_row = np.ndarray(shape=(1, num_stocks * NHistData), dtype=np.float32)
final_row_prices = np.ndarray(shape=(1, num_stocks * NHistData), dtype=np.float32)

# Build the taining datasets in the correct format with the matching labels. So if calculate based on last 30 stock prices then the desired
# result is the 31st. So note that the first 29 data points can't be used.
# Rather than use the stock price, use the pricing deltas.

pickle_file = "traindata.pickle"

if loadNew == True or not os.path.exists(pickle_file):
    for i in range(1, numTrainData):
        for j in range(num_stocks):
            for k in range(NHistData):
                train_dataset[i-1][j * NHistData + k] = trainData[dow30[j]][i + k] - trainData[dow30[j]][i + k - 1]
            train_labels[i-1][j] = trainData[dow30[j]][i + NHistData] - trainData[dow30[j]][i + NHistData - 1]     

    for i in range(1, numValidData):
        for j in range(num_stocks):
            for k in range(NHistData):
                valid_dataset[i-1][j * NHistData + k] = trainData[dow30[j]][i + k + numTrainData] - trainData[dow30[j]][i + k + numTrainData - 1]
            valid_labels[i-1][j] = trainData[dow30[j]][i + NHistData + numTrainData] - trainData[dow30[j]][i + NHistData + numTrainData - 1]

    for i in range(1, numTestData):
        for j in range(num_stocks):
            for k in range(NHistData):
                test_dataset[i-1][j * NHistData + k] = trainData[dow30[j]][i + k + numTrainData + numValidData] - trainData[dow30[j]][i + k + numTrainData + numValidData - 1]
            test_labels[i-1][j] = trainData[dow30[j]][i + NHistData + numTrainData + numValidData] - trainData[dow30[j]][i + NHistData + numTrainData + numValidData - 1]

    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise

else:
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']
      del save  # hint to help gc free up memory    

for j in range(num_stocks):
    for k in range(NHistData):
            final_row_prices[0][j * NHistData + k] = trainData[dow30[j]][k + len(trainData.index) - NHistData]
            final_row[0][j * NHistData + k] = trainData[dow30[j]][k + len(trainData.index) - NHistData] - trainData[dow30[j]][k + len(trainData.index) - NHistData - 1]

print('Training set', train_dataset.shape, train_labels.shape)

print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)   

# This accuracy function is used for reporting progress during training, it isn't actually
# used for training.
def accuracy(predictions, labels):
  err = np.sum( np.isclose(predictions, labels, 0.0, 0.005) ) / (predictions.shape[0] * predictions.shape[1])
  return (100.0 * err)

batch_size = 4
num_hidden = 16
num_labels = num_stocks

graph = tf.Graph()

# input is 30 days of dow 30 prices normalized to be between 0 and 1.
# output is 30 values for normalized next day price change of dow stocks
# use a 4 level neural network to compute this.

with graph.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, num_stocks * NHistData))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  tf_final_dataset = tf.constant(final_row)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal(
      [NHistData * num_stocks, num_hidden], stddev=0.05))
  layer1_biases = tf.Variable(tf.zeros([num_hidden]))
  layer2_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_hidden], stddev=0.05))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer3_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_hidden], stddev=0.05))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [num_hidden, num_labels], stddev=0.05))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
    hidden = tf.tanh(tf.matmul(data, layer1_weights) + layer1_biases)
    hidden = tf.tanh(tf.matmul(hidden, layer2_weights) + layer2_biases)
    hidden = tf.tanh(tf.matmul(hidden, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  beta = 0.0
  #loss = (tf.reduce_mean(
  #  tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  #  + tf.nn.l2_loss(layer1_weights)*beta
  #  + tf.nn.l2_loss(layer2_weights)*beta
  #  + tf.nn.l2_loss(layer3_weights)*beta
  #  + tf.nn.l2_loss(layer4_weights)*beta)
  loss = tf.nn.l2_loss( tf.sub(logits, tf_train_labels))
  # loss = tf.reduce_max( tf.abs(tf.sub(logits, tf_train_labels)))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = logits
  valid_prediction = model(tf_valid_dataset)
  test_prediction = model(tf_test_dataset)
  next_prices = model(tf_final_dataset)
  
num_steps = 2052

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    acc = accuracy(predictions, batch_labels)
#    if (acc > 45):    #if sufficiently accurate then stop.
#      print('Minibatch loss at step %d: %f' % (step, l))
#      print('Minibatch accuracy: %.1f%%' % acc)        
#      break;
    if (step % 100 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % acc)
      if numValidData > 0:
          print('Validation accuracy: %.1f%%' % accuracy(
              valid_prediction.eval(), valid_labels))
  if numTestData > 0:        
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

  predictions = next_prices.eval() * factors
  print("Stock    Last Close  Predict Chg   Predict Next      Current     Current Chg       Error")
  i = 0
  for x in dow30:
      yhfeed = Share(x)
      currentPrice = float(yhfeed.get_price())
      print( "%-6s  %9.2f  %9.2f       %9.2f       %9.2f     %9.2f     %9.2f" % (x,
             final_row_prices[0][i * NHistData + NHistData - 1] * factors[i],
             predictions[0][i],
             final_row_prices[0][i * NHistData + NHistData - 1] * factors[i] + predictions[0][i],
             currentPrice,
             currentPrice - final_row_prices[0][i * NHistData + NHistData - 1] * factors[i],
             abs(predictions[0][i] - (currentPrice - final_row_prices[0][i * NHistData + NHistData - 1] * factors[i]))) )
      i = i + 1
  
