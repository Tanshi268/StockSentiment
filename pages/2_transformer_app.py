import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import Flatten
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

#For reproducability
from numpy.random import seed
seed(1)
tf.random.set_seed(2)

st.set_page_config(page_title='StockSentiment - Analyzing the Stocks and Twitter Data', page_icon=':bar_chart:', layout='wide')

# Function to load the dataset
@st.cache_data(persist=True)
def load_data():
    yfinance_df = pd.read_csv("stock_yfinance_data.csv")
    return yfinance_df

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    
    # Normalization and Attention
    # "EMBEDDING LAYER"
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    
    # "ATTENTION LAYER"
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    
    # FEED FORWARD Part - you can stick anything here or just delete the whole section - it will still work. 
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation = "relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def lr_scheduler(epoch, lr, warmup_epochs=30, decay_epochs=100, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    if epoch <= warmup_epochs:
        pct = epoch / warmup_epochs
        return ((base_lr - initial_lr) * pct) + initial_lr

    if epoch > warmup_epochs and epoch < warmup_epochs+decay_epochs:
        pct = 1 - ((epoch - warmup_epochs) / decay_epochs)
        return ((base_lr - min_lr) * pct) + min_lr

    return min_lr

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    
    for _ in range(num_transformer_blocks):  # This is what stacks our transformer blocks
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="elu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x) #this is a pass-through
    return keras.Model(inputs, outputs)

st.title('Stock Movement Prediction using Transformers')

# Load data
yfinance_df = load_data()

# Sidebar - Stock selection
st.sidebar.header('Stock Selection')
stocks = yfinance_df['Stock Name'].unique().tolist()
st1 = st.sidebar.selectbox('Select a stock', stocks)

stock = yfinance_df.filter(['Close', 'Date', 'Stock Name'])
stock=stock[stock["Stock Name"] == st1]

# Plot raw data
st.subheader(f'Raw Data for {st1}')
st.write(stock)

stock = stock.drop(columns = ['Stock Name'])

target = 'Close'
training_set = stock[:200].values 
test_set = stock[200:].values

test_set_return = stock['Close'][200:].pct_change().values

d1, d2 = stock['Date'][199:200].values[0], stock['Date'][200:201].values[0]

st.subheader(f'Training and Testing Data Visualization')
stock[target][:200].plot(figsize=(16,4),legend=True)
stock[target][200:].plot(figsize=(16,4),legend=True)
plt.legend([f'Training set (Before {d1})',f'Test set ({d2} and beyond)'])
plt.title(f'{st1} stock price')
plt.show()
st.pyplot()

# Scaling the training set - I've tried it without scaling and results are very poor.
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set[:,0:1])

timesteps = 8
# First, we create data sets where each sample has with 8 timesteps and 1 output
# So for each element of training set, we have 8 previous training set elements 
x_train = []
y_train = []
for i in range(timesteps,training_set.shape[0]):
    x_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.LearningRateScheduler(lr_scheduler)
            ]

input_shape = x_train.shape[1:]

model = build_model(
    input_shape,
    head_size=46, # Embedding size for attention
    num_heads=60, # Number of attention heads
    ff_dim=55, # Hidden layer size in feed forward network inside transformer
    num_transformer_blocks=5,
    mlp_units=[256],
    mlp_dropout=0.4,
    dropout=0.14,
)

model.compile(
    loss="mean_squared_error",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["mean_squared_error"],
)
#model.summary()


history = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=20,
    callbacks=callbacks,
)

# First we have to frontload the test data before the inital values of our test_set
# Some functions to help out with
def plot_predictions(test,predicted,symbol):
    plt.figure(figsize=(16,4))
    st.subheader("Actual VS Predicted Values")
    plt.plot(test, color='red',label=f'Real {symbol} Stock Price')
    plt.plot(predicted, color='blue',label=f'Predicted {symbol} Stock Price')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{symbol} Stock Price')
    plt.legend()
    plt.show()
    st.pyplot()

def plot_return_predictions(test,predicted,symbol):
    plt.figure(figsize=(16,4))
    st.subheader("Actual VS Predicted Return Values")
    plt.plot(test, color='red',label=f'Real {symbol} Stock Price Returns')
    plt.plot(predicted, color='blue',label=f'Predicted {symbol} Stock Price Return')
    plt.title(f'{symbol} Stock Return Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{symbol} Stock Price Returns')
    plt.legend()
    plt.show()
    st.pyplot()
    
def return_rmse(test,predicted):
    st.subheader("Evaluation Parameter")
    rmse = math.sqrt(mean_squared_error(test, predicted))
    r2 = r2_score(test, predicted)
    st.write(f"Root Mean Squared Error: {rmse}")
    st.write(f"R2 - Score: {r2}")

def shift(xs, n):
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


dataset_total = pd.concat((stock[target][:200], stock[target][200:]),axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set) - timesteps:].values
inputs = inputs.reshape(-1,1)
inputs  = sc.fit_transform(inputs)

X_test = []
for i in range(timesteps,test_set.shape[0] + timesteps):
    X_test.append(inputs[i-timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


print(test_set[-3],test_set[-2], test_set[-1])
shifted_test_set = shift(test_set, 1) #The shift function is defined early in the notebook
print(shifted_test_set[-3],shifted_test_set[-2], shifted_test_set[-1])

print(predicted_stock_price[-1])
prediction_error = test_set[:,0:1] - predicted_stock_price # This is the error on the same day
#Before we can calculate the predicted return we have to shift the test_set to the day before so we use the shifted_test_set
predicted_return = (shifted_test_set[:,0:1] - predicted_stock_price) / shifted_test_set[:,0:1]

st.subheader("Loss VS Epoch Visualization")
plt.figure(figsize=(16,4))
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'], loc='upper left')
plt.show()
st.pyplot()

plot_predictions(test_set[:,0:1],predicted_stock_price,st1)
return_rmse(test_set[:,0:1],predicted_stock_price)

plot_return_predictions(test_set_return,predicted_return,st1)
return_rmse(test_set_return[1:], predicted_return[1:])

st.subheader("100days and 200days Moving Average")
ma100 = stock.Close.rolling(100).mean()
ma200 = stock.Close.rolling(200).mean()
plt.figure(figsize = (12,6))
plt.plot(stock.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.show()
st.pyplot()
