import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title='StockSentiment - Analyzing the Stocks and Twitter Data', page_icon=':bar_chart:', layout='wide')

# Load data
@st.cache_data(persist=True)
def load_data():
    yfinance_df = pd.read_csv("stock_yfinance_data.csv")
    return yfinance_df

def main():
    st.title('Stock Movement Prediction using LSTM')
    
    # Load data
    yfinance_df = load_data()

    # Sidebar - Stock selection
    st.sidebar.header('Stock Selection')
    stocks = yfinance_df['Stock Name'].unique().tolist()
    stock = st.sidebar.selectbox('Select a stock', stocks)

    # Create a new dataframe with only the 'Close column 
    data = yfinance_df.filter(['Close', 'Date', 'Stock Name'])

    data=data[data["Stock Name"] == stock] 

    # Convert the dataframe to a numpy array
    dataset = data.values

    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset) * .95 ))

    # Plot raw data
    st.subheader(f'Raw Data for {stock}')
    st.write(data)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset[:, 0:1])

    # Create the training data set 
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()
            
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='relu'))
    model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(x_train, y_train, batch_size=1, epochs=20)

    # Create the testing data set
    # Create a new array containing scaled values from index 1543 to 2002 
    test_data = scaled_data[training_data_len - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
        
    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test[:,0:1], predictions))
    r2 = r2_score(y_test[:,0:1], predictions)
    metrics_eval(r2, rmse)

    # Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Visualize the data
    st.subheader("Stock Movement Visualization")
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close','Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')
    plt.show()
    st.pyplot()

    st.subheader("Loss VS Epoch Visualization")
    plt.figure(figsize=(16,4))
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss'], loc='upper left')
    plt.show()
    st.pyplot()

    # Visualize the data
    st.subheader("Actual VS Prediction")
    plt.figure(figsize=(16,6))
    plt.title('Model')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(valid['Close'])
    plt.plot(valid['Predictions'])
    plt.legend(['Actual', 'Prediction'], loc='upper right')
    plt.show()
    st.pyplot()

    st.subheader("100days and 200days Moving Average")
    ma100 = data.Close.rolling(100).mean()
    ma200 = data.Close.rolling(200).mean()
    plt.figure(figsize = (12,6))
    plt.plot(data.Close)
    plt.plot(ma100, 'r')
    plt.plot(ma200, 'g')
    plt.show()
    st.pyplot()

def metrics_eval(r2, rmse):
    st.subheader("Evaluation Parameter")
    st.write(f"R2 - Score: {r2}")
    st.write(f"Root Mean Squared Error: {rmse}")

if __name__ == '__main__':
    main()
