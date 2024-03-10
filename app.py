import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.callbacks import EarlyStopping
from keras_self_attention import SeqSelfAttention
from transformers.models.bert.modeling_bert import BertModel
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import mean_squared_error
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("StockSentiment - Analyzing the Stocks and Twitter Data")
st.sidebar.title("StockSentiment and Prediction")
st.markdown("Welcome to StockSentiment: In this project, we will be analyzing the stock data and twitter data.📈🐦")
st.sidebar.markdown("This application is a Streamlit dashboard to analyze the sentiment of Tweets related to stocks.📈🐦")

# Load data
@st.cache_data(persist=True)
def load_data(d):
    df = pd.read_csv(d)
    return df


def get_sentiment(tweet):
    analysis = TextBlob(str(tweet))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

'''# Function to preprocess data
def preprocess_data(data):
    # Convert Date column to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    # Sort data by Date
    data.sort_values(by='Date', inplace=True)
    return data'''

# Function to create LSTM model
def create_lstm_model(input_dim):
    model = Sequential()
    model.add(Embedding(input_dim, 128, input_length=100))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def lstm_model():
    # Train LSTM model
    model = create_lstm_model(len(X_train))

    st.title("Stock Movement Prediction using LSTM")

    # Sidebar
    st.sidebar.header("Model Configuration")
    epochs = st.sidebar.slider("Number of epochs", 1, 20, 10)
    batch_size = st.sidebar.slider("Batch size", 1, 64, 32)

    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)

    # Evaluate model
    loss = model.evaluate(X_test, y_test)
    st.write(f"Test Loss: {loss}")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Plot actual vs predicted prices
    st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()}))

def transformer_model():
    # Create Transformer model
    st.title("Stock Movement Prediction using Transformer")
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    X_train_tokens = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='tf')
    X_test_tokens = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors='tf')

    input_shape = X_train_tokens['input_ids'].shape[1:]

    inputs = Input(shape=input_shape, dtype='int32')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_output = bert_model(inputs)
    flat_output = Flatten()(bert_output.last_hidden_state)
    dense1 = Dense(128, activation='relu')(flat_output)
    dropout = Dropout(0.5)(dense1)
    output = Dense(1, activation='linear')(dropout)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X_train_tokens, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=2)

    # Evaluate model
    loss = model.evaluate(X_test_tokens, y_test)
    st.write(f"Test Loss: {loss}")

    # Make predictions
    predictions = model.predict(X_test_tokens)

    # Plot actual vs predicted prices
    st.line_chart(pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()}))

d1 = "stock_tweets.csv"
d2 = "stock_yfinance_data.csv"
tweets_df = load_data(d1)
yfinance_df = load_data(d2)

tweets_df["Date"] = tweets_df["Date"].str.split(" ").str[0]

tweets_df['Sentiment'] = tweets_df['Tweet'].apply(get_sentiment)

# Merge data
merged_df = pd.merge(tweets_df, yfinance_df, on=["Date", "Stock Name"])
merged_df = merged_df.sort_values('Date')

st.sidebar.markdown("### Dataset Used")
select=st.sidebar.selectbox('Dataset',['Twitter','yFinance'], key='1')

if not st.sidebar.checkbox("Hide",True):
    st.markdown("### Dataset Used")
    if select=="Twitter":
        st.write(tweets_df)
    else:
        st.write(yfinance_df)

st.sidebar.subheader("Show random Tweet")
random_tweet=st.sidebar.radio('Sentiment',('positive','neutral','negative'))
st.sidebar.markdown(merged_df.query('Sentiment == @random_tweet')[["Tweet"]].sample(n=1).iat[0,0])

st.sidebar.markdown("### Number of tweets by sentiment")
select=st.sidebar.selectbox('Visualization type',['Histogram','Pie Chart'], key='1')

sentiment_count=merged_df['Sentiment'].value_counts()
sentiment_count=pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})

if not st.sidebar.checkbox("Hide",True):
    st.markdown("### Number of tweets by sentiment")
    if select=="Histogram":
        fig=px.bar(sentiment_count,x='Sentiment', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(fig)
    else:
        fig=px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)

st.sidebar.header("Word Cloud")
word_sentiment=st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))

if not st.sidebar.checkbox("Hide", True, key='3'):
    st.header('Word Cloud for %s sentiment' % (word_sentiment))
    df=data[data['Sentiment']==word_sentiment]
    words=' '.join(merged_df['Tweet'])
    processed_words=' '.join([word for word in words.split() if 'http' not in word and not word.startswith('0') and word!='RT'])
    wordcloud=WordCloud(stopwords=STOPWORDS, background_color='white', height=640, width=800).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()        

X = []
y = []
for i in range(len(merged_df) - 101):
    X.append(merged_df['Close'].values[i:i+100])
    y.append(merged_df['Close'].values[i+100])

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.sidebar.header("Model")
model_used=st.sidebar.radio('Model to be Used:', ('LSTM', 'Transformer'))

if not st.sidebar.checkbox("Hide", True, key='3'):
    st.header('%s Model' % (model_used))
    if model_used == "LSTM":
        lstm_model()
    else:
        transformer_model()
