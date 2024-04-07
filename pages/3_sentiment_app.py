import pandas as pd
import yfinance as yf
import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

st.set_option('deprecation.showPyplotGlobalUse', False)
# Load data
@st.cache_data(persist=True)
def load_data(d):
    df = pd.read_csv(d)
    return df

# Streamlit UI
st.title('Tweet Sentiment Analysis and its Impact on Stocks')

# Input tweet
tweet = st.text_input('Enter a tweet:')
if tweet:
    # Sentiment analysis
    sentiment = TextBlob(tweet).sentiment.polarity
    st.write('Sentiment Score:', sentiment)
    # Impact on stocks
    if sentiment > 0:
        st.write('Positive sentiment might have a positive impact on the stock.')
    elif sentiment < 0:
        st.write('Negative sentiment might have a negative impact on the stock.')
    else:
        st.write('Neutral sentiment may not have a significant impact on the stock.')

d1 = 'stock_yfinance_data.csv'
d2 = 'stock_tweets.csv'

# Load Yahoo Finance dataset
yfinance_df = load_data(d1)

# Load Twitter dataset
twitter_df = load_data(d2)

# Preprocess Twitter dataset and extract sentiment
def get_sentiment(tweet):
    analysis = TextBlob(tweet)
    return analysis.sentiment.polarity

twitter_df['Sentiment'] = twitter_df['Tweet'].apply(get_sentiment)

twitter_df["Date"] = twitter_df["Date"].str.split(" ").str[0]

# Sidebar - Stock selection
st.subheader('Stock Selection')
stocks = yfinance_df['Stock Name'].unique().tolist()
stock = st.selectbox('Select a stock', stocks)

# Merge data
merged_df = pd.merge(twitter_df, yfinance_df, on = ["Date", "Stock Name"])
merged_df = merged_df.sort_values('Date')
data=merged_df[merged_df["Stock Name"] == stock]

# Calculate correlation between sentiment and 'Close' column
correlation, _ = pearsonr(data['Sentiment'], data['Close'])
print("Correlation between sentiment and 'Close' column:", correlation)

# Create MinMaxScaler object
scaler = MinMaxScaler(feature_range=(0, 200))  # Scale to a higher range (0 to 10)

# Reshape the sentiment column for scaling
sentiment_values = data['Sentiment'].values.reshape(-1, 1)

# Fit and transform the sentiment column
scaled_sentiment = scaler.fit_transform(sentiment_values)

# Update the 'Sentiment' column in the merged DataFrame with scaled values
data['Scaled Sentiment'] = scaled_sentiment

# Print the scaled sentiment column
print(data['Scaled Sentiment'])

plt.figure(figsize = (20,4))
plt.plot(data['Close'][::252])
plt.plot(data['Scaled Sentiment'][::252])
#plt.xlabel("Date")
plt.legend(['Close', 'Sentiment'])
plt.title("Sentiment VS Close",size = 'x-large',color = 'blue')
plt.xticks(rotation = 90, size = 'small')
plt.tight_layout()
plt.show()
st.pyplot()
