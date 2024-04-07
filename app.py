import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import mplfinance as mpf
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob

st.set_page_config(page_title='StockSentiment - Analyzing the Stocks and Twitter Data', page_icon=':bar_chart:', layout='wide')
st.title("StockSentiment - Analyzing the Stocks and Twitter Data")
st.sidebar.title("StockSentiment and Prediction")
st.markdown("Welcome to StockSentiment: In this project, we will be analyzing the stock data and twitter data.📈🐦")
st.sidebar.markdown("This application is a Streamlit dashboard to analyze the sentiment of Tweets related to stocks.📈🐦")

# Load data
@st.cache_data(persist=True)
def load_data(d):
    df = pd.read_csv(d)
    return df

def closing_price(merged_df, company_list, stock_name):
    st.header("Closing Price Visualization")
    plt.figure(figsize = (15, 10))
    plt.subplots_adjust(top = 1.25, bottom = 1.2)
    
    for company in company_list:
        if(stock_name == company):
            plt.scatter(merged_df['Date'], merged_df['Adj Close'])
            plt.ylabel('Adj Close')
            plt.xlabel('Date')
            plt.title(f"Closing Price of {company}")
            
    plt.tight_layout()
    st.pyplot()

def volume_sale(merged_df, company_list, stock_name):
    st.header("Volume of Sales Visualization")
    plt.figure(figsize = (15, 10))

    for company in company_list:
        if(stock_name == company):
            plt.scatter(merged_df['Date'], merged_df['Volume'])
            plt.ylabel('Volume')
            plt.xlabel('Date')
            plt.title(f"Volume of {company}")
            
    plt.tight_layout()
    st.pyplot()

def tweets_per_day(tweets_df):
    st.subheader("No. of Tweets per Day")
    d = {}
    for i in tweets_df["Date"]:
        d[i] = d.get(i, 0) + 1

    keys = d.keys()
    values = d.values()

    plt.figure(figsize = (100,20))
    plt.bar(keys, values)
    plt.xlabel("Keys")
    plt.ylabel("Values")
    plt.title("No. of Tweets per Day",size = 'x-large',color = 'blue')
    plt.xticks(rotation = 90, size = 'small')
    plt.tight_layout()
    plt.show()
    st.pyplot()

def candlestick_plot(yfinance_df):
    st.header("Candlestick Chart of YFinance Data")
    # Plot candlestick chart
    mpf.plot(yfinance_df[:500], type='candle', style = 'charles', volume = True)
    st.pyplot()

def get_sentiment(tweet):
    analysis = TextBlob(str(tweet))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "positive"
    elif polarity < 0:
        return "negative"
    else:
        return "neutral"

d1 = "stock_tweets.csv"
d2 = "stock_yfinance_data.csv"
tweets_df = load_data(d1)
yfinance_df = load_data(d2)

tweets_df["Date"] = tweets_df["Date"].str.split(" ").str[0]

tweets_df['Sentiment'] = tweets_df['Tweet'].apply(get_sentiment)

# Merge data
merged_df = pd.merge(tweets_df, yfinance_df, on = ["Date", "Stock Name"])
merged_df = merged_df.sort_values('Date')

yfinance_df['Date'] = pd.to_datetime(yfinance_df['Date'])
yfinance_df.index = pd.DatetimeIndex(yfinance_df['Date'])

# Displaying Dataset
st.sidebar.markdown("### Dataset Used")
select=st.sidebar.selectbox('Dataset',['Twitter','YFinance'], key = '1')

if not st.sidebar.checkbox("Hide", False, key = '2'):
    if select == "Twitter":
        # Displaying twitter dataset
        st.subheader("Twitter Dataset")
        st.write(tweets_df)

        # Show random Tweet
        st.subheader("Show random Tweet")
        random_tweet=st.radio('Sentiment',('positive','neutral','negative'))
        st.markdown(merged_df.query('Sentiment == @random_tweet')[["Tweet"]].sample(n = 1).iat[0,0])

        # Tweets per day
        tweets_per_day(tweets_df)

        # Number of tweets by sentiment - Histogram, Pie Chart
        st.markdown("### Number of tweets by sentiment")
        select=st.selectbox('Visualization type',['Histogram','Pie Chart'], key = '3')

        sentiment_count = merged_df['Sentiment'].value_counts()
        sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})

        if not st.checkbox("Hide", False, key = '4'):
            if select == "Histogram":
                fig = px.bar(sentiment_count,x = 'Sentiment', y = 'Tweets', color = 'Tweets', height = 500)
                st.plotly_chart(fig)
            else:
                fig = px.pie(sentiment_count, values = 'Tweets', names = 'Sentiment')
                st.plotly_chart(fig)

        # Word Cloud of Tweets Sentiment
        st.subheader("Word Cloud")
        word_sentiment = st.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))

        if not st.checkbox("Hide", False, key = '5'):
            st.subheader('Word Cloud for %s sentiment' % (word_sentiment))
            df = merged_df[merged_df['Sentiment'] == word_sentiment]
            words=' '.join(df['Tweet'])
            processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('0') and word!='RT'])
            wordcloud = WordCloud(stopwords = STOPWORDS, background_color = 'white', height = 640, width = 800).generate(processed_words)
            plt.imshow(wordcloud)
            st.pyplot()

    else:
        # Displaying yfinance dataset
        st.subheader("YFinance Dataset")
        st.write(yfinance_df)

        # Describing yfinance dataset
        st.subheader("Description of YFinance Dataset")
        st.write(yfinance_df.describe())

        company_list=list(set(yfinance_df["Stock Name"]))
        stock_name = st.text_input("Stock Name : ")

        # Closing prize visualization
        closing_price(merged_df, company_list, stock_name)

        # Volume of Sales
        volume_sale(merged_df, company_list, stock_name)

        # Candlestick Plot
        candlestick_plot(yfinance_df)
