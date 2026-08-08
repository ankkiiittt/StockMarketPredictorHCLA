import yfinance as yf
ticker = yf.Ticker("AAPL")
news = ticker.news
print(f"Number of news items: {len(news)}")
if len(news) > 0:
    print(news[0])
