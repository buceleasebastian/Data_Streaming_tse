# Big Data, High frequency data analysis: Streaming and Processing Live Financial Data

The aim of this project addresses two main points: streaming live financial data and computing online statistics on these data. The streaming and processing of live financial data is an area of great interest for financial professionals and data scientists. The ability to collect, analyze, and act upon real-time market data can provide nvestors with a competitive advantage, allowing them to make quicker and more informed decisions. The increasing availability of streaming financial data presents new opportunities for data analysis and machine learning applications. 

The ticker choice for this report is Apple Stock price: AAPL. In the first part of the report we will explain how we can update the financial data within a span of time, whereas in the second part we will try to analyze through different statistics the data obtained.

## Streaming data

First of all, we create a dataframe where the start and end date are chosen to initialize the CSV. The dates are "randomly" chosen in the sense that we have to update the data no matter the choice of these dates. In other words, these dates are only a way to start the CSV. The ticker symbol is AAPL, and we start the CSV with January 2023. 

```
# define the stock symbol and time range
symbol = 'AAPL'
start_date = '2023-01-01'
end_date = '2023-02-01'

# retrieve the stock price data
data = yf.download(symbol, start=start_date, end=end_date, interval='1h')

# save the data as a CSV file
data.to_csv(f'{symbol}_prices.csv')
```

Once the CSV is created, the first point which is an important key of the project is how we can update the CSV everytime there is new available data. In the code below we start by obtaining the last available data in `latest_data`, and we compare it with the last row of the CSV already created. If they differ, it means that new data is available and should be added. 

In the second step, we store in `new_data` the data available since the last row of the already existing CSV to the last available data provided by `yfinance`. Finally, we added to the CSV this new available data.

```
while True:
    # retrieve the latest available stock price data
    tz = pytz.timezone('America/New_York')
    now = datetime.datetime.now(tz)
    end_date = Timestamp(now)
    latest_data = yf.download(symbol, period='1d', interval='1h').iloc[-1]

    # check if the latest data is different from the last row in the existing DataFrame
    if not data.empty and not latest_data.equals(data.tail(1).iloc[0]):
        # retrieve any new stock price data since the last update
        new_data = yf.download(symbol, start=data.index[-1] + pd.Timedelta(hours=1), end=end_date, interval='1h')

        # append the new data to the existing DataFrame
        data = pd.concat([data, new_data])

        # save the updated data as a CSV file
        data.to_csv(filename)

    # wait for 1 hour before checking again
    time.sleep(1800)
```



## Processing data
