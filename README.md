# Big Data, High frequency data analysis: Streaming and Processing Live Financial Data

<p align="center">
<img src="https://github.com/buceleasebastian/Data_Streaming_tse/blob/main/images/candlestick_chart.svg" width=40% height=40%>
</p>

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
In this second step, there are two main points to bear in mind:
- Online statistics are computed
- This statistics are updated everytime the CSV changes. In other words, these stats are constantly "listening" any change in the CSV.

Computing different statistics on the stock price can provide useful insights into the stock's historical performance, trends, and potential future movements. The online statistics to be computed are the following:
- Mean: general sense of the stock's performance
- The Relative Strength Index (RSI) is a momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions
- Bollinger bands of the stock: technical analysis tool that uses a moving average and two standard deviations to create a range around the stock's price.
- Exponential Moving Average of the stock: type of moving average that gives more weight to recent prices.



```
class Mean :
    n: float = 0
    value : float = 0

    def update(self, x):
        self.n += 1
        self.value += (x - self.value)/self.n
```

```
def compute_RSI(df, n):
    diff = df['Close'].diff()
    gain = diff.where(diff > 0, 0)
    loss = - diff.where(diff < 0, 0)
    avg_gain = gain.rolling(window = 14).mean()
    avg_loss = loss.rolling(window =14).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1+rs))
```

```
def compute_bollinger_bands(df, window, no_std):

    """
    Computes values of Low and High Bollinger bands of the stock, which represent an interval in which the value of the stock is contained

    -------
    Arguments :

    df : dataframe
    window : time window for computing the values
    no_std : number of standard deviations used in calculation
    """


    rolling_mean = df['Close'].rolling(window).mean()
    rolling_std = df['Close'].rolling(window).std()
    df['Bollinger_High'] = rolling_mean + (rolling_std*no_std)
    df['Bollinger_Low'] = rolling_mean + (rolling_std*no_std)
    return df
```

```
class EMA :
    """"
    Dataclass that computes and updates the Exponential Moving Average of the stock

    ----------
    Attributes:

    w : weight assigned to the new value
    ema : value of the current exponential moving average

    ---------
    Methods :
    
    update : updates the value of the exponential moving average
    reset : resets the value of the exponential moving average
    
    """

    w : float
    ema : float = 0

    def update(self, x):
        if self.ema == 0:
            self.ema = x
        else : 
            self.ema = self.w *x + (1-self.w)*self.ema

    def reset(self):
        self.ema = 0
```

Finally, we are also implementing a LSTM (Long Short-Term Memory), which is a type of recurrent neural network (RNN) architecture, that has the ability to capture long-term dependencies in sequential data. LSTMs are useful for predicting stock prices because they are able to model the complex relationships and patterns that exist in sequential financial data (hourly stock prices in our case). They are particularly well-suited to this task because they have the ability to capture long-term dependencies and remember past trends, which is critical in forecasting stock prices. 

```
#LSTM initialization
model = Sequential()

#First layer
model.add(LSTM(units = 50, return_sequences=True, input_shape = (10,1)))

#Second layer
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))

#Third layer with dropout regularization
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))

#Output layer
model.add(Dense(units = 1))

#Compilation
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
```


Here explain the listening part...

```

```





