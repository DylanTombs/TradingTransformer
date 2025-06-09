import backtrader as bt
import pandas as pd
from RsiEmaStrategy import RsiEmaStrategy

data = pd.read_csv('AAPL_2020-01-01_2023-01-01.csv')
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)


# Make sure your data columns match what you're passing
datafeed = bt.feeds.PandasData(
    dataname=data,
    datetime=None,  # Assuming index is datetime
    open='open',
    high='high',
    low='low',
    close='close',
    volume='volume',
    openinterest=-1
)


def RunSimulation(strategy):
 
    cerebro = bt.Cerebro()
    cerebro.adddata(datafeed)
    cerebro.addstrategy(strategy)
    cerebro.broker.set_cash(10000)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    cerebro.plot() 

RunSimulation(RsiEmaStrategy)  