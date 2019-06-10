import logging

import pandas as pd


class DataCollectorHistoricalCSV:

    def __init__(self):
        self.logger = logging.getLogger("rnnTrader.DataCollectorHistoricalCSV")
        self.logger.setLevel(logging.INFO)
        self.logger.info("Init DataCollectorHistoricalCSV")

    # example data uses this date format: "%Y-%m-%d %H-%p"
    def load_data(self, file):
        self.logger.info("load CSV-data")
        dataset = f'data/{file}.csv'
        df = pd.read_csv(dataset,
                         names=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume BTC', 'volume USD'])

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %I-%p')
        df.set_index("date", inplace=True)
        df = df.drop("symbol", 1)
        df = df.drop("volume BTC", 1)
        df = df.drop("volume USD", 1)
        print(df.head())
        return df
