import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import datetime
import random
from itertools import count


class StockDataset(Dataset):

    def __init__(self, csv_file, first_datestr, last_datestr, datestr_format, window_len, dataset_size):

        self.df = pd.read_csv(csv_file)
        self.window_len = window_len
        self.size = dataset_size

        # Dataset as a list of randomly selected dataframes with continuous dates
        self.dataset = []

        # Generate batches of dataframes
        e_datetime = datetime.datetime.strptime(first_datestr, datestr_format).date()
        l_datetime = datetime.datetime.strptime(last_datestr, datestr_format).date()

        for i in range(self.size):
            # Generate a single dataframe
            date_range = []
            while True:
                date_range.clear()
                rdate = e_datetime + (l_datetime - e_datetime) * random.random()  # A random date in the range
                for j in range(100):
                    s = (rdate + datetime.timedelta(days=j)).strftime(datestr_format)
                    if s in self.df["Date"].values:
                        date_range.append(s)
                    if len(date_range) >= 30:
                        break
                # Check if last date in range
                if datetime.datetime.strptime(date_range[-1], datestr_format).date() < l_datetime:
                    break

            self.dataset.append(self.df[self.df["Date"].isin(date_range)])
            print(len(self.dataset))

    def __len__(self):
        # Return the size of the dataset
        return self.size

    def __getitem__(self, idx):
        return self.dataset[idx]



if __name__ == "__main__":
    csv_file = "AAPL-Updated.csv"

    window_len = 30  # Number of trading days in a window
    dataset_size = 100

    earliest_date = "01-02-2014"
    latest_date = "01-18-2019"
    datestr_format = "%m-%d-%Y"

    dataset = StockDataset(csv_file, earliest_date, latest_date, datestr_format, window_len, dataset_size)
