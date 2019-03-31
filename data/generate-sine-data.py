# Import dependencies
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random

# Create a sine wave dataset in a similar format to our stock data
# We'll use minute data from 9:30am to 4:00pm.
def main():
    minute_data = []
    sines = []
    for date in pd.date_range("01-02-2014", "01-18-2019"):
        new_minutes = list(pd.date_range(str(date.date()) + " 9:30",
                           str(date.date()) + " 16:00", freq="1min")
                          )
        minute_data += new_minutes

        # Create sine data
        x = np.arange(len(new_minutes))
        y = np.sin(2*np.pi * random.randint(1,8) * (x/len(new_minutes)))

        day_sines = y.tolist()
        sines += day_sines

    df = pd.DataFrame()
    df['Times'] = minute_data
    df['Sines'] = sines

    df.to_csv("sin-data.csv", index=False)

if __name__ == "__main__":
    main()
