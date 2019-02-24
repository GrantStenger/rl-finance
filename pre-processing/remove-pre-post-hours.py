# Import Dependencies
import pandas as pd
import numpy as np

def is_valid_time(s):
    h = int(s[:2])
    m = int(s[3:])
    if h > 9 and h < 16:
        return True
    elif h == 9 and m >= 30:
        return True
    elif h == 16 and m == 0:
        return True
    else:
        return False

def remove_pre_post_hours():
    df = pd.read_csv("../data/AAPL.csv")
    df = df.drop(['High', 'Low', 'Close'], axis=1)
    df = df[df['Time'].map(str).map(is_valid_time)]
    df.to_csv("../data/AAPL-Accurate-Hours.csv", index=False)

def main():
    remove_pre_post_hours()

if __name__ == "__main__":
    main()
