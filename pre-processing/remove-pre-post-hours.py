# Import Dependencies
import pandas as pd
import numpy as np

def substring(s):
    return s[:2]

def main():
    df = pd.read_csv("../data/AAPL.csv")
    df = df.drop(['High', 'Low', 'Close'], axis=1)
    df = df[df['Time'].map(str).map(substring).map(int) >= 9]
    df.to_csv("../data/aapl_updated.csv", index=False)

if __name__ == "__main__":
    main()
