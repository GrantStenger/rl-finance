# Import Dependencies
import pandas as pd

def slash2dash(s):
    s = s.split('/')
    s = "-".join(s)
    return s

def main():
    df = pd.read_csv("../data/AAPL-Updated.csv")
    df['Date'] = df['Date'].map(str).map(slash2dash)
    df.to_csv("../data/AAPL-Updated.csv", index=False)

if __name__ == "__main__":
    main()
