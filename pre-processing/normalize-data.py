# Import Dependencies
import pandas as pd

def main():
    df = pd.read_csv("../data/AAPL-Updated.csv")
    normalized_data = []
    for i in range(1, len(df)):
        per_change = (df['Open'][i] - df['Open'][i-1]) / (df['Open'][i-1]) * 100
        normalized_data.append(per_change)
    normalized_data = ["N/A"] + normalized_data
    df['Percent-Change'] = normalized_data
    df.to_csv("../data/AAPL-Normalized.csv", index=False)

if __name__ == "__main__":
    main()
