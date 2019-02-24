# Import Dependencies
import pandas as pd

#### Remove pre and post trading hours ####
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

def remove_pre_post_hours(df):
    df = df.drop(['High', 'Low', 'Close'], axis=1)
    df = df[df['Time'].map(str).map(is_valid_time)]
    return df

#### Change slashes to dashes ####
def slash2dash(s):
    s = s.split('/')
    s = "-".join(s)
    return s

def slash_to_dash(df):
    df['Date'] = df['Date'].map(str).map(slash2dash)
    return df

#### Normalize Data ####
def normalize_data(df):
    normalized_data = []
    for i in range(1, len(df)):
        per_change = (df['Open'][i] - df['Open'][i-1]) / (df['Open'][i-1]) * 100
        normalized_data.append(per_change)
    normalized_data = ["N/A"] + normalized_data
    df['Percent-Change'] = normalized_data
    return df

#### Main ####
def main():
    df = pd.read_csv("../data/AAPL.csv")
    df = remove_pre_post_hours(df)
    df = df.reset_index(drop=True)
    df = slash_to_dash(df)
    df = normalize_data(df)
    print(df)

if __name__ == "__main__":
    main()
