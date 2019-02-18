# Import Dependencies
import pandas as pd

def main():
    original_df = pd.read_csv("../data/AAPL.csv")
    df = pd.DataFrame(original_df.Date, columns=['Date']).loc[:]
    df['Time'] = original_df.Time
    df['Open'] = original_df.Open
    df['Volume'] = original_df.Volume

    #create placeholer rows to store hour and minute values of each data point
    h = []
    m = []

    #split 'time' string into hour and minute strings, then convert them into integers and add to an array
    for i in range(len(df)):
        (hr,mn) = df.Time[i].split(":")
        hr = int(hr)
        mn = int(mn)
        h.append(hr)
        m.append(mn)

    #add hour and minute columns
    df['Hour'] = h
    df['Minute'] = m

    #check if in market hours, if it's not delete rows
    for j in range(len(df)):
        if j % 10000 == 0:
            print(j)
        if df.Hour[j] < 9 or (df.Hour[j] == 9 and df.Minute[j] < 30) or (df.Hour[j] >= 16 and df.Minute[j] > 0) or df.Hour[j] > 16:
             df.drop([j],axis=0,inplace=True)

    #delete placeholder 'hour' and 'minute' columns
    df.drop(['Hour','Minute'],axis=1,inplace=True)
    print(df.head)

    df.to_csv("aapl_updated", index=False)

if __name__ == "__main__":
    main()
