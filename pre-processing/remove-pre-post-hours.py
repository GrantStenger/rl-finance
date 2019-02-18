# Import Dependencies
import pandas as pd
import numpy as np

def main():
    original_df = pd.read_csv("../data/AAPL.csv")
    original_df = original_df.values
    # print(original_df)
    df = np.array(original_df[:,[0,1,2,6]])
    # df['Time'] = original_df.Time
    # df['Open'] = original_df.Open
    # df['Volume'] = original_df.Volume
    print(df)

    # #split 'time' string into hour and minute strings, then convert them into integers and add to an array
    # for i in range(len(df)):
    #     if i % 10 == 0:
    #         print("i status:", i)
    #     print("here1")
    #     (hr,mn) = df.Time[i].split(":")
    #     print("here2")
    #     hr = int(hr)
    #     mn = int(mn)
    #     print("here3")
    #     if hr < 9 or (hr == 9 and mn < 30) or (hr >= 16 and mn > 0) or hr > 16:
    #         df.drop([i],axis=0,inplace=True)
    #         print("here4")
    #     else:
    #         print("here5")
    #
    # print(df.head)

    # df.to_csv("aapl_updated", index=False)

if __name__ == "__main__":
    main()
