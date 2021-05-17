import pandas as pd
import glob
import matplotlib.pyplot as plot
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import datetime
from math import ceil
# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

def augFeatures(train):
    train["time"] = pd.to_datetime(train["time"])
    train["month"] = train["time"].dt.month
    train["day"] = train["time"].dt.day
    train["hour"] = train["time"].dt.hour
    train=train.drop("time",axis=1)
    return train
def output_data(path, data):
    import pandas as pd

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return
def getday(y=2017,m=8,d=15,h=0,n=0):
    the_date = datetime.datetime(y,m,d,hour=h,minute=0,second=0)
    result_date = the_date + datetime.timedelta(days=n)
    d = result_date.strftime("%Y-%m-%d %H:%M:%S")
    return d
if __name__ == "__main__":
    args = config()

    data = [["2018-01-01 00:00:00", "buy", 2.5, 3],
            ["2018-01-01 01:00:00", "sell", 3, 5]]
    cons=pd.read_csv(args.consumption)
    gen=pd.read_csv(args.generation)
    train=augFeatures(cons)
    train["generation"]=gen["generation"]
    train=train[["generation","consumption","month","day","hour"]]
    month=train["month"][0]
    day=train["day"][0]
    model=tf.keras.models.load_model("model.h5",compile=False)
    model.compile(loss="mse", optimizer="adam")
    train=train.values.reshape((1,168,5))
    y=model.predict(train)
    index=0
    output=[]
    for i in y[0][:24]:
        temp=[]
        if(i[0]<0.01):
            y[0][index][0]=0
            i[0]=0
        if(i[1]<0):
            y[0][index][1]=0
            i[1]=0
        date=getday(2018,month,day,index,7)
        print(date,"{:.2f} {:.2f}".format(i[0], i[1]))
        action=""
        target_price=0
        target_volume=0
        if(i[0]-i[1] > 0):
            action="sell"
            target_volume=(i[0]-i[1])
            target_price=2.4
        else:
            action="buy"
            target_volume=(abs(i[0]-i[1]))
            target_price=2.2
        temp.append(date)
        temp.append(action)
        temp.append(target_price)
        temp.append(target_volume)
        output.append(temp)
        index+=1
    output=pd.DataFrame(output,columns=['time','action','target_price','target_volume'])
    output_data(args.output,output)