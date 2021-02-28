
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 


def data_preprocessing(input_df):
    input_df['Date']=pd.to_datetime(input_df['Date']) 
    input_df = input_df.set_index('Date')
    input_df.sort_values('Date', inplace=True, ascending=True)
    normal = input_df.apply(lambda x:(x-x.min())/(x.max()-x.min()))
    transformed = (input_df.groupby([lambda x: x.year,lambda x: x.quarter]).transform(lambda x: (x-x.min())/(x.max()-x.min())))
    return normal,transformed

def create_dataset(input_df, look_back,prediction):
    X, Y = [], []
    for i in range(len(input_df)-look_back-1):
        X.append(dataset[i:(i+look_back), 0])
        Y.append(dataset[i + look_back:i + look_back + prediction, 0])
    return np.array(X), np.array(Y)

def final_prediction(input_df):
    final =[]
    for i in range(0,len(input_df)-1):
        final.append((input_df[i][1]+input_df[i+1][0])/2)
    final.append(input_df[len(input_df)-1][0])   
    return final

data = pd.read_csv("./data/BTC.csv")

data,quarter = data_preprocessing(data)


dataset = data.Close.values 
d = data.Open.values
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, 1))
X, Y = create_dataset(dataset, look_back = 3, prediction = 1)    
x_train , x_test = X[:int(np.round(len(X)*0.8))],X[int(np.round(len(X)*0.8)):]
y_train , y_test = Y[:int(np.round(len(Y)*0.8))],Y[int(np.round(len(Y)*0.8)):]
# reshape input to be [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

train_mean_absolute_error =[]
train_mean_squared_error =[]
test_mean_absolute_error,test_mean_squared_error =[],[]
for i in range(0,3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(30,return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(25))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')
    history = model.fit(x_train, y_train, epochs=20, batch_size=70, validation_data=(x_test, y_test), 
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)
    model.summary()
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    
    train_mean_absolute_error.append(mean_absolute_error(y_train, train_predict[:,0]))
    train_mean_squared_error.append(mean_squared_error(y_train, train_predict[:,0]))
    test_mean_absolute_error.append(mean_absolute_error(y_test, test_predict[:,0]))
    test_mean_squared_error.append(mean_squared_error(y_test, test_predict[:,0]))

print('Train Mean Absolute Error:', sum(train_mean_absolute_error)/len(train_mean_absolute_error))
print('Train Root Mean Squared Error:',sum(train_mean_squared_error)/len(train_mean_squared_error))

print('Test Mean Absolute Error:', sum(test_mean_absolute_error)/len(test_mean_absolute_error) )

print('Test Root Mean Squared Error:',sum(test_mean_squared_error)/len(test_mean_squared_error))
aa=[x for x in range(200)]
plt.figure(figsize=(8,4))
plt.plot(aa, y_test[:200,0], marker='*', label="actual")
plt.plot(aa, test_predict[:200], 'g', label="prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('price', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();

# net worth changes
temp = pd.read_csv("./data/BTC.csv")
fluc=[]

balance = 10
pos=0
for i in range (1, len(test_predict)):
    if test_predict[i]>test_predict[i-1]:
        # performing buy:
        balance-=y_test[i-1] 
        pos+=1
        networth= pos*y_test[i-1]+ balance
        fluc.append(networth)
    elif test_predict[i]<test_predict[i-1]:
        balance+=y_test[i-1]
        networth= pos*y_test[i-1]+ balance
        fluc.append(networth)
        pos-=1
  
plt.plot(fluc)
