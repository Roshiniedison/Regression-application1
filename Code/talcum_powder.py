import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

dataraw = pd.read_csv('tal_powder.csv')

dr=dataraw.dropna()
dataraw.info()
uniq=dr['Variety'].unique()
dn=dr.replace(uniq,np.arange(4))

x=dn.iloc[:,1:-1]
y=dn.iloc[:,-1]

#
#oe = OrdinalEncoder()
#oe.fit(x)
#X = oe.transform(x)
#print(X)


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.1,shuffle=True)

model=LinearRegression()
model.fit(xtrain,ytrain)
model.predict(xtest)

data=['Rose',30,10,50,30,1200,27]

df=pd.DataFrame(data)
data=df.replace(uniq,np.arange(4))
data=np.array(data)
data=data.astype('float32')
data=data.reshape(1,7)
p=model.predict(data)
print('profit:',int(p))

#model = pickle.load(open('model.sav', 'rb'))
#pickle.dump(model, open('model.sav', 'wb'))

print('profit accuracy:',model.score(xtest,ytest))


x1=dn.iloc[:,[1,2,4,5,6,7,8]]
y1=dn.iloc[:,3]


xtrain,xtest,ytrain,ytest=train_test_split(x1,y1,test_size=.1,shuffle=True)

model=LinearRegression()
model.fit(xtrain,ytrain)
model.predict(xtest)


data1=['Rose',30,10,50,30,1200,27]
data1.append(int(p))
del data1[3]

df=pd.DataFrame(data1)
data1=df.replace(uniq,np.arange(4))
data1=np.array(data1)
data1=data1.astype('float32')
data1=data1.reshape(1,7)
p1=model.predict(data1)

print('production',int(p1))

print('production accuracy:',model.score(xtest,ytest))
