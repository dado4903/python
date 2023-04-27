import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def non_func(x):
    y = 1.6345 - 0.6235*np.cos(0.6067*x) - 1.3501*np.sin(0.6067*x) - 1.1622 * np.cos(2*x*0.6067) - 0.9443*np.sin(2*x*0.6067)
    return y

def add_noise(y):
    np.random.seed(14)
    varNoise = np.max(y) - np.min(y)
    y_noisy = y + 0.1*varNoise*np.random.normal(0,1,len(y))
    return y_noisy
 
x = np.linspace(1,10,50)
y_true = non_func(x)
y_measured = add_noise(y_true)

x = x[:, np.newaxis]
y_measured = y_measured[:, np.newaxis]

# make polynomial features
poly_low = PolynomialFeatures(degree=2)
poly_med = PolynomialFeatures(degree=6)
poly_high = PolynomialFeatures(degree=15)
xlow = poly_low.fit_transform(x)
xmed = poly_med.fit_transform(x)
xhigh = poly_high.fit_transform(x)
    
np.random.seed(12)
indeksi_high = np.random.permutation(len(xhigh))
indeksi_train_high = indeksi_high[0:int(np.floor(0.7*len(xhigh)))]
indeksi_test_high = indeksi_high[int(np.floor(0.7*len(xhigh)))+1:len(xhigh)]
indeksi_med = np.random.permutation(len(xmed))
indeksi_train_med = indeksi_med[0:int(np.floor(0.7*len(xmed)))]
indeksi_test_med = indeksi_med[int(np.floor(0.7*len(xmed)))+1:len(xmed)]
indeksi_low = np.random.permutation(len(xlow))
indeksi_train_low = indeksi_low[0:int(np.floor(0.7*len(xlow)))]
indeksi_test_low = indeksi_low[int(np.floor(0.7*len(xlow)))+1:len(xlow)]

x_high_train = xhigh[indeksi_train_high,]
y_high_train = y_measured[indeksi_train_high]
x_med_train = xmed[indeksi_train_med,]
y_med_train = y_measured[indeksi_train_med]
x_low_train = xlow[indeksi_train_med,]
y_low_train = y_measured[indeksi_train_med]

xtesthigh = xhigh[indeksi_test_high,]
ytesthigh = y_measured[indeksi_test_high]
xtestmed = xmed[indeksi_test_med,]
ytestmed = y_measured[indeksi_test_med]
xtestlow = xlow[indeksi_test_low,]
ytestlow = y_measured[indeksi_test_low]

linearModelhigh = lm.LinearRegression()
linearModelhigh.fit(x_high_train,y_high_train)
linearModelmed = lm.LinearRegression()
linearModelmed.fit(x_med_train,y_med_train)
linearModellow = lm.LinearRegression()
linearModellow.fit(x_low_train,y_low_train)

yhigh = linearModelhigh.predict(xtesthigh)
MSE_test_high = mean_squared_error(ytesthigh, yhigh)
y_train_high = linearModelhigh.predict(x_high_train)
MSE_train_high = mean_squared_error(y_high_train, y_train_high)
ymed = linearModelmed.predict(xtestmed)
MSE_test_med = mean_squared_error(ytestmed, ymed)
y_train_med = linearModelmed.predict(x_med_train)
MSE_train_med = mean_squared_error(y_med_train, y_train_med)
ylow = linearModellow.predict(xtestlow)
MSE_test_low = mean_squared_error(ytestlow, ylow)
y_train_low = linearModellow.predict(x_low_train)
MSE_train_low = mean_squared_error(y_low_train, y_train_low)

plt.figure(0)
plt.plot(xtesthigh[:,1],yhigh,'ob',label='predicted_high')
plt.plot(xtesthigh[:,1],ytesthigh,'oc',label='test_high')
plt.legend(loc = 4)
plt.show()

plt.figure(1)
plt.plot(xtestmed[:,1],ymed,'or',label='predicted_med')
plt.plot(xtestmed[:,1],ytestmed,'oy',label='test_med')
plt.legend(loc = 4)
plt.show()

plt.figure(2)
plt.plot(xtestlow[:,1],ylow,'og',label='predicted_low')
plt.plot(xtestlow[:,1],ytestlow,'om',label='test_low')
plt.legend(loc = 4)
plt.show()

plt.figure(3)
plt.plot(x,y_true,'k',label='f')
plt.plot(x, linearModelhigh.predict(xhigh),'r-',label='model_high')

plt.plot(x, linearModelmed.predict(xmed),'g:',label='model_med')

plt.plot(x, linearModellow.predict(xlow),'b-.',label='model_low')

plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 4)
plt.show()

print("TESTNA\n")
print ("HIGH: ",MSE_test_high,"\n")
print ("MED: ",MSE_test_med,"\n")
print ("LOW: ",MSE_test_low,"\n")

print("TRAIN\n")
print("LOW: ",MSE_train_low,"\n")
print("MED: ",MSE_train_med,"\n")
print("HIGH: ",MSE_train_high,"\n")