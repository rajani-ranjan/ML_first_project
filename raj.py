import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import r2_score
from sklearn import linear_model

data = pd.read_csv('D:\\practical\\FuelConsumptionCo2.csv')
# pd.set_option('display.max_columns', None)
# print(data.describe())
'''       MODELYEAR   ENGINESIZE    CYLINDERS  FUELCONSUMPTION_CITY  \
count     1067.0  1067.000000  1067.000000           1067.000000   
mean      2014.0     3.346298     5.794752             13.296532   
std          0.0     1.415895     1.797447              4.101253   
min       2014.0     1.000000     3.000000              4.600000   
25%       2014.0     2.000000     4.000000             10.250000   
50%       2014.0     3.400000     6.000000             12.600000   
75%       2014.0     4.300000     8.000000             15.550000   
max       2014.0     8.400000    12.000000             30.200000   

       FUELCONSUMPTION_HWY  FUELCONSUMPTION_COMB  FUELCONSUMPTION_COMB_MPG  \
count          1067.000000           1067.000000               1067.000000   
mean              9.474602             11.580881                 26.441425   
std               2.794510              3.485595                  7.468702   
min               4.900000              4.700000                 11.000000   
25%               7.500000              9.000000                 21.000000   
50%               8.800000             10.900000                 26.000000   
75%              10.850000             13.350000                 31.000000   
max              20.500000             25.800000                 60.000000   

       CO2EMISSIONS  
count   1067.000000  
mean     256.228679  
std       63.372304  
min      108.000000  
25%      207.000000  
50%      251.000000  
75%      294.000000  
max      488.000000  
'''
s_data = data[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
               'FUELCONSUMPTION_COMB_MPG', 'CO2EMISSIONS']]

# WE HAVE TO BUILD A MODEL TO PREDICT THE VALUE OF CO2EMISSION OF CAR
# For single variable predict model
msk = np.random.rand(len(data)) < 0.8
train = s_data[msk]
test = s_data[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
'''
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
Coefficients:  [[38.36331642]]
Intercept:  [127.91797767]
'''

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

'''
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y, test_y_))
Mean absolute error: 22.28
Residual sum of squares (MSE): 875.02
R2-score: 0.76
'''
# raj = regr.predict([[float(input('inter the engine size: '))]])

# print(raj, 'co2 emission approx ')
train_mx = np.asanyarray(
    train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
           'FUELCONSUMPTION_COMB_MPG']])
train_my = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_mx, train_my)

test_mx = np.asanyarray(
    test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB',
          'FUELCONSUMPTION_COMB_MPG']])
test_my = np.asanyarray(test[['CO2EMISSIONS']])
test_my_ = regr.predict(test_mx)

'''
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_my_ - test_my)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_my_ - test_my) ** 2))
print("R2-score: %.2f" % r2_score(test_my, test_my_))

Mean absolute error: 14.95
Residual sum of squares (MSE): 524.50
R2-score: 0.89
'''

m_raj = regr.predict([[float(input('inter the engine size (in float value ): ')), int(input('inter the Cylinder (in integer value): ')),
                       float(input('inter the FUELCONSUMPTION_CITY (in float value): ')),
                       float(input('inter the FUELCONSUMPTION_HWY (in float value ): ')),
                       float(input('inter the FUELCONSUMPTION_COMB (in float value ): ')),
                       float(input('inter the FUELCONSUMPTION_COMB_MPG (in float value ): '))]])
print(m_raj, 'co2 emission approx ')
