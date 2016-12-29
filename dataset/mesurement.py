import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
def all_day():
    years = [str(i) for i in range(1999, 2017)]
    months = []
    hours = []
    for i in range(1,13):
        tmp = str(i)
        if i < 10:
            tmp = '0' + tmp
        months.append(tmp)
    for i in range(0,24):
        tmp = str(i)
        if i < 10:
            tmp = '0' + tmp
        hours.append(tmp)
    rootdir = 'NREL_SSRL_BMS_SKY_CAM/SSRL_SKY/'
    f = open('1999_2016_hours.csv', 'w')
    for y in years:
        for m in months:
            dir = rootdir + y + '/' + m + '/'
            for parent, dirname, filenames in os.walk(dir):
                dirname.sort()
                for d in dirname:
                    for h in hours:
                        tmp = y +  m + d + h + '00'
                        print 'Tmp:', tmp
                        f.write(tmp + '\n')

    f.close()

def get_data(start_time, end_time, col):
    all_hours = np.loadtxt('1999_2016_hours.csv', delimiter=',', dtype='str')
    data = np.loadtxt('./NREL_SSRL_BMS_IRANDMETE/input_data/ir_data.csv', delimiter=',')
    idx = 0
    start_hour_idx = 0
    start_idx = 0
    while all_hours[idx] < end_time:
        if all_hours[idx] == start_time:
            start_hour_idx = idx
        idx += 1
        if all_hours[idx] < '200801010000':
            start_idx += 1
    end_hour_idx = idx
    # start_hour_idx = start_idx
    # end_hour_idx = start_idx
    print start_hour_idx, end_hour_idx
    ndata = data[(start_hour_idx - start_idx):(end_hour_idx - start_idx), col]
    print all_hours[start_hour_idx:end_hour_idx]
    return ndata, all_hours[start_hour_idx:end_hour_idx]

def ir_season_hour(start_time = '201001010000', end_time = '201201010000', col = 5):
    spring = ['03','04','05']
    summer = ['06','07','08']
    autumn = ['09','10','11']
    winter = ['12','01','02']
    data, hour_id = get_data(start_time, end_time, col)
    print data.shape
    ir_h_spring = [0] * 24
    ir_h_autumn = [0] * 24
    ir_h_summer = [0] * 24
    ir_h_winter = [0] * 24
    num_spring = 1
    num_autumn = 1
    num_summer = 1
    num_winter = 1
    h = 0
    for i in range(data.shape[0]):
        if data[i] != -99999:
            m = (hour_id[i])[4:6]
            print m
            if m in spring:
                ir_h_spring[h] += data[i]
                num_spring += 1
            elif m in autumn:
                ir_h_autumn[h] += data[i]
                num_autumn += 1
            elif m in summer:
                ir_h_summer[h] += data[i]
                num_summer += 1
            else:
                ir_h_winter[h] += data[i]
                num_winter += 1
        h += 1
        if h >= 24:
            h = 0
    ir_h_spring = [i / float(num_spring) for i in ir_h_spring]
    ir_h_summer = [i / float(num_summer) for i in ir_h_summer]
    ir_h_autumn = [i / float(num_autumn) for i in ir_h_autumn]
    ir_h_winter = [i / float(num_winter) for i in ir_h_winter]
    x = range(24)
    print ir_h_autumn
    print ir_h_winter
    print ir_h_summer
    print ir_h_spring
    plt.plot(x, ir_h_spring)
    plt.plot(x, ir_h_summer)
    plt.plot(x, ir_h_winter)
    plt.plot(x, ir_h_autumn)
    plt.show()

def ir_mean(start_time = '201201010000', end_time = '201401010000', col = 5):
    data = get_data(start_time, end_time, col)
    print data.shape
    max_v = np.max(data)
    min_v = np.min(data)
    std_v = np.std(data)
    print max_v, min_v, std_v
    x = range(data.shape[0])
    plt.plot(x, data)
    plt.show()

def feature_importance():
    X = np.loadtxt('NREL_SSRL_BMS_IRANDMETE/input_data/ir_data.csv', delimiter=',')
    y = np.loadtxt('NREL_SSRL_BMS_IRANDMETE/input_data/target_data.csv', delimiter=',')
    X_train, X_test = X[:70000], X[70000:]
    y_train, y_test = y[:70000], y[70000:]
    est = RandomForestRegressor(n_estimators=1000, n_jobs=32)
    est = est.fit(X_train, y_train)
    y_predict = est.predict(X_test)
    for i in range(len(X_test)):
        print abs(y_test - y_predict)
    print est.feature_importances_
    print mean_squared_error(y_test, est.predict(X_test))

if __name__ == '__main__':
    #all_day()
    #ir_season_hour()
    feature_importance()
