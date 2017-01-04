import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.ensemble import RandomForestRegressor
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

def split_season_data(data, hour_id):
    spring = ['03', '04', '05']
    summer = ['06', '07', '08']
    autumn = ['09', '10', '11']
    winter = ['12', '01', '02']

    # print data.shape
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
        if data[i] != -99999 or data[i] != -11111:
            m = (hour_id[i])[4:6]
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
    return ir_h_spring, ir_h_summer, ir_h_autumn, ir_h_winter
def draw_season_hour(feature_name = 'ABC', col = 5, start_time = '201001010000', end_time = '201201010000'):
    print 'season_hour'
    plt.figure()
    data, hour_id = get_data(start_time, end_time, col)
    spring, summer, autumn, winter = split_season_data(data, hour_id)
    #print 'spring:', np.max(ir_h_spring)
    x = range(24)
    # print ir_h_autumn
    # print ir_h_winter
    # print ir_h_summer
    # print ir_h_spring
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    #plt.sca(ax1)
    plt.plot(x, spring, 'r', label='spring')
    plt.plot(x, summer, 'g', label='summer')
    plt.plot(x, autumn, 'y', label='autumn')
    plt.plot(x, winter, 'b', label='winter')
    #plt.legend((l1,l2,l3,l4),('spring','summer','autumn','winter'))
    plt.legend()
    plt.title(feature_name)
    plt.xlabel('Hours')
    plt.ylabel('Value')
    #plt.sca(ax2)
    #plt.figtext('max/min/mean/std')

    # plt.text(np.argmax(ir_h_spring) - 4, np.max(ir_h_spring), str(format(np.max(ir_h_spring), '.3f')) + '/' + str(
    #     format(np.min(ir_h_spring), '.3f')) + '/' + str(format(np.mean(ir_h_spring), '.3f')) + '/' + str(
    #     format(np.std(ir_h_spring), '.3f')))
    # plt.text(np.argmax(ir_h_summer) - 4, np.max(ir_h_summer), str(format(np.max(ir_h_summer), '.3f')) + '/' + str(
    #     format(np.min(ir_h_summer), '.3f')) + '/' + str(format(np.mean(ir_h_summer), '.3f')) + '/' + str(
    #     format(np.std(ir_h_summer), '.3f')))
    # plt.text(np.argmax(ir_h_autumn) - 4, np.max(ir_h_autumn), str(format(np.max(ir_h_autumn), '.3f')) + '/' + str(
    #     format(np.min(ir_h_autumn), '.3f')) + '/' + str(format(np.mean(ir_h_autumn), '.3f')) + '/' + str(
    #     format(np.std(ir_h_autumn), '.3f')))
    # plt.text(np.argmax(ir_h_winter) - 4, np.max(ir_h_winter), str(format(np.max(ir_h_winter), '.3f')) + '/' + str(
    #     format(np.min(ir_h_winter), '.3f')) + '/' + str(format(np.mean(ir_h_winter), '.3f')) + '/' + str(
    #     format(np.std(ir_h_winter), '.3f')))

    #plt.legend((l1, l2, l3, l4), ('spring', 'summer', 'autumn', 'winter'))
    #plt.show()
    plt.savefig('pic/' + feature_name + '.jpg')

def read_feature_col():
    path = 'mete_feature_col.csv'
    rows = np.loadtxt(path,delimiter=',',dtype='str')
    with open('stat.csv', 'wb') as f:
        for i in rows:
            print i[0]
            lst = statistic_data(feature_name=i[0], col = int(i[1]))
            res = None
            print lst
            for i in lst:
                if res is None:
                    res = str(i)
                else:
                    res = res + ',' + str(i)
            f.write(res + '\n')

def statistic_data(feature_name, col, start_time = '201201010000', end_time = '201401010000'):
    data, hour_id = get_data(start_time, end_time, col)
    spring, summer, autumn, winter = split_season_data(data, hour_id)
    lst = [feature_name]
    lst.extend([np.max(spring), np.min(spring), np.mean(spring), np.std(spring)])
    lst.extend([np.max(summer), np.min(summer), np.mean(summer), np.std(summer)])
    lst.extend([np.max(autumn), np.min(autumn), np.mean(autumn), np.std(autumn)])
    lst.extend([np.max(winter), np.min(winter), np.mean(winter), np.std(winter)])
    return lst

def feature_importance():
    X_train = np.loadtxt('NREL_SSRL_BMS_IRANDMETE/input_data/train/mete_train_data.csv', delimiter=',')
    print 'X_train finished'
    y_train = np.loadtxt('NREL_SSRL_BMS_IRANDMETE/input_data/train/target_train_data.csv', delimiter=',')
    print 'y_train finished'
    X_test = np.loadtxt('NREL_SSRL_BMS_IRANDMETE/input_data/test/mete_test_data.csv', delimiter=',')
    print 'X_test finished'
    y_test = np.loadtxt('NREL_SSRL_BMS_IRANDMETE/input_data/test/target_test_data.csv', delimiter=',')
    print 'y_test finished'
    est = RandomForestRegressor(n_estimators=1000, n_jobs=8)
    est = est.fit(X_train, y_train)
    y_predict = est.predict(X_test)
    for i in range(len(X_test)):
        print abs(y_test - y_predict)
    print est.feature_importances_
    print mean_squared_error(y_test, est.predict(X_test))
if __name__ == '__main__':
    #all_day()
    read_feature_col()
    #draw_season_hour()
    #feature_importance()
