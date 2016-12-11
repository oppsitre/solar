import numpy as np
import os
start_hour = 200801010000
end_hour = 201607312300
year = range(2008,2017)
path = '/media/lcc/Windows/Downloads/feat/'
def concate_year_data(net_name):
    data = None
    for y in year:
        tmp = np.loadtxt(path + str(y) + '_' + net_name + '.csv', delimiter=',')
        if data is None:
            data = tmp
        else:
            data = np.row_stack((data, tmp))
    np.savetxt(path + net_name + '.csv', data, fmt='%.4f', delimiter=',')

def file_list():
    file = []
    for parent, dirnames, filenames in os.walk(path):
        if len(dirnames) == 0:
            y = parent[-10:-6]
            m = parent[-5:-3]
            d = parent[-2:]
            print y,m,d
            file.append(int(y+m+d))
    sorted(file)
    file = np.array([str(i) for i in file])
    np.savetxt('file_list.csv', file, fmt= '%.8s')

def pad_data(net_name):
    new_data = []
    start_day = '20080101'
    end_day = '20160731'

    day_list = np.loadtxt('file_list.csv', dtype='str')
    data = np.loadtxt(path + net_name + '.csv', delimiter=',',dtype='str')
    date = [str(int(float(i))) for i in data[:,0]]
    feat = np.array(data[:,1:], dtype = 'float')
    feat_size = feat.shape[1]
    print feat
    print date
    print day_list
    idx = 0
    for day in day_list:
        hours = range(0,24)
        for hour in hours[:5]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day + str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            new_data.append([0] * feat_size)
        for hour in hours[5:-4]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day +  str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            print 'Day_Hour:', day_hour
            print 'Idx:', idx
            print date[idx]
            while date[idx] < day_hour:
                idx += 1
            if date[idx] == day_hour:
                new_data.append(feat[idx])
            else:
                f = np.array([-99999]*feat_size)
                new_data.append(f)
        for hour in hours[-4:]:
            if hour < 10:
                day_hour = day + '0' + str(hour) + '00'
            else:
                day_hour = day + str(hour) + '00'
            if day_hour < start_day:
                continue
            if day > end_day:
                break
            new_data.append([0] * feat_size)

    new_data = np.array(new_data)
    print new_data.shape
    np.savetxt('pad_data_' + net_name + '.csv', new_data, fmt='%.4f',delimiter=',')



    #data = np.loadtxt(path + net_name + '.csv', delimiter=',')
    # start_day = 20080101
    # end_day = 20160731
    # hour = 24
    # for y in year:
    #     for month in [1,2,3,4,5,6,7,8,9,10,11,12]:
    #         month_str = str(month)
    #         if month < 10 : month_str = '0' + month_str
    #         print path + str(y) + '/' + str(month) + '/'
    #         for parent, dirnames, filenames in os.walk(path + str(y) + '/' + month_str)):
    #             print ''
    #             # if len(dirnames) == 0:
    #             #
    #             # print 'Index:', idx, 'Month:', month
    #             # print dirnames
    #             # print filenames
    #             # idx += 1

if __name__ == '__main__':
    net_name = 'googlenet'
    pad_data(net_name)