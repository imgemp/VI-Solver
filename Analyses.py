__author__ = 'clemens'
import numpy as np
import os
import re


def load_data():
    data = {}
    direct = '../'
    for res in os.listdir(direct):
        if res.endswith('.npy'):
            data[res.split('.')[0]] = np.load(direct+res)
            # if 'deficient' in res or 'mp' in res:
            #     print 'hi'
            #     data[res.split('.')[0]] *= .5
            #     data[res.split('.')[0]] += .5
            # print data[res.split('.')[0]].shape
    return data


def do_analysis():
    data = load_data()
    averages ={}
    for key in data:
        averages_tmp = np.zeros((3,200))
        for i in range(3):
            for j in range(200):
                tmpdat = (np.array(data[key]) + data[key].min())/(data[key].max()-data[key].min())
                # tmpdat = np.array(data[key])
                averages_tmp[i, j] = np.mean(tmpdat[i, j, :100, 1])
        averages[key] = averages_tmp
        print '\n -~-~-~-~-~-~-~-~-~-~-'
        print key
        print 'Max values:      ', averages_tmp.max(axis=1)
        print 'Min values:      ', averages_tmp.min(axis=1)
        print 'Average values:  ', averages_tmp.mean(axis=1)