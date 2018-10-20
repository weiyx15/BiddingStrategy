# -*- coding: utf-8 -*-
"""
Author: weiyx15

k-means clustering for typical generator curves from generator_offers2017.csv
input file:  generator_offers2017.csv
output file: cluster_step_function_offers_2017.csv
             cluster_curve_offers_2017.csv
             cluster_curve_offers_2017.png
             cluster_step_function_offers_2017.png
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''' parameter definition '''

inputfile = 'GenerationOffers\generator_offers2017.csv'

outputfile1 = 'GenerationOffers\cluster_step_function_offers_2017.csv'
outputfile2 = 'GenerationOffers\cluster_curve_offers_2017.csv'

outputpng1 = 'plot\cluster_step_function_offers_2017.png'
outputpng2 = 'plot\cluster_curve_offers_2017.png'

k = 3               # n_clusters

iteration = 500 #max iteration times

''' data preparatoin '''

headers=['name','tf','mw1','mw2','mw3','mw4','mw5','mw6','mw7','mw8','mw9',\
         'mw10','bid1','bid2','bid3','bid4','bid5','bid6','bid7','bid8',\
         'bid9','bid10']

dataframe = pd.read_csv(inputfile, header=0, names=headers, \
dtype={'name':str, 'tf':str, 'mw1':np.float, 'mw2':np.float, 'mw3':np.float,\
       'mw4':np.float, 'mw5':np.float, 'mw6':np.float, 'mw7':np.float, \
       'mw8':np.float, 'mw9':np.float, 'mw10':np.float, 'bid1':np.float,\
       'bid2':np.float, 'bid3':np.float, 'bid4':np.float, 'bid5':np.float,\
       'bid6':np.float, 'bid7':np.float, 'bid8':np.float, 'bid9':np.float, \
       'bid10':np.float})

# convert data mistakenly labelled as 'TRUE' to 'FALSE'
dataframe.loc[dataframe.isnull().sum(axis=1)==18,'tf'] = 'FALSE'

df_true = dataframe[dataframe['tf'] == 'TRUE']  #curve data

df_false = dataframe[dataframe['tf'] == 'FALSE']    #step function data

''' deal with step function offers '''
# most dimensions data missing
# handcraft-features: max_mw & max_bid 

# function: return max value in an array
def func(arr):
    ret = 0
    for a in arr:
        if a != np.nan:
            ret = max(ret, a)
    return ret

df_false['mw_max'] = df_false.apply(lambda x: func([x.mw1,x.mw2,x.mw3,x.mw4,\
        x.mw5,x.mw6,x.mw7,x.mw8,x.mw9,x.mw10]), axis=1)  # feature 1: max MW

df_false['bid_max'] = df_false.apply(lambda x: func([x.bid1, x.bid2, x.bid3,\
        x.bid4, x.bid5, x.bid6, x.bid7, x.bid8, x.bid9, x.bid10]), axis=1)
    # feature 2: max bid price

data = df_false[['mw_max','bid_max']]     # data for clustering

def Cluster(data, b_curve):
    data_zs = (data - data.mean())/data.std() #data normalization
    
    model = KMeans(n_clusters = k, max_iter = iteration) #k-means model

    model.fit(data_zs) #begin to cluster
    
    df1 = pd.Series(model.labels_).value_counts() #number of elements of each cluster
    
    df2 = pd.DataFrame(model.cluster_centers_) #cluster center
    
    df2.columns = data.columns
    
    df2 = df2 * data.std() + data.mean()
    
    dfcenter = pd.concat([df2, df1], axis = 1) #create DATAFRAME rcenter for cluster centers
    
    dfcenter.columns = list(data.columns) + ['# of elements'] #name new columns
    
    # print k cluster centers
    print(dfcenter)
    
    r = pd.concat([data, pd.Series(model.labels_, index = data.index)], axis = 1)
    
    r.columns = list(data.columns) + ['clusters'] #name new columns
    
    # cluster visualization (using 'ax' making 3 clusters on one plot)
    if b_curve:
        ax = plt.subplot(111, projection='3d')
        colorlist = ['b', 'g', 'orange']
        for i in range(3):
            xx = r[r['clusters']==i]['mw_max']
            yy = r[r['clusters']==i]['k1']
            zz = r[r['clusters']==i]['k0']
            ax.scatter(xx,yy,zz,c=colorlist[i])
            ax.set_xlabel('max MW')
            ax.set_ylabel('k1')
            ax.set_zlabel('k0')
            ax.set_title(str(k)+' clusters of 2017 generator curve offers')
    else:
        ax = r[r['clusters']==0].plot.scatter(x=data.columns[0], \
          y=data.columns[1], color='b')
        ax = r[r['clusters']==1].plot.scatter(x=data.columns[0], \
              y=data.columns[1], color='g', ax=ax)
        ax = r[r['clusters']==2].plot.scatter(x=data.columns[0], \
              y=data.columns[1], color='orange', ax=ax)
        ax.set_title(str(k)+' clusters of 2017 generator step function offers')
    fig = ax.get_figure()
    if b_curve:
        fig.savefig(outputpng2) # save to png
        r.to_csv(outputfile2)   #save to csv file
    else:
        fig.savefig(outputpng1) # save to png
        r.to_csv(outputfile1)   #save to csv file

Cluster(data, False)

''' deal with curve offers '''
# some pieces loss parts of data
# use quadratic interpolation to fill the blank

# row-wise quadratic interpolation
df_true = df_true.ix[:,'mw1':'bid10']   # useful info

def quadreg1(arr):   # given x,y, return quadratic regression coeffient 
    a1 = np.array(arr[0:10])                # MWs
    a2 = np.array(arr[10:20])               # BIDs
    a1 = a1[np.where(~np.isnan(a1))]        # drop nans
    a2 = a2[np.where(~np.isnan(a2))]
    weights = np.polyfit(a1, a2, 1)         # linear fit
    return weights[0]

def quadreg0(arr):   # given x,y, return quadratic regression coeffient 
    a1 = np.array(arr[0:10])                # MWs
    a2 = np.array(arr[10:20])               # BIDs
    a1 = a1[np.where(~np.isnan(a1))]        # drop nans
    a2 = a2[np.where(~np.isnan(a2))]
    weights = np.polyfit(a1, a2, 1)         # linear fit
    return weights[1]

df_true['mw_max'] = df_true.apply(lambda x: func([x.mw1,x.mw2,x.mw3,\
       x.mw4,x.mw5,x.mw6,x.mw7,x.mw8,x.mw9,x.mw10]), axis=1)
df_true['k1'] = df_true.apply(lambda x: quadreg1([x.mw1,x.mw2,x.mw3,\
       x.mw4,x.mw5,x.mw6,x.mw7,x.mw8,x.mw9,x.mw10,x.bid1,x.bid2,x.bid3,\
       x.bid4,x.bid5,x.bid6,x.bid7,x.bid8,x.bid9,x.bid10]), axis=1)
df_true['k0'] = df_true.apply(lambda x: quadreg0([x.mw1,x.mw2,x.mw3,\
       x.mw4,x.mw5,x.mw6,x.mw7,x.mw8,x.mw9,x.mw10,x.bid1,x.bid2,x.bid3,\
       x.bid4,x.bid5,x.bid6,x.bid7,x.bid8,x.bid9,x.bid10]), axis=1)
    # feature: mw_max & k1 & k0 (from BID = k1*MW + k0)

data = df_true[['mw_max','k1','k0']]

Cluster(data, True)