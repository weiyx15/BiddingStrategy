# -*- coding: utf-8 -*-
"""
Author: weiyx15

random sample generators from generator_offers2017.csv 
to display the typical generator offers curve
input file:  generator_offers2017.csv
output file: typical_step_function_offers.png
             typical_curve_offers.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' parameter definition '''

inputfile = 'GenerationOffers\generator_offers2017.csv'

outputpng1 = 'plot\typical_step_function_offers.png'
outputpng2 = 'plot\\typical_curve_offers.png'

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

df_t_len = len(df_true)         # number of items in df_true

df_f_len = len(df_false)        # number of items in df_falses

rand_true = np.random.randint(0, df_t_len)  # random index in df_true

rand_false = np.random.randint(0, df_f_len) # random index in df_false

xt = dataframe.ix[rand_true, 'mw1':'mw10'].dropna()  # MWs of a curve

yt = dataframe.ix[rand_true, 'bid1':'bid10'].dropna()   # BIDs of a curve

plt.plot(xt, yt)

plt.savefig(outputpng2)