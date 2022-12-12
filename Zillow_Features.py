
import numpy as np
import pandas as pd
import sys
import os
import gc
import random
pd.options.display.max_columns = None
pd.options.mode.chained_assignment = None
pd.options.display.float_format

from sklearn.model_selection import train_test_split

import lightgbm as lgb

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


def clean_and_fixing(data):
    df = data.copy()
    ## pool
    # bad_index = df[df.poolcnt.isnull()].index
    # df.loc[bad_index,'poolcnt']=0
    # if no pool then poolsize equal to 0
    bad_index = df[df.poolcnt == 0].index
    df.loc[bad_index, 'poolsizesum'] = 0
    # if have a pool
    have_pool_index = df[(df['poolcnt'] > 0) & (df['poolsizesum'].isnull())].index
    poolsize = df.groupby(['regionidcity', 'regionidzip'])['poolsizesum'].transform('median')
    df.loc[have_pool_index, 'poolsizesum'] = poolsize

    # unitcnt
    bad_index = df[df.unitcnt.isnull()].index
    df.loc[bad_index, 'unitcnt'] = 1

    # decktypeid
    df['decktypeid'] = np.where(df['decktypeid'].isnull(), 0, 1)

    # fireplacent
    df['fireplacecnt'] = np.where(df['fireplacecnt'].isnull(), 0, 1)

    # hashottuborspa
    df['hashottuborspa'] = np.where(df['hashottuborspa'] == '', 0, 1)

    # taxdelinquencyflag
    df['taxdelinquencyflag'] = np.where(df['taxdelinquencyflag'] == '', 0, 1)
    # airconditiontypeid
    df['airconditioningtypeid'] = np.where((df.airconditioningtypeid.isnull()) & (df.heatingorsystemtypeid == 2), 1,
                                           df.airconditioningtypeid)

    # heatingorsystemtypeid
    bad_index = df[df.heatingorsystemtypeid.isnull()].index
    df.loc[bad_index, 'heatingorsystemtypeid'] = 0

    ##finishedfloor1squarefeet is not supposed to be bigger than total calculatedfinishedsquarefeet
    index = df.loc[df['calculatedfinishedsquarefeet'] < df['finishedfloor1squarefeet']].index
    df.loc[index, 'finishedfloor1squarefeet'] = np.nan

    # garagetotalsqft
    df[(df.garagecarcnt == 0) & (df['garagetotalsqft'] > 0)].index
    df.loc[index, 'garagecarcnt'] = np.nan

    # taxvaluedollarcnt
    bad_index = df[df['taxvaluedollarcnt'].isnull()].index
    df.loc[bad_index, 'taxvaluedollarcnt'] = df['structuretaxvaluedollarcnt'] + df['landtaxvaluedollarcnt']

    # # #structuretaxvaluedollarcnt
    # bad_index = df[df['structuretaxvaluedollarcnt'].isnull()].index
    # df.loc[bad_index,'structuretaxvaluedollarcnt'] = df['taxvaluedollarcnt']  -  df['landtaxvaluedollarcnt']

    # # #landtaxvaluedollarcnt
    # bad_index = df[df['landtaxvaluedollarcnt'].isnull()].index
    # df.loc[bad_index,'landtaxvaluedollarcnt'] = df['taxvaluedollarcnt']  -  df['structuretaxvaluedollarcnt']

    # bad_index = df[df['structuretaxvaluedollarcnt']==0].index
    # df.loc[bad_index,['finishedsquarefeet12','calculatedfinishedsquarefeet','garagetotalsqft','finishedsquarefeet50' ,'lotsizesquarefeet','finishedfloor1squarefeet','yearbuilt']] = np.nan

    return df

