

import numpy as np
import pandas as pd
import torch
from torch import nn
import sys
import os

from src.data.datamodule import DataLoaders
from src.data.pred_dataset import *

DSETS = ['ettm1', 'ettm2', 'etth1', 'etth2', 'electricity',
         'traffic', 'illness', 'weather', 'exchange'
        ]

DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../datasets/'

def get_dls(params):
    
    assert params.dset in DSETS, f"Unrecognized dset (`{params.dset}`). Options include: {DSETS}"
    if not hasattr(params,'use_time_features'): params.use_time_features = False

    datasetCls = Dataset_Custom
    size = [params.context_points, 0, params.target_points]

    if params.dset == 'ettm1':
        folder_name = 'ETT-small/'
        data_file_name = 'ETTm1.csv'
        datasetCls = Dataset_ETT_minute

    elif params.dset == 'ettm2':
        folder_name = 'ETT-small/'
        data_file_name = 'ETTm2.csv'
        datasetCls = Dataset_ETT_minute

    elif params.dset == 'etth1':
        folder_name = 'ETT-small/'
        data_file_name = 'ETTh1.csv'
        datasetCls = Dataset_ETT_hour

    elif params.dset == 'etth2':
        folder_name = 'ETT-small/'
        data_file_name = 'ETTh2.csv'
        datasetCls = Dataset_ETT_hour

    elif params.dset == 'electricity':
        folder_name = 'electricity/'
        data_file_name = 'electricity.csv'

    elif params.dset == 'traffic':
        folder_name = 'traffic/'
        data_file_name = 'traffic.csv'
    
    elif params.dset == 'weather':
        folder_name = 'weather/'
        data_file_name = 'weather.csv'

    elif params.dset == 'illness':
        folder_name = 'illness/'
        data_file_name = 'national_illness.csv'

    elif params.dset == 'exchange':
        folder_name = 'exchange_rate/'
        data_file_name = 'exchange_rate.csv'


    root_path = DATA_PATH + folder_name
    
    dls = DataLoaders(
                datasetCls=datasetCls,
                dataset_kwargs={
                'root_path': root_path,
                'data_path': data_file_name,
                'features': params.features,
                'scale': True,
                'size': size,
                'use_time_features': params.use_time_features
                },
                batch_size=params.batch_size,
                workers=params.num_workers,
                )
                
    # dataset is assume to have dimension len x nvars
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls



if __name__ == "__main__":
    class Params:
        dset= 'etth2'
        context_points= 384
        target_points= 96
        batch_size= 64
        num_workers= 8
        with_ray= False
        features='M'
    params = Params 
    dls = get_dls(params)
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
    breakpoint()
