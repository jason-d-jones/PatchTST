import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *

class patchParams():
    pass

class PatchTSTPretrainer(object):
    def __init__(self,
        # Dataset and dataloader
        dset_pretrain:str='etth1', # dataset name
        context_points:int=512, # sequence length
        target_points:int=96, # forecast horizon
        batch_size:int=64, # batch size
        num_workers:int=0, # number of workers for DataLoader
        scaler:str='standard', # scale the input data
        features:str='M', # for multivariate model or univariate model

        # Patch
        patch_len:int=12, # patch length
        stride:int=12, # stride between patch
        
        # RevIN
        revin:int=1, # reversible instance normalization
        
        # Model args
        n_layers:int=3, # number of Transformer layers
        n_heads:int=16, # number of Transformer heads
        d_model:int=128, # Transformer d_model
        d_ff:int=512, # Tranformer MLP dimension 
        dropout:float=0.2, # Transformer dropout
        head_dropout:float=0.2, # head dropout 
        
        # Pretrain mask
        mask_ratio:float=0.4, # masking ratio for the input
        
        # Optimization args
        n_epochs_pretrain:int=10, # number of pre-training epochs
        lr:float=1e-4, # learning rate
        
        # model id to keep track of the number of models saved
        pretrained_model_id:int=1, # id of the saved pretrained model
        model_type:str='based_model' # for multivariate model or univariate model
        ):
        # Dataset and dataloader
        self.dset_pretrain = dset_pretrain
        self.dset = self.dset_pretrain
        self.context_points = context_points
        self.target_points = target_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scaler = scaler
        self.features = features

        # Patch
        self.patch_len = patch_len
        self.stride = stride

        # RevIN
        self.revin = revin

        # Model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.head_dropout = head_dropout

        # Pretrain mask
        self.mask_ratio = mask_ratio

        # Optimization
        self.n_epochs_pretrain = n_epochs_pretrain
        self.lr = lr

        # self.model = model
        self.pretrained_model_id = pretrained_model_id
        self.model_type = model_type

        self.save_pretrained_model = 'patchtst_pretrained_cw'+str(self.context_points)+'_patch'+str(self.patch_len) + '_stride'+str(self.stride) + '_epochs-pretrain' + str(self.n_epochs_pretrain) + '_mask' + str(self.mask_ratio)  + '_model' + str(self.pretrained_model_id)
        self.save_path = 'saved_models/' + self.dset_pretrain + '/masked_patchtst/' + self.model_type + '/'
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)


# get available GPU devide
# set_device()


    def get_model(self, c_in):
        """
        c_in: number of variables
        """
        # get number of patches
        num_patch = (max(self.context_points, self.patch_len)-self.patch_len) // self.stride + 1    
        print('number of patches:', num_patch)
        
        # get model
        model = PatchTST(c_in=c_in,
                    target_dim=self.target_points,
                    patch_len=self.patch_len,
                    stride=self.stride,
                    num_patch=num_patch,
                    n_layers=self.n_layers,
                    n_heads=self.n_heads,
                    d_model=self.d_model,
                    shared_embedding=True,
                    d_ff=self.d_ff,                        
                    dropout=self.dropout,
                    head_dropout=self.head_dropout,
                    act='relu',
                    head_type='pretrain',
                    res_attention=False
                    )        
        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
        return model


    def find_lr(self):
        params = patchParams()
        params.dset = self.dset
        params.context_points = self.context_points
        params.target_points = self.target_points
        params.features = self.features
        # params.use_time_features = self.use_time_features
        params.batch_size = self.batch_size
        params.num_workers = self.num_workers

        # get dataloader
        dls = get_dls(params)    
        model = self.get_model(dls.vars)

        # get loss
        loss_func = torch.nn.MSELoss(reduction='mean')

        # get callbacks
        cbs = [RevInCB(dls.vars, denorm=False)] if self.revin else []
        cbs += [PatchMaskCB(patch_len=self.patch_len, stride=self.stride, mask_ratio=self.mask_ratio)]
            
        # define learner
        learn = Learner(dls, model, 
                            loss_func, 
                            lr=self.lr, 
                            cbs=cbs,
                            )       

        # fit the data to the model
        suggested_lr = learn.lr_finder()
        print('suggested_lr', suggested_lr)
        return suggested_lr


    def pretrain_func(self, lr=None):

        if lr is None:
            lr = self.lr

        params = patchParams()
        params.dset = self.dset
        params.context_points = self.context_points
        params.target_points = self.target_points
        params.features = self.features
        params.batch_size = self.batch_size
        params.num_workers = self.num_workers

        # get dataloader
        dls = get_dls(params)    

        # get model     
        model = self.get_model(dls.vars)
        # get loss
        loss_func = torch.nn.MSELoss(reduction='mean')
        # get callbacks
        cbs = [RevInCB(dls.vars, denorm=False)] if self.revin else []
        cbs += [
            PatchMaskCB(patch_len=self.patch_len, stride=self.stride, mask_ratio=self.mask_ratio),
            SaveModelCB(monitor='valid_loss', fname=self.save_pretrained_model,                       
                            path=self.save_path)
            ]
        # define learner
        learn = Learner(dls, model, 
                            loss_func, 
                            lr=lr, 
                            cbs=cbs,
                            #metrics=[mse]
                            )                        
        # fit the data to the model
        learn.fit_one_cycle(n_epochs=self.n_epochs_pretrain, lr_max=lr)

        train_loss = learn.recorder['train_loss']
        valid_loss = learn.recorder['valid_loss']
        df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
        df.to_csv(self.save_path + self.save_pretrained_model + '_losses.csv', float_format='%.6f', index=False)


if __name__ == '__main__':
    
    pretrainer = PatchTSTPretrainer()
    suggested_lr = pretrainer.find_lr()
    # Pretrain
    pretrainer.pretrain_func(suggested_lr)
    print('pretraining completed')
    

