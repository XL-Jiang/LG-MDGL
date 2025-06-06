# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 17:25:00 2022

@author: 86178
"""
from Module import *
from L_MDGL import L_MDGL
from G_MDGL import G_MDGL

#### Local to  Global Multi-scale Fusion and Classification
class LG_MDGL(nn.Module):
    def __init__(self, input_dim1,input_dim2, input_dim3,hidden_dim,topk, phd_dim,num_classes,num_heads, num_time,window_size, kernel_size,cls_token='max', readout='baro'):
        super(LG_MDGL, self).__init__()
        assert cls_token in ['sum', 'mean','max', 'param']
        if cls_token == 'sum':
            self.cls_token = lambda x: x.sum(1)
        elif cls_token == 'mean':
            self.cls_token = lambda x: x.mean(1)
        elif cls_token == 'max':
            self.cls_token = lambda x: x.max(1)[0]
        elif cls_token == 'param':
            self.cls_token = lambda x: x[:,-1,:]
        else:
            raise ValueError("Invalid cls_token type")
        #Local Multi-scale Dynamic Graph Learning
        self.L_MDGL1= L_MDGL(hidden_dim,num_time,window_size, kernel_size)
        self.L_MDGL2 = L_MDGL(hidden_dim, num_time, window_size, kernel_size)
        self.L_MDGL3 = L_MDGL(hidden_dim, num_time, window_size, kernel_size)
        # Global Multi-scale Dynamic Graph Learning
        self.G_MDGL= G_MDGL(input_dim1,input_dim2,input_dim3, hidden_dim,topk, window_size,num_heads, num_time,cls_token=cls_token, readout=readout)
        #Non-imaging Data Processing
        self.mlp = nn.Sequential(nn.BatchNorm1d(phd_dim),nn.Linear(phd_dim, hidden_dim), nn.ReLU(),nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        # Classfier
        self.out = nn.Sequential(nn.BatchNorm1d(hidden_dim*2), nn.Linear(hidden_dim*2, hidden_dim*2*2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(hidden_dim*2*2, num_classes))
        # self.out1 = nn.Sequential(nn.BatchNorm1d(hidden_dim*2), nn.Linear(hidden_dim*2, hidden_dim*2*2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(hidden_dim*2*2, hidden_dim))  # b hidden_dim*2
        self.softmax = nn.Softmax(dim=1)
        self.model_init()
    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, dtimesreies1,dtimesreies2,dtimesreies3,phd_ftrs):
        #b w t n
        minibatch_size, num_windows, num_timepoints,num_nodes1 = dtimesreies1.shape[:4]
        minibatch_size, num_windows, num_timepoints, num_nodes2 = dtimesreies2.shape[:4]
        minibatch_size, num_windows, num_timepoints, num_nodes3 = dtimesreies3.shape[:4]
        #L_MDGL
        dy_local1,dy_adjs1 = self.L_MDGL1(dtimesreies1) #b w n1 t,(b w) n1 n1
        dy_local2, dy_adjs2 = self.L_MDGL2(dtimesreies2)  # b w n2 t,(b w) n2 n2
        dy_local3, dy_adjs3 = self.L_MDGL2(dtimesreies3)  # b w n3 t,(b w) n3 n3
        #G_MDGL
        dy_mambaout,attention = self.G_MDGL(dy_local1,dy_local2,dy_local3,dy_adjs1,dy_adjs2,dy_adjs3) #b hidden_dim
        #process Non-imaging Data
        phd_out = self.mlp(phd_ftrs) #b hidden_dim
        #cat fMRI & phd
        cat = torch.cat((dy_mambaout,phd_out),dim=1)#b hidden_dim*2
        # latent = self.out1(cat)
        # latent=
        # Classfier
        out = self.out(cat)#b 2
        logit = self.softmax(out)
        return logit.squeeze(1),phd_out,attention

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)