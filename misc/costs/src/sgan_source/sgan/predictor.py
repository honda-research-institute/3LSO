#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
function call returns positions predicted by Social GAN

    IN: the multi-dimensional array (obs_len, num_agents, positions)
    OUT: the multi-dimensional array (pred_len, num_agents, positions)

Created on Fri Jun 21 16:52:59 2019

@author: sbae
"""
import torch
import sys, os
import numpy as np
from attrdict import AttrDict
import time

# sys.path.insert(0,'/home/sbae/NNMPC.jl/python/sgan_source/')-- Sangjae: I just noticed it was manually coded from my side...
#sys.path.insert(0,'/home/sbae/NNMPC_CARLA/sgan_source/sgan') --- ADI
# sys.path.insert(0,'/home/workspace/NNMPC_CARLA/sgan_source/sgan') -- Sangjae
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))

# from sgan.data.loader import data_loader
from models import TrajectoryGenerator
from sgan_utils import relative_to_abs
torch.manual_seed(42) 

class Predictor:
    def __init__(self, model_path, pred_len):
        self.model_path = model_path
        self.pred_len = pred_len
        self.generator = self.get_generator(self.model_path)
        

    def get_generator(self, path):
        paths = [path]
        for path in paths:
            checkpoint = torch.load(path, map_location= lambda storage, loc: storage) #checkpoint = torch.load(path)  ADI
            self.args = AttrDict(checkpoint['args'])
            return self.gen(checkpoint)

    def gen(self, checkpoint):
        args = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=args.obs_len,
            pred_len=self.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm)
        generator.load_state_dict(checkpoint['g_state'])
        #generator.cuda()  ADI
        generator.train()
        return generator


    # def predict(self, data):
    #     # data example:
    # #    data = np.array([
    # #       [ 1,  1.000e+00,  8.460e+00,  3.590e+00],
    # #       [ 1,  2.000e+00,  1.364e+01,  5.800e+00],
    # #       [ 2,  1.000e+00,  9.570e+00,  3.790e+00],
    # #       [ 2,  2.000e+00,  1.364e+01,  5.800e+00],
    # #       [ 3,  1.000e+00,  1.067e+01,  3.990e+00],
    # #       [ 3,  2.000e+00,  1.364e+01,  5.800e+00],
    # #       [ 4,  1.000e+00,  1.173e+01,  4.320e+00],
    # #       [ 4,  2.000e+00,  1.209e+01,  5.750e+00],
    # #       [ 5,  1.000e+00,  1.281e+01,  4.610e+00],
    # #       [ 5,  2.000e+00,  1.137e+01,  5.800e+00],
    # #       [ 6,  1.000e+00,  1.281e+01,  4.610e+00],
    # #       [ 6,  2.000e+00,  1.031e+01,  5.970e+00],
    # #       [ 7,  1.000e+00,  1.194e+01,  6.770e+00],
    # #       [ 7,  2.000e+00,  9.570e+00,  6.240e+00],
    # #       [ 8,  1.000e+00,  1.103e+01,  6.840e+00],
    # #       [ 8,  2.000e+00,  8.730e+00,  6.340e+00]])
    #     data = np.array(data)
    #     dset, _ = data_loader(self.args, data)
    #     pred_traj = self.run_generator(self.args, dset, self.generator)
    #     return pred_traj


    def run_generator(self, args, dset, generator):
        obs_traj = dset.obs_traj.permute(2,0,1)         #.cuda() ADI
        obs_traj_rel = dset.obs_traj_rel.permute(2,0,1) #.cuda() ADI
        seq_start_end = torch.tensor(dset.seq_start_end)#.cuda() ADI

        with torch.no_grad():
            pred_traj_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end
            )

            pred_traj = relative_to_abs(
                pred_traj_rel, obs_traj[-1]
            )

        return pred_traj[0,:,:].tolist()


    def predict_batch(self, x_history, xhat_history, Nveh, Nsamples, planning_dt, model_dt, x_reference=None):
        # skip_cols = int(model_dt/planning_dt) #! choose model_dt only 0.3 for now .  
        # x_history = x_history[::skip_cols] # (H,Nsample,2)
        # xhat_history = xhat_history[::skip_cols] # (H,Nveh*Nsample,2)
        obs_traj = np.append(x_history, xhat_history,axis=1) #(H,(Nveh+1)*Nsamples,2)
        try:
            x_reference = x_reference[:,:,:2]-np.append(x_history[-1][np.newaxis,:,:],x_reference[:-1,:,:2],axis=0)
            x_reference = torch.from_numpy(x_reference)
        except:
            pass
        try:
            obs_traj_rel = np.append(np.zeros((1,obs_traj.shape[1],2)),obs_traj[1:]-obs_traj[:-1],axis=0) 
        except:
            obs_traj_rel = np.zeros(obs_traj.shape)
        seq_start_end = np.column_stack((np.arange(Nsamples) * (Nveh+1), (np.arange(Nsamples) + 1) * (Nveh+1))) # (Nsample,2)
        
        obs_traj_tensor = torch.from_numpy(obs_traj).type(torch.float)          #.cuda() ADI
        obs_traj_rel_tensor = torch.from_numpy(obs_traj_rel).type(torch.float)  #.cuda() ADI
        seq_start_end_tensor = torch.from_numpy(seq_start_end)                  #.cuda() ADI
        
        pred_traj = self.run_generator_batch(obs_traj_tensor, obs_traj_rel_tensor, seq_start_end_tensor, x_reference, self.generator)
        
        
        pred_traj = np.reshape(np.array(pred_traj),(self.pred_len, Nveh+1,Nsamples,2))  
        obs_traj = np.reshape(obs_traj[-1],(Nveh+1,Nsamples,2))[np.newaxis,:,:,:]
        dx0 = (pred_traj[:1,:,:,0]-obs_traj[:1,:,:,0]) #the model_dt->planning_dt interpolation 
        dy0 = (pred_traj[:1,:,:,1]-obs_traj[:1,:,:,1]) * 1 # (Nveh+1, Nsamples)
        dx = np.append(dx0,(pred_traj[1:,:,:,0]-pred_traj[:-1,:,:,0]),axis=0)
        dy = np.append(dy0,(pred_traj[1:,:,:,1]-pred_traj[:-1,:,:,1]),axis=0)
        psi = np.arctan2(dy,dx)
        v = np.hypot(dx,dy)/planning_dt
        #TODO for when planning_dt \neq model_dt, the skip_cols in the dxdy part need to be revised
        #TODO additionally, the below xhat prediction must be changed using dx and dy with obs_traj
        xhat_predictions = np.stack((pred_traj[:,:,:,0], pred_traj[:,:,:,1], psi, v), axis= -1) # (pred_len, Nveh+1, Nsamples, 4)
        return xhat_predictions
    
    # def predict_batch(self, obs_traj, obs_traj_rel, seq_start_end):
    #     obs_traj = np.array(obs_traj)
    #     obs_traj_rel = np.array(obs_traj_rel)
    #     seq_start_end = np.array(seq_start_end)

    #     obs_traj = torch.from_numpy(obs_traj).type(torch.float)          #.cuda() ADI
    #     obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float)  #.cuda() ADI
    #     seq_start_end = torch.from_numpy(seq_start_end)                  #.cuda() ADI
    #     pred_traj = self.run_generator_batch(obs_traj, obs_traj_rel, seq_start_end, self.generator)

    #     return pred_traj


    def run_generator_batch(self, obs_traj, obs_traj_rel, seq_start_end, x_reference, generator):
        with torch.no_grad():
            pred_traj_rel = generator(
                obs_traj, obs_traj_rel, seq_start_end, x_reference
            )
            pred_traj = relative_to_abs(
                pred_traj_rel, obs_traj[-1]
            )

        return pred_traj.tolist()
