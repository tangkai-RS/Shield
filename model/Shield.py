import os
import torch
import numbers
import math
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from model.kmeans import batch_KMeans
from model.meanshift import batch_MeanShift
from PIL import Image
from einops import rearrange

from model.autoencoder import AutoEncoder
from model.density import GaussianDensityTorch


class Shield(nn.Module):
    
    def __init__(self, args):
        super(Shield, self).__init__()
        self.args = args
        self.beta = args.beta  # coefficient of the clustering term 
        self.lamda = args.lamda  # coefficient of the reconstruction term
        self.device = torch.device(args.device)
        
        # Validation check
        if not self.beta > 0:
            msg = 'beta should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.beta))
        
        if not self.lamda > 0:
            msg = 'lamda should be greater than 0 but got value = {}.'
            raise ValueError(msg.format(self.lamda))
        
        if len(self.args.hidden_dims) == 0:
            raise ValueError('No hidden layer specified.')
        
        if args.clustering == 'kmeans':
            self.clustering = batch_KMeans(args)
        elif args.clustering == 'meanshift':
            self.clustering = batch_MeanShift(args)
        else:
            raise RuntimeError('Error: no clustering chosen')
   
        self.autoencoder = AutoEncoder(args).to(self.device)
        
        self.criterion1 = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.wd)
    
    """ Compute the Equation (5) in the original paper on a data batch """
    def _loss(self, X, cluster_id):
        # TODO: add a contrastive loss to push the distance between each cluster center as far as possible.
        batch_size = X.size()[0]
        rec_X = self.autoencoder(X)
        latent_X = self.autoencoder(X, latent=True)  
        
        # Reconstruction error
        rec_loss = self.lamda * self.criterion1(X, rec_X)
        
        # Regularization term on clustering
        dist_loss = torch.tensor(0.).to(self.device)
        clusters = torch.FloatTensor(self.clustering.clusters).to(self.device)
        for i in range(batch_size):
            diff_vec = latent_X[i] - clusters[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1),
                                            diff_vec.view(-1, 1))
            dist_loss += 0.5 * self.beta * torch.squeeze(sample_dist_loss)
        # dist_loss = dist_loss / batch_size
        
        return (rec_loss + dist_loss, 
                rec_loss.detach().cpu().numpy(),
                dist_loss.detach().cpu().numpy())
    
    def pretrain(self, train_loader, epoch=20, verbose=True):
        
        if not self.args.pretrain:
            return
        if not isinstance(epoch, numbers.Integral):
            msg = '`epoch` should be an integer but got value = {}'
            raise ValueError(msg.format(epoch))
        
        if verbose:
            print('========== Start pretraining ==========')
        
        rec_loss_list = []
        
        self.train()
        for e in range(epoch):
            for batch_idx, (data, _) in enumerate(train_loader):
                batch_size = data.size()[0]
                data = data.to(self.device).view(batch_size, -1)
                rec_X = self.autoencoder(data)
                loss = self.criterion1(data, rec_X)
                
                if verbose and (batch_idx+1) % self.args.log_interval == 0:
                    msg = 'Epoch: {:02d} | Batch: {:03d} | Rec-Loss: {:.3f}'
                    print(msg.format(e, batch_idx+1, 
                                     loss.detach().cpu().numpy()))
                    rec_loss_list.append(loss.detach().cpu().numpy())
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.eval()
        
        if verbose:
            print('========== End pretraining ==========\n')
        
        # Initialize clusters in self.clustering after pre-training
        batch_X = []
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            latent_X = self.autoencoder(data, latent=True) # batchsize dim
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.clustering.init_cluster(batch_X)
        
        return rec_loss_list
    
    def optimize_features(self, epoch, train_loader, verbose=True):
        
        for batch_idx, (data, _) in enumerate(train_loader):
            batch_size = data.size()[0]
            data = data.to(self.device).view(batch_size, -1)
            # Get the latent features
            with torch.no_grad():
                latent_X = self.autoencoder(data, latent=True)
                latent_X = latent_X.cpu().numpy()
            
            # [Step-1] Update the assignment results
            cluster_id = self.clustering.update_assign(latent_X)
            
            # [Step-2] Update clusters in batch Clustering
            elem_count = np.bincount(cluster_id, 
                                     minlength=self.args.n_clusters)
            for k in range(self.args.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                self.clustering.update_cluster(latent_X[cluster_id == k], k)
            
            # [Step-3] Update the network parameters    
            loss, rec_loss, dist_loss = self._loss(data, cluster_id)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose and (batch_idx+1) % self.args.log_interval == 0:
                msg = 'Epoch: {:02d} | Batch: {:03d} | Loss: {:.3f} | Rec-' \
                      'Loss: {:.3f} | Dist-Loss: {:.3f}'
                print(msg.format(epoch, batch_idx+1, 
                                 loss.detach().cpu().numpy(),
                                 rec_loss, dist_loss))

  
    def detect(self, cluster_pre, latent_pre, refine_cluster_map, latent_after):
        '''
        cluster latent_per: N * latent_dim numpy array
        refinec_cluster_map latent_after: H * W numpy array and latent_dim * H * W tensor
        '''
        # pre-class
        # model disturbation of pre image's latent features
        print("============ Start detect anomaly at patch scale ================\n")
        anomaly_score = np.zeros(refine_cluster_map.shape, dtype=np.float32)
        anomaly_mean_cls = []
        for cls in range(self.args.n_clusters):
            latent_pre_cls = latent_pre[cluster_pre==cls, :]
            gde = GaussianDensityTorch()
            gde.fit(latent_pre_cls, self.device)
        
            # detect the anomaly in the latent_after features based on the mahalanobis_distance and pre gde
            latent_after_cls = latent_after[:, refine_cluster_map==cls].transpose(1, 0)
            anomaly_cls = gde.predict(latent_after_cls)
            # 2023-10-10 update log convert orginal dataset to normal dataset
            if self.args.with_log:
                anomaly_cls = np.log(anomaly_cls.cpu().numpy())
            else:
                anomaly_cls = anomaly_cls.cpu().numpy()
            anomaly_score[refine_cluster_map==cls] = anomaly_cls
            anomaly_mean_cls.append(np.nanmean(anomaly_cls))
        
        # weights_cls = 1 / np.asarray(anomaly_mean_cls)
        weights_cls = np.asarray(anomaly_mean_cls) ** - 1
        nan_mask = np.isnan(weights_cls)
        weights_sum = np.sum(weights_cls[~nan_mask])
        weights_cls = weights_cls / weights_sum
        anomaly_score_norm = np.zeros(refine_cluster_map.shape, dtype=np.float32)
        for cls in range(self.args.n_clusters):
            anomaly_score_norm[refine_cluster_map==cls] = anomaly_score[refine_cluster_map==cls] * weights_cls[cls]
            
        return [anomaly_score, anomaly_score_norm], weights_cls
    
    def detect_pixel(self, cluster_pre, latent_pre, refine_cluster_map, latent_after, weights_cls, idxs: tuple):
        '''
        cluster latent_per: N * latent_dim numpy array
        refinec_cluster_map latent_after: H * W numpy array and latent_dim * H * W tensor
        '''
        print("============ Start detect anomaly at pixel scale ================\n")
        row_idxs, col_idxs = idxs
        cls_idxs = refine_cluster_map[row_idxs, col_idxs]
        anomaly_score = np.zeros_like(cls_idxs, dtype=np.float32)
        for cls in range(self.args.n_clusters):
            if len(cls_idxs==cls) == 0:
                continue
            latent_pre_cls = latent_pre[cluster_pre==cls, :]
            gde = GaussianDensityTorch()
            gde.fit(latent_pre_cls, self.device)
        
            # detect the anomaly in the latent_after features based on the mahalanobis_distance and pre gde
            latent_after_cls = latent_after[:, cls_idxs==cls].transpose(1, 0)
            anomaly_cls = gde.predict(latent_after_cls)
            anomaly_score[cls_idxs==cls] = anomaly_cls.cpu().numpy() * weights_cls[cls]
            
        return anomaly_score

    def back_to_img(self, epoch, dataset, anomaly_scores_list, img_row, img_col, visual=True):
        '''reverse the anomaly_score to the original image'''
        for name, anomaly_scores in zip(['original', 'norm'], anomaly_scores_list):
            anomaly_map = np.ones(shape=(img_row, img_col))
            for i, (path, _) in enumerate(dataset.imgs):
                row_idx, col_idx = os.path.basename(path).split('.')[0].split('_')
                row_idx, col_idx = int(row_idx), int(col_idx)
                anomaly_score = anomaly_scores[i]
                anomaly_map[row_idx, col_idx] = anomaly_score        
                
            if visual:
                save_floder = os.path.join(self.args.exp, 'visual')    
                if not os.path.exists(save_floder):
                    os.makedirs(save_floder)
                save_path = os.path.join(save_floder, name + '_anomaly_map' + '_epoch_' + str(epoch) + '.png')
                plt.figure(dpi=300, figsize=(3.5, 2))
                sns.heatmap(anomaly_map, cmap='jet', cbar=True)
                plt.axis('off')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close() 
            Image.fromarray(anomaly_map).save(save_path.replace('.png', '.tif'))
                   
        return anomaly_map

    def save_anomaly_map(self, epoch, anomaly_maps_list):
        for name, anomaly_map in zip(['original', 'norm'], anomaly_maps_list):    
            save_floder = os.path.join(self.args.exp, 'visual') 
            save_path = os.path.join(save_floder, name + '_anomaly_map' + '_epoch_' + str(epoch) + '.tif')
            Image.fromarray(anomaly_map).save(save_path)
        
        