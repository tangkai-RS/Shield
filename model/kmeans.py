import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from PIL import Image
from torch import Tensor


def _parallel_compute_distance(X, cluster):
    n_samples = X.shape[0]
    dis_mat = np.zeros((n_samples, 1))
    for i in range(n_samples):
        dis_mat[i] += np.sqrt(np.sum((X[i] - cluster) ** 2, axis=0))
    return dis_mat


class batch_KMeans(object):
    '''cluster batch data during training'''
    def __init__(self, args):
        self.args = args
        self.latent_dim = args.latent_dim
        self.n_clusters = args.n_clusters
        self.clusters = np.zeros((self.n_clusters, self.latent_dim))
        self.count = 100 * np.ones((self.n_clusters))  # serve as learning rate
        self.n_jobs = args.n_jobs
    
    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)
        
        return dis_mat
    
    def init_cluster(self, X):
        """ Generate initial clusters using sklearn.Kmeans """
        model = KMeans(n_clusters=self.n_clusters,
                       n_init=20)
        model.fit(X)
        self.clusters = model.cluster_centers_  # copy clusters
    
    def update_cluster(self, X, cluster_idx):
        """ Update clusters in Kmeans on a batch of data """
        n_samples = X.shape[0]
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.clusters[cluster_idx] + 
                               eta * X[i])
            self.clusters[cluster_idx] = updated_cluster
    
    def update_assign(self, X):
        """ Assign samples in `X` to clusters """
        dis_mat = self._compute_dist(X)
        
        return np.argmin(dis_mat, axis=1)
    

class whole_Kmeans(object):
    '''cluster whole dataset to check visualization of original image'''
    def __init__(self, args, img_row, img_col):
        self.args = args
        self.n_jobs = args.n_jobs
        self.n_clusters = args.n_clusters
        self.batch = args.batch_size
        self.img_row = img_row
        self.img_col = img_col
        self.device = args.device
        self.n_jobs = args.n_jobs
              
    def _define_cmap(self):
        colors = dict((
                        (0, (250, 230, 160, 255)), # 淡黄色
                        (1, (0, 154, 0, 255)),   # 绿色
                        (2, (220, 0, 0, 255)),  # 红色
                        (3, (255, 170, 0, 255)),  # 橙色
                        (4, (0, 0, 180, 255)), # 蓝色
                        (5, (230, 0, 255, 255)), # 紫色
                        (6, (255, 181, 197, 255)), # 粉色
                        (7, (255, 230, 0, 255)), # 黄色
                    ))
        self.colors = [list(v[0:3]) for _, v in colors.items()]
        for k in colors:
            v = colors[k]
            _v = [_v / 255.0 for _v in v]
            colors[k] = _v
        index_colors = [colors[key] if key in colors else
                        (255, 255, 255, 0) for key in range(0, len(colors))]
        cmap = plt.matplotlib.colors.ListedColormap(index_colors, 'Classification', len(index_colors))               
        return cmap
    
    def _cluster_to_RGB(self, cluster_map):
        h, w = cluster_map.shape
        rgb = np.zeros([h, w, 3]).astype(np.uint8)
        for i in range(self.n_clusters):
            rgb[cluster_map==i, :] = self.colors[i]
        return rgb        

    def _RGB_to_cluster(self, cluster_rgb):
        '''cluster_rgb [H W C]'''
        onehot = []
        for i, color in enumerate(self.colors):
            color = np.expand_dims(color, axis=(0, 1))
            cmap = np.all(np.equal(cluster_rgb, color), axis=2)
            onehot.append(cmap)        
        onehot_mask = np.stack(onehot, axis=0)
        unique_mask = np.argmax(onehot_mask, axis=0).astype(np.uint8)
        return unique_mask
            
    def _compute_features(self, cluster_loader, model, N):
        model.eval()
        with torch.no_grad():
            for i, (input_tensor, _) in enumerate(cluster_loader):
                batch_size = input_tensor.size()[0]
                input_var = input_tensor.to(self.device).view(batch_size, -1)                
                aux = model.autoencoder(input_var, latent=True).data.cpu().numpy()

                if i == 0:
                    features = np.zeros((N, aux.shape[1]), dtype='float32')

                aux = aux.astype('float32')
                if i < len(cluster_loader) - 1:
                    features[i * self.batch: (i + 1) * self.batch] = aux
                else:
                    # special treatment for final batch
                    features[i * self.batch:] = aux
        return features
    
    def run_kmeans(self, cluster_loader, model, N, latent=False, update_centers=True):
        print("============ Extract Deep features or Running kmeans ================\n")
        features = self._compute_features(cluster_loader, model, N)
        model = KMeans(n_clusters=self.n_clusters)
        cluster_labels = model.fit_predict(features)
        if update_centers:
            self.clusters = model.cluster_centers_
        if latent:
            return cluster_labels, torch.from_numpy(features).to(self.device)
        else:
            return cluster_labels

    def run_kmeans_or(self, img, latent=False, patch_size=9, visual=True):
        print("============ Running kmeans for Or Image ================\n")
        def generate_patch_dataset(img, patch_size):
            '''generate patch without overlapping'''
            def pad_image_mutiple_bands(img, ud, lr, mode='symmetric'):
                """pad image for mutiplt bands lr: tuple ud: tuple"""
                c, _, _ = img.shape
                img_list = []
                for i in range(c):
                    img_temp = img[i, :, :]
                    img_temp = np.pad(img_temp, (ud, lr), mode=mode)
                    img_list.append(img_temp)
                return np.stack(img_list, axis=0)   
            
            c, h_or, w_or = img.shape
            pad_h = int((h_or % patch_size) / 2)
            pad_w = int((w_or % patch_size) / 2)
            
            pad_tuple = (pad_w, pad_h)
            img_pad = pad_image_mutiple_bands(img, pad_tuple, pad_tuple, mode='symmetric')
            _, h_pad, w_pad = img_pad.shape
            
            col_num = int(w_pad / patch_size)
            row_num = int(h_pad / patch_size)
            
            dataset = []
            for i in range(row_num):
                for j in range(col_num):
                    patch = img_pad[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]        
                    dataset.append(patch)
            return dataset, row_num, col_num
        
        features, row_num, col_num = generate_patch_dataset(img, patch_size)
        features = np.asarray(features)
        features = features.reshape(features.shape[0], -1)
        model = KMeans(n_clusters=self.n_clusters)
        cluster_labels = model.fit_predict(features)
        
        if visual:
            cluster_map = cluster_labels.reshape(row_num, col_num)
            save_floder = os.path.join(self.args.exp, 'visual')    
            if not os.path.exists(save_floder):
                os.makedirs(save_floder)
            save_path = os.path.join(save_floder, 'pre' + '_or' + '.png')
            png_save_path = save_path.replace('.png', '_RGB.png')
            _ = self._define_cmap()
            Image.fromarray(self._cluster_to_RGB(cluster_map)).save(png_save_path)

        if latent and visual:
            return cluster_labels, features, png_save_path
        elif latent and (not visual):
            return cluster_labels, features
        else:
            return cluster_labels, png_save_path

    def back_to_img(self, epoch, dataset, cluster_labels, flag='train', visual=True):    
        cluster_map = np.zeros(shape=(self.img_row, self.img_col))
        for i, (path, _) in enumerate(dataset.imgs):
            row_idx, col_idx = os.path.basename(path).split('.')[0].split('_')
            row_idx, col_idx = int(row_idx), int(col_idx)
            cluster_label = cluster_labels[i]
            cluster_map[row_idx, col_idx] = cluster_label
        
        if visual:
            save_floder = os.path.join(self.args.exp, 'visual')    
            if not os.path.exists(save_floder):
                os.makedirs(save_floder)
            save_path = os.path.join(save_floder, flag + '_epoch_' + str(epoch) + '.png')
            
            # plt.figure(dpi=300, figsize=(3.5, 2))
            # plt.imshow(cluster_map, cmap=self._define_cmap())
            # plt.axis('off')
            # plt.savefig(save_path, bbox_inches='tight')
            # plt.close()
            
            _ = self._define_cmap()
            png_save_path = save_path.replace('.png', '_RGB.png')
            Image.fromarray(self._cluster_to_RGB(cluster_map)).save(png_save_path)
             
        return png_save_path
                
    def interpolate_back_to_img(self, dataset, latent):    
        latent = torch.from_numpy(latent).to(self.device)
        latent_map = torch.zeros(size=(self.args.latent_dim, self.img_row, self.img_col)).to(self.device)
        
        for i, (path, _) in enumerate(dataset.imgs):
            row_idx, col_idx = os.path.basename(path).split('.')[0].split('_')
            row_idx, col_idx = int(row_idx), int(col_idx)
            latent_map[:, row_idx, col_idx] = latent[i]
        
        row_inter = self.img_row * self.args.patch_size
        col_inter = self.img_col * self.args.patch_size
        latent_map = F.interpolate(latent_map.float().unsqueeze(0), size=(row_inter,col_inter), mode='bilinear').squeeze()
              
        return latent_map
    
    def assign_cluster(self, X):
        """ Assign samples in `X` to clusters """
        print("============ Obtain from-to ================\n")
        if isinstance(X, Tensor):
            X = X.cpu().numpy()
        c, h, w = X.shape
        X = X.reshape(c, -1).transpose(1, 0) # N * dim
        dis_mat = self._compute_dist(X)
        cluster = np.argmin(dis_mat, axis=1)
        
        return cluster.reshape(h, w).astype(np.uint8)
    
    def _compute_dist(self, X):
        dis_mat = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_compute_distance)(X, self.clusters[i])
            for i in range(self.n_clusters))
        dis_mat = np.hstack(dis_mat)
        
        return dis_mat