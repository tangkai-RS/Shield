import os
import torch
import numpy as np
import time
import yaml
import argparse

from model.Shield import Shield
from PIL import Image
from model.kmeans import whole_Kmeans
from model.densecrf import densecrf
from loguru import logger
from torch.utils.data import Dataset
from utils import *


class ToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x).type(torch.FloatTensor)


class Normalize(object):
    def __init__(self, means, stds):
        self.means = means
        self.stds = stds
        
    def __call__(self, x):
        x = np.ascontiguousarray(x).transpose(2, 0, 1)
        return (x - self.means) / self.stds
 

class DatasetPatch(Dataset):
    def __init__(self, dataset, idxs, means, stds):
        self.dataset = dataset
        self.means = means
        self.stds = stds
        self.imgs = idxs

    def __len__(self):
        return len(self.dataset)
    
    def _transform_features(self, features):
        features = features.astype(np.float32)
        features = (features - self.means) / self.stds
        return torch.from_numpy(features).type(torch.FloatTensor)
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        data = np.ascontiguousarray(data)
        return self._transform_features(data), 0


class Args(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.__dict__.update(cfg)

        self.train_dir = f"./dataset/{self.dataset}/im1_{self.patch_size}{self.suffix}"
        self.test_dir = f"./dataset/{self.dataset}/im2_{self.patch_size}{self.suffix}"
        self.not_first = True

        if not os.path.exists(self.train_dir) or self.patch_size == 1:
            self.train_dir = f"./dataset/{self.dataset}/im1{self.suffix}"
            self.test_dir = f"./dataset/{self.dataset}/im2{self.suffix}"
            self.not_first = False

        self.gt_dir = f"./dataset/{self.dataset}/GT.png"

        if self.suffix == '.png' or self.exp_sub_folder == 'forest_timeseries':
            self.crf_dir = f"./dataset/{self.dataset}/im1_{self.patch_size}.png"
        else:
            self.crf_dir = f"./dataset/{self.dataset}/im1_crf_{self.patch_size}.png"

        self.exp = f"./experiments/{self.exp_sub_folder}/experiments_{self.dataset}_p{self.patch_size}_c{self.n_clusters}{self.debug}"
        self.log_path = f"./experiments/{self.exp_sub_folder}/experiments_{self.dataset}_p{self.patch_size}_c{self.n_clusters}{self.debug}/eval_log.log"

        self.input_dim = self.patch_size * self.patch_size * self.band_num

        # Fixed/default values not in YAML but still configurable later
        self.clustering = 'kmeans'
        self.pretrain = True
        self.workers = 0
        if self.resume is None:
            self.resume_epoch = self.epoch - 1
            self.resume = f"{self.exp}/checkpoint_{self.resume_epoch}.pth.tar"
        self.cluster_interval = 1
        self.n_jobs = 2
        self.log_interval = 10


def train_model(args, model, Kmeans, 
                train_loader, test_loader, train_cluster_loader,
                train_dataset, test_dataset):
    seed_torch()

    model.train()
    rec_loss_list = model.pretrain(train_loader, epoch=args.pre_epoch)

    for e in range(args.epoch):
        model.train()
        model.optimize_features(e, train_loader)

        if (e % args.cluster_interval == 0) or (e == args.epoch - 1):
            cluster_labels_train = Kmeans.run_kmeans(train_cluster_loader, model, len(train_dataset))
            Kmeans.back_to_img(e, train_dataset, cluster_labels_train, flag='pre')

            cluster_labels_test = Kmeans.run_kmeans(test_loader, model, len(test_dataset))
            Kmeans.back_to_img(e, test_dataset, cluster_labels_test, flag='after')

        torch.save({'epoch': e,
                    'state_dict': model.state_dict(),
                    'optimizer': model.optimizer.state_dict()},
                   os.path.join(args.exp, f'checkpoint_{e}.pth.tar'))

    return rec_loss_list


def detect_anomaly(args, model, Kmeans,
                   train_loader, test_loader, train_cluster_loader,
                   train_dataset, test_dataset,
                   **kwargs):

    checkpoint = torch.load(args.resume)
    epoch = os.path.basename(args.resume).split('.')[0].split('_')[-1]
    model.load_state_dict(checkpoint['state_dict'])

    cluster_pre, latent_pre = Kmeans.run_kmeans(train_cluster_loader, model, len(train_dataset), latent=True)
    cluster_path = Kmeans.back_to_img(epoch, train_dataset, cluster_pre, flag='pre')
    Image.fromarray(interploate_image_mutiplt_bands(cluster_path, ratio=args.patch_size)).save(cluster_path)

    refine_cluster_path = cluster_path.replace('.png', '_refine.png')
    if (args.n_clusters > 1) and (args.patch_size > 1):
        densecrf(args.crf_dir, cluster_path, refine_cluster_path)
    else:
        refine_cluster_path = cluster_path

    refine_cluster_map = np.asarray(Image.open(refine_cluster_path)).copy()
    refine_cluster_map = Kmeans._RGB_to_cluster(refine_cluster_map)

    _, latent_after = Kmeans.run_kmeans(test_loader, model, len(test_dataset), latent=True, update_centers=False)

    anomaly_maps_patch, weights_cls = model.detect(cluster_pre, latent_pre, cluster_pre, latent_after.transpose(1, 0))
    anomaly_maps_patch = [anomaly_map.reshape((kwargs['row_num'], kwargs['col_num'])) for anomaly_map in anomaly_maps_patch]
    model.save_anomaly_map(epoch, anomaly_maps_patch)

    flags = ['original_patch_scale', 'norm_patch_scale']
    anomaly_maps_patch_binary = []
    thresholds = []

    for anomaly_map, flag in zip(anomaly_maps_patch, flags):
        threshold_std = np.nanmean(anomaly_map) + args.gamma * np.nanstd(anomaly_map)
        anomaly_binary_std = np.zeros_like(anomaly_map)
        anomaly_binary_std[anomaly_map > threshold_std] = 255
        anomaly_binary_std = interploate_image_mutiplt_bands(anomaly_binary_std, ratio=args.patch_size)
        Image.fromarray(anomaly_binary_std.astype(np.uint8)).save(
            os.path.join(args.exp, 'visual', f"{epoch}_binary_map_std_{flag}.png")
        )

        anomaly_binary_otsu, threshold_otsu = otsu(anomaly_map)
        anomaly_binary_otsu = interploate_image_mutiplt_bands(anomaly_binary_otsu, ratio=args.patch_size)
        Image.fromarray(anomaly_binary_otsu.astype(np.uint8)).save(
            os.path.join(args.exp, 'visual', f"{epoch}_binary_map_otsu_{flag}.png")
        )

        if flag == 'norm_patch_scale':
            anomaly_maps_patch_binary += [anomaly_binary_std, anomaly_binary_otsu]
            thresholds += [threshold_std, threshold_otsu]

    for anomaly_binary_map, flag, threshold in zip(anomaly_maps_patch_binary, ['std', 'otsu'], thresholds):
        patch_cand, row_idxs, col_idxs = extract_specific_loc_patch(anomaly_binary_map, img_path=args.test_dir, patch_size=args.patch_size)
        dataset_cand = DatasetPatch(patch_cand, None, kwargs['means'], kwargs['stds'])

        dataloader = torch.utils.data.DataLoader(
            dataset_cand,
            batch_size=args.batch_size,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True if args.workers > 0 else False,
            shuffle=False
        )

        _, latent_cand = Kmeans.run_kmeans(dataloader, model, len(dataset_cand), latent=True)

        anomaly_score_pixel = model.detect_pixel(
            cluster_pre, latent_pre, refine_cluster_map,
            latent_cand.transpose(1, 0),
            weights_cls, (row_idxs, col_idxs)
        )

        anomaly_binary_pixel = np.zeros_like(anomaly_score_pixel)
        anomaly_binary_pixel[anomaly_score_pixel > threshold] = 255

        anomaly_map_pixel = np.zeros_like(refine_cluster_map)
        anomaly_map_pixel[row_idxs, col_idxs] = anomaly_binary_pixel

        Image.fromarray(anomaly_map_pixel.astype(np.uint8)).save(
            os.path.join(args.exp, 'visual', f"{epoch}_binary_map_norm_{flag}_pixel_scale.png")
        )

        if flag == 'std':
            anomaly_binary_std = anomaly_map_pixel

    print("=============== Done ===============\n")
    return anomaly_binary_std

    
def main(args, model, Kmeans, 
         train_loader, test_loader, train_cluster_loader,
         train_dataset, test_dataset,
         **kwargs):
    
    if 'train' in args.mode:
        res = train_model(args, model, Kmeans,
                           train_loader, test_loader, train_cluster_loader,
                           train_dataset, test_dataset)
    
    if 'detect' in args.mode:
        res = detect_anomaly(args, model, Kmeans,
                              train_loader, test_loader, train_cluster_loader,
                              train_dataset, test_dataset,
                              **kwargs)
    return res

              
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shield')
    parser.add_argument('--config', type=str, default='configs/landslide.yaml', help='Path to YAML config file')
    args_cli = parser.parse_args()

    args = Args(config_path=args_cli.config) 
    trace = logger.add("eval_log.log")

    # record time cost
    start = time.time()
    
    # creating checkpoint repo
    exp_check = os.path.join(args.exp)
    if not os.path.isdir(exp_check):
        os.makedirs(exp_check)

    # preprocessing of data
    train_patch, train_idxs, row_num, col_num = generate_patch_dataset(
        args.train_dir, patch_size=args.patch_size, not_first=args.not_first, suffix=args.suffix
    )
    test_patch, test_idxs, row_num, col_num = generate_patch_dataset(
        args.test_dir, patch_size=args.patch_size, not_first=args.not_first, suffix=args.suffix
    )
        
    img1 = imread(args.train_dir)
    means = np.mean(img1, axis=(1, 2)).reshape(img1.shape[0], 1, 1)
    stds = np.std(img1, axis=(1, 2)).reshape(img1.shape[0], 1, 1)
    img2 = imread(args.test_dir)
    
    train_dataset = DatasetPatch(train_patch, train_idxs, means, stds)
    test_dataset = DatasetPatch(test_patch, test_idxs, means, stds)
        
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        shuffle=False
    )
    
    train_cluster_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
        shuffle=False
    )

    # main 
    model = Shield(args) 
    Kmeans = whole_Kmeans(args, row_num, col_num)   
    
    ad_map = main(
        args, model, Kmeans,
        train_loader, test_loader, train_cluster_loader,
        train_dataset, test_dataset,
        row_num=row_num, col_num=col_num,
        img1=img1, img2=img2,
        means=means, stds=stds
    )
    
    # evaluation
    # if ('detect' in args.mode) and os.path.exists(args.gt_dir):
    #     logger.info(args.exp)
    #     f1s = []
    #     for m in ['change', 'unchange']:
    #         eval_res = accuracy_assessment_single(args.gt_dir, ad_map, mode=m, patch_size=args.patch_size)
    #         logger.info(eval_res) 
    #         f1s.append(eval_res['F1'])
    #     logger.info(np.mean(f1s))
    
    end = time.time()
    print(f"Time Cost: {(end-start)/60:.2f} minutes")   
