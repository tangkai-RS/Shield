import os
import torch
import random
import numpy as np
from PIL import Image
from osgeo import gdal
from tqdm import trange
from model.densecrf import densecrf
from sklearn.metrics import (cohen_kappa_score, confusion_matrix)


def imread(path):
    return gdal.Open(path).ReadAsArray()


def imsave(img, path, dtype='uint8', no_data=None):
    if len(img.shape) == 3:
        (n, h, w) = img.shape
    else:
        (h, w) = img.shape
        n = 1
       
    if dtype == 'uint8':
        datatype = gdal.GDT_Byte
    elif dtype == 'uint16':
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, w, h, n, datatype, ['COMPRESS=LZW'])   
           
    if len(img.shape) == 3:
        for t in range(n):
            dataset.GetRasterBand(t + 1).WriteArray(img[t])
            if no_data is not None:
                dataset.GetRasterBand(t + 1).SetNoDataValue(no_data)
    else:
        dataset.GetRasterBand(1).WriteArray(img)
        if no_data is not None:
            dataset.GetRasterBand(1).SetNoDataValue(no_data)
    del dataset


def aggregate_mean_mutiple_bands(img, window=20, return_pad_img=False):
    '''aggregate_mean for single images

    Parameters
    ----------
    img : numpy array C * H * W
        image array
    window : int, optional
        the ratio of the aggreated operation, by default 10

    Returns
    -------
    numpy array
        aggregated image using mean
    '''      
    c, h, w = img.shape
    pad_size_h = int(np.ceil(h / window) * window - h)      
    pad_size_w = int(np.ceil(w / window) * window - w)
    if pad_size_h % 2 == 0:
        pad_size_h /= 2
        ud = (int(pad_size_h), int(pad_size_h))
    else:
        ud = (pad_size_h, 0)
    if pad_size_w % 2 == 0:
        pad_size_w /= 2
        lr = (int(pad_size_w), int(pad_size_w))
    else:
        lr = (pad_size_w, 0)

    img_agg_list = []
    img_pad_list = []
    for i in range(c):
        img_temp = img[i, :, :]
        img_temp = np.pad(img_temp, (ud, lr), mode='symmetric') # (up, down, left, right)
        if return_pad_img:
            img_pad_list.append(img_temp)
        
        rows = np.arange(0, img_temp.shape[0], window)
        cols = np.arange(0, img_temp.shape[1], window)
        windows = [img_temp[row:row+window, col:col+window] for row in rows for col in cols]
        aggregated_shape = tuple((np.asarray(img_temp.shape) / window).astype(np.int64))
        img_agg = np.asarray([np.nanmean(window) for window in windows]).reshape(aggregated_shape)
        img_agg_list.append(img_agg)
    
    if return_pad_img:            
        return np.stack(img_agg_list, axis=0), np.stack(img_pad_list, axis=0)
    else:
        return np.stack(img_agg_list, axis=0)
    
    
def aggregate_mode_mutiple_bands(img, window=20):
    c, h, w = img.shape
    pad_size_h = int(np.ceil(h / window) * window - h)      
    pad_size_w = int(np.ceil(w / window) * window - w)
    if pad_size_h % 2 == 0:
        pad_size_h /= 2
        ud = (int(pad_size_h), int(pad_size_h))
    else:
        ud = (pad_size_h, 0)
    if pad_size_w % 2 == 0:
        pad_size_w /= 2
        lr = (int(pad_size_w), int(pad_size_w))
    else:
        lr = (pad_size_w, 0)

    def get_mode(array):
        vals, counts = np.unique(array, return_counts=True)
        index = np.argmax(counts) 
        return vals[index]
        
    img_agg_list = []
    for i in range(c):
        img_temp = img[i, :, :]
        img_temp = np.pad(img_temp, (ud, lr), mode='symmetric') # (up, down, left, right)
        
        rows = np.arange(0, img_temp.shape[0], window)
        cols = np.arange(0, img_temp.shape[1], window)
        windows = [img_temp[row:row+window, col:col+window] for row in rows for col in cols]
        aggregated_shape = tuple((np.asarray(img_temp.shape) / window).astype(np.int64))
        img_agg = np.asarray([get_mode(window) for window in windows]).reshape(aggregated_shape)
        img_agg_list.append(img_agg)
    else:
        return np.stack(img_agg_list, axis=0)


def get_pad_imgsize(img_path, patch_size):
    '''generate patch without overlapping'''
    img = imread(img_path)
    c, h_or, w_or = img.shape
    
    pad_h = int((h_or % patch_size) / 2)
    pad_w = int((w_or % patch_size) / 2)
    
    pad_tuple = (pad_w, pad_h)
    img_pad = pad_image_mutiple_bands(img, pad_tuple, pad_tuple, mode='symmetric')
    _, h_pad, w_pad = img_pad.shape
    
    col_num = int(w_pad / patch_size)
    row_num = int(h_pad / patch_size)
    
    return row_num, col_num
    

def generate_patch_dataset(img_path, patch_size, not_first=False, suffix='png'):
    '''generate patch without overlapping'''
    img = imread(img_path)
    _, h_or, w_or = img.shape
    
    # cal missing h and w
    miss_h = np.ceil(h_or / patch_size) * patch_size - h_or
    miss_w = np.ceil(w_or / patch_size) * patch_size - w_or
    
    # for up and down padding 
    if miss_h % 2 == 0:
        pad_h = int(miss_h / 2)
        pad_ud = (pad_h, pad_h)
    else:
        pad_u = np.floor(miss_h / 2) + miss_h % 2 # divide 2 and plus remainder
        pad_d = np.floor(miss_h / 2)
        pad_ud = (int(pad_u), int(pad_d))
    # for left and right padding
    if miss_w % 2 == 0:
        pad_w = int(miss_w / 2)
        pad_lr = (pad_w, pad_w)
    else:
        pad_l = np.floor(miss_w / 2) + miss_w % 2
        pad_r = np.floor(miss_w / 2)
        pad_lr = (int(pad_l), int(pad_r))
    
    img_pad = pad_image_mutiple_bands(img, pad_ud, pad_lr, mode='symmetric')
    _, h_pad, w_pad = img_pad.shape
    # save pad image to original image path
    if not not_first:
        if suffix == '.png':
            Image.fromarray(img_pad.astype(np.uint8).transpose(1, 2, 0)).save(img_path.replace('.png', '_' + str(patch_size)+'.png'))
        elif suffix == '.tif':
            img_png_path = img_path.replace('.tif', '.png')
            if not os.path.exists(img_png_path):
                print('There lack png RGB img for orignal img, thus maybe influence followed process of dense conditional random field!')
            else:
                img_png = imread(img_png_path)
                img_png_pad = pad_image_mutiple_bands(img_png, pad_ud, pad_lr, mode='symmetric')
                Image.fromarray(img_png_pad.astype(np.uint8).transpose(1, 2, 0)).save(img_path.replace('.tif', '_crf_' + str(patch_size) + '.png'))
            imsave(img_pad, img_path.replace('.tif', '_' + str(patch_size) + '.tif'), dtype='uint16')

    col_num = int(w_pad / patch_size)
    row_num = int(h_pad / patch_size)
    
    dataset = []
    idxs = []
    for i in trange(row_num):
        for j in range(col_num):
            patch = img_pad[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            dataset.append(patch)
            idxs.append([str(i) + '_' + str(j) + '.png', '_'])
    return dataset, idxs, row_num, col_num


def extract_specific_loc_patch(anomaly_binary_map, img_path, patch_size):
    '''generate patch with overlapping'''
    img = imread(img_path)
    _, h_or, w_or = img.shape
    
    # cal missing h and w
    miss_h = np.ceil(h_or / patch_size) * patch_size - h_or
    miss_w = np.ceil(w_or / patch_size) * patch_size - w_or
    
    # for up and down padding 
    if miss_h % 2 == 0:
        pad_h = int(miss_h / 2)
        pad_ud = (pad_h, pad_h)
    else:
        pad_u = np.floor(miss_h / 2) + miss_h % 2 # divide 2 and plus remainder
        pad_d = np.floor(miss_h / 2)
        pad_ud = (int(pad_u), int(pad_d))
    # for left and right padding
    if miss_w % 2 == 0:
        pad_w = int(miss_w / 2)
        pad_lr = (pad_w, pad_w)
    else:
        pad_l = np.floor(miss_w / 2) + miss_w % 2
        pad_r = np.floor(miss_w / 2)
        pad_lr = (int(pad_l), int(pad_r))

    img_pad = pad_image_mutiple_bands(img, pad_ud, pad_lr, mode='symmetric')
    c, h, w = img_pad.shape
    
    anomaly_idxs = np.argwhere(anomaly_binary_map == 255)
    row_idxs = []
    col_idxs = []
    dataset = []
    
    half = int((patch_size-1) / 2)
    for i in range(len(anomaly_idxs)):
        row_c, col_c = anomaly_idxs[i]
        row_up = row_c - half
        row_down = row_c + half + 1
        col_l = col_c - half
        col_r = col_c + half + 1
        if (row_up >= 0) and (row_down <= h) and (col_l >= 0) and (col_r <= w):
            patch = img_pad[:, row_up:row_down, col_l:col_r]
            row_idxs.append(row_c)
            col_idxs.append(col_c)
            dataset.append(patch)
    return dataset, row_idxs, col_idxs
    

def pad_img_match_patch_size(img_path, patch_size):
    '''generate patch without overlapping'''
    img = imread(img_path)
    _, h_or, w_or = img.shape
    
    # cal missing h and w
    miss_h = np.ceil(h_or / patch_size) * patch_size - h_or
    miss_w = np.ceil(w_or / patch_size) * patch_size - w_or
    
    # for up and down padding 
    if miss_h % 2 == 0:
        pad_h = int(miss_h / 2)
        pad_ud = (pad_h, pad_h)
    else:
        pad_u = np.floor(miss_h / 2) + miss_h % 2 # divide 2 and plus remainder
        pad_d = np.floor(miss_h / 2)
        pad_ud = (int(pad_u), int(pad_d))
    # for left and right padding
    if miss_w % 2 == 0:
        pad_w = int(miss_w / 2)
        pad_lr = (pad_w, pad_w)
    else:
        pad_l = np.floor(miss_w / 2) + miss_w % 2
        pad_r = np.floor(miss_w / 2)
        pad_lr = (int(pad_l), int(pad_r))
    
    img_pad = pad_image_mutiple_bands(img, pad_ud, pad_lr, mode='symmetric')
    return img_pad


def pad_image_mutiple_bands(img, ud, lr, mode='symmetric'):
    """pad image for mutiplt bands lr: tuple ud: tuple"""
    c, _, _ = img.shape
    img_list = []
    for i in range(c):
        img_temp = img[i, :, :]
        img_temp = np.pad(img_temp, (ud, lr), mode=mode)
        img_list.append(img_temp)
    return np.stack(img_list, axis=0)


def interploate_image_mutiplt_bands(img, ratio, mode=0):
    from scipy.ndimage import zoom
    '''nearest_neighbor or bilinear for mutiplt bands images

    Parameters
    ----------
    img : numpy array C * H * W
        image array
    ratio: int or float
        interploated ratio between coarse and fine image
    mode: int, optional
        0 is nearest_neighbor; 1 is bilinear; 3 is cube, default is 0

    Returns
    -------
    numpy array
        interploated image array using nearest neighbor algorithm
    '''  
    if isinstance(img, str):
        img = imread(img)
    input_img_cp = np.copy(img)   
    if len(input_img_cp.shape) < 3:
        return zoom(input_img_cp, ratio, order=mode).astype(np.uint8)
    else:
        c, h, w = input_img_cp.shape
    output_img = []
    for i in range(c):
        temp  = zoom(input_img_cp[i, :, :], ratio, order=mode).astype(np.uint8)
        output_img.append(temp)       
    return np.stack(output_img).transpose(1, 2, 0)


def otsu(data, num=400, get_bcm=True):
    # from https://github.com/ChenHongruixuan/ChangeDetectionRepository/blob/master/Methodology/util/cluster_util.py
    """
    generate binary change map based on otsu
    :param data: cluster data
    :param num: intensity number
    :param get_bcm: bool, get bcm or not
    :return:
        binary change map
        selected threshold
    """
    max_value = np.nanmax(data)
    min_value = np.nanmin(data)

    if len(data.shape) > 1:
        total_num = data.shape[0] * data.shape[1]
    elif len(data.shape) == 1:
        total_num = data.shape[0]

    # total_num = data.shape[1]
    step_value = (max_value - min_value) / num
    value = min_value + step_value
    best_threshold = min_value
    best_inter_class_var = 0
    while value <= max_value:
        data_1 = data[data <= value]
        data_2 = data[data > value]
        if data_1.shape[0] == 0 or data_2.shape[0] == 0:
            value += step_value
            continue
        w1 = data_1.shape[0] / total_num
        w2 = data_2.shape[0] / total_num

        mean_1 = data_1.mean()
        mean_2 = data_2.mean()

        inter_class_var = w1 * w2 * np.power((mean_1 - mean_2), 2)
        if best_inter_class_var < inter_class_var:
            best_inter_class_var = inter_class_var
            best_threshold = value
        value += step_value
    if get_bcm:
        bwp = np.zeros(data.shape)
        bwp[data <= best_threshold] = 0
        bwp[data > best_threshold] = 255
        return bwp, best_threshold
    else:
        return best_threshold


def accuracy_assessment_single(gt_path, changed_map, mode='change', patch_size=None):
    """
   assess accuracy of changed map based on ground truth
   :param gt_changed: changed ground truth
   :param gt_unchanged: unchanged ground truth
   :param changed_map: changed map
   :return: dict(precision=P, recall=R, F1=F1, oa=OA, kappa=kappa)
   """
    gt = imread(gt_path) # 255 = changed 0 = unchanged

    h, w = changed_map.shape
    h_gt, w_gt = gt.shape
    
    if (h != h_gt) or (w != w_gt):
        # cal missing h and w
        miss_h = np.ceil(h_gt / patch_size) * patch_size - h_gt
        miss_w = np.ceil(w_gt / patch_size) * patch_size - w_gt
        
        # for up and down padding 
        if miss_h % 2 == 0:
            pad_h = int(miss_h / 2)
            pad_ud = (pad_h, pad_h)
        else:
            pad_u = np.floor(miss_h / 2) + miss_h % 2 # divide 2 and plus remainder
            pad_d = np.floor(miss_h / 2)
            pad_ud = (int(pad_u), int(pad_d))
        # for left and right padding
        if miss_w % 2 == 0:
            pad_w = int(miss_w / 2)
            pad_lr = (pad_w, pad_w)
        else:
            pad_l = np.floor(miss_w / 2) + miss_w % 2
            pad_r = np.floor(miss_w / 2)
            pad_lr = (int(pad_l), int(pad_r))
        
        gt = np.pad(gt, (pad_ud, pad_lr), mode='symmetric')
    
    changed_map = np.reshape(changed_map, (-1,))
    gt = np.reshape(gt, (-1,))

    if mode == 'change':
        labels = [1, 2] # 2 is p
    elif mode == 'unchange':
        labels = [2, 1] # 1 is p
    else:
        raise ValueError('mode must be change or unchange')
        
    cm = np.ones((h * w,))
    cm[changed_map == 255] = 2

    gt[gt == 255] = 2 # 2 = changed
    gt[gt == 0] = 1 # 1 = unchanged

    conf_mat = confusion_matrix(y_true=gt, y_pred=cm,
                                labels=labels)
    kappa = cohen_kappa_score(y1=gt, y2=cm,
                            labels=labels)

    tn = conf_mat[0, 0]
    fn = conf_mat[1, 0]
    tp = conf_mat[1, 1]
    fp = conf_mat[0, 1]
    P = np.round((tp / (tp + fp)) * 100, 2)
    R = np.round((tp / (tp + fn)) * 100, 2)
    F1 = np.round((2 * P * R / (R + P)), 2)
    OA = np.round(((tp + tn) / (tn + fp + fn + tp)) * 100, 2)
    kappa = np.round(kappa * 100, 2)
    
    return dict(precision=P, recall=R, F1=F1, oa=OA, kappa=kappa)


def RGB_to_cluster(cluster_rgb):
    '''cluster_rgb [H W C]'''
    onehot = []
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
    colors = [list(v[0:3]) for _, v in colors.items()]
    for i, color in enumerate(colors):
        color = np.expand_dims(color, axis=(0, 1))
        cmap = np.all(np.equal(cluster_rgb, color), axis=2)
        onehot.append(cmap)        
    onehot_mask = np.stack(onehot, axis=0)
    unique_mask = np.argmax(onehot_mask, axis=0).astype(np.uint8)
    return unique_mask
    
  
def seed_torch(seed=2025):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    

