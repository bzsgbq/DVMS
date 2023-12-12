import os
import numpy as np
import cv2
from tqdm import tqdm


'''Configs:'''
dataset_name = 'Wu_MMSys_17'
SALMAP_SHAPE = (64, 128)


def resize_salmaps():
    '''Resize saliency maps to SALMAP_SHAPE'''
    salmaps_src_folder = os.path.join('.', dataset_name, 'saliency_src')
    salmaps_dst_folder = os.path.join('.', dataset_name, f'saliency_{SALMAP_SHAPE[0]}x{SALMAP_SHAPE[1]}')
    os.makedirs(salmaps_dst_folder, exist_ok=True)
    for fn in tqdm(os.listdir(salmaps_src_folder)):
        file_path = os.path.join(salmaps_src_folder, fn)
        src_salmaps = np.load(file_path)
        dst_salmaps = np.zeros((src_salmaps.shape[0], SALMAP_SHAPE[0], SALMAP_SHAPE[1]))
        for i in range(src_salmaps.shape[0]):
            salmap = src_salmaps[i]
            salmap = cv2.resize(salmap, dsize=(SALMAP_SHAPE[1], SALMAP_SHAPE[0]), interpolation=cv2.INTER_AREA)
            dst_salmaps[i] = salmap
        dst_file_path = os.path.join(salmaps_dst_folder, fn)
        np.save(dst_file_path, dst_salmaps)


if __name__ == '__main__':
    resize_salmaps()