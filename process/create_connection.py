import numpy as np
import argparse
import cv2
import os, time
from tqdm import tqdm
import mmcv
def direction_process_d1(imgpath, savedir):
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = np.where(img > 0, 1, 0)
    shp = img.shape

    img_pad = np.zeros([shp[0] + 4, shp[0] + 4])
    img_pad[2:-2, 2:-2] = img
    dir_array0 = np.zeros([shp[0], shp[1], 3])
    dir_array1 = np.zeros([shp[0], shp[1], 3])
    dir_array2 = np.zeros([shp[0], shp[1], 3])

    for i in range(shp[0]):
        for j in range(shp[1]):
            if img[i, j] == 0:
                continue
            dir_array0[i, j, 0] = img_pad[i, j]
            dir_array0[i, j, 1] = img_pad[i, j + 2]
            dir_array0[i, j, 2] = img_pad[i, j + 4]
            dir_array1[i, j, 0] = img_pad[i + 2, j]
            dir_array1[i, j, 1] = img_pad[i + 2, j + 2]
            dir_array1[i, j, 2] = img_pad[i + 2, j + 4]
            dir_array2[i, j, 0] = img_pad[i + 4, j]
            dir_array2[i, j, 1] = img_pad[i + 4, j + 2]
            dir_array2[i, j, 2] = img_pad[i + 4, j + 4]
    cv2.imwrite(os.path.join(savedir, imgpath.split('/')[-1].split('.')[0] + '_0' + '.png'), dir_array0 * 255)
    cv2.imwrite(os.path.join(savedir, imgpath.split('/')[-1].split('.')[0] + '_1' + '.png'), dir_array1 * 255)
    cv2.imwrite(os.path.join(savedir, imgpath.split('/')[-1].split('.')[0] + '_2' + '.png'), dir_array2 * 255)


def direction_process_d3(imgpath, savedir):
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    img = np.where(img > 0, 1, 0)
    shp = img.shape

    img_pad = np.zeros([shp[0] + 8, shp[0] + 8])
    img_pad[4:-4, 4:-4] = img
    dir_array0 = np.zeros([shp[0], shp[1], 3])
    dir_array1 = np.zeros([shp[0], shp[1], 3])
    dir_array2 = np.zeros([shp[0], shp[1], 3])

    for i in range(shp[0]):
        for j in range(shp[1]):
            if img[i, j] == 0:
                continue
            dir_array0[i, j, 0] = img_pad[i, j]
            dir_array0[i, j, 1] = img_pad[i, j + 4]
            dir_array0[i, j, 2] = img_pad[i, j + 8]
            dir_array1[i, j, 0] = img_pad[i + 4, j]
            dir_array1[i, j, 1] = img_pad[i + 4, j + 4]
            dir_array1[i, j, 2] = img_pad[i + 4, j + 8]
            dir_array2[i, j, 0] = img_pad[i + 8, j]
            dir_array2[i, j, 1] = img_pad[i + 8, j + 4]
            dir_array2[i, j, 2] = img_pad[i + 8, j + 8]
    cv2.imwrite(os.path.join(savedir, imgpath.split('/')[-1].split('.')[0] + '_0' + '.png'), dir_array0 * 255)
    cv2.imwrite(os.path.join(savedir, imgpath.split('/')[-1].split('.')[0] + '_1' + '.png'), dir_array1 * 255)
    cv2.imwrite(os.path.join(savedir, imgpath.split('/')[-1].split('.')[0] + '_2' + '.png'), dir_array2 * 255)


def process(imgname):
    imgdir="/data/chusai_release/road/gt"
    savedir_d1='/data/chusai_release/road/connect_8_d1'
    savedir_d3='/data/chusai_release/road/connect_8_d3'
    direction_process_d1(os.path.join(imgdir, imgname), savedir_d1)
    direction_process_d3(os.path.join(imgdir, imgname), savedir_d3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--base_dir', type=str, default='/data/chusai_release/road',
        help='Base directory for Spacenent Dataset.')
    args = parser.parse_args()

    gt_path = os.path.join(args.base_dir, 'gt')#ground truth

    start = time.time()
    ##  connectivity cube
    _ = mmcv.track_parallel_progress(process, os.listdir(gt_path), 16)
    end = time.time()
    print('Finished Creating connectivity cube, time {0}s'.format(end - start))

if __name__ == "__main__":
    main()

