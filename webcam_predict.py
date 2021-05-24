# -*- coding: UTF-8 -*-
# ---------------------------------------------------------------------------
# Official code of our paper:Bilateral Grid Learning for Stereo Matching Network
# Written by Bin Xu
# ---------------------------------------------------------------------------
from __future__ import print_function, division
import argparse
import os
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from datasets.data_io import get_transform
import torch.utils.data
import time
from datasets import __datasets__
import gc
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from models.bgnet import BGNet
from models.bgnet_plus import BGNet_Plus
import numpy as np
import cv2

import webcamgrabber
import visualise

def test_transform(left_img, right_img):
    orig_h, orig_w = left_img.shape[:2]
    
    # h, w = left_img.shape[:2]
    # top_pad = 384 - h
    # right_pad = 1280 - w
    # assert top_pad >= 0 and right_pad >= 0

    left_img = np.ascontiguousarray(left_img, dtype=np.float32)
    right_img = np.ascontiguousarray(right_img, dtype=np.float32)

    # left_img = np.lib.pad(
    #     left_img, ((top_pad, 0), (0, right_pad)), mode='symmetric')
    # right_img = np.lib.pad(
    #     right_img, ((top_pad, 0), (0, right_pad)), mode='symmetric')

    preprocess = get_transform()
    left_processed = preprocess(left_img)
    right_processed = preprocess(right_img)


    torch.unsqueeze(left_processed, 0)
    torch.unsqueeze(right_processed, 0)
    torch.unsqueeze(left_processed, 0)
    torch.unsqueeze(right_processed, 0)
    return left_processed, right_processed

def main():
    model = BGNet_Plus().cuda()

    checkpoint = torch.load('models/Sceneflow-IRS-BGNet-Plus.pth',
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint)
    model.eval()


    cam = webcamgrabber.Arducam(
        "udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=JPEG, payload=96 ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink")
    left_img, right_img = cam.read()
    orig_h, orig_w = left_img.shape[:2]
    vis = visualise.Visualiser(cam.Q_)


    inference_time = 0
    framecount = 0
    while True:
        framecount += 1

        left_img, right_img = cam.read()
        left_img = cv2.resize(left_img, (640, 384), cv2.INTER_CUBIC)
        right_img = cv2.resize(right_img, (640, 384), cv2.INTER_CUBIC)
        left_img_mono = left_img[:, :, 0]
        right_img_mono = right_img[:, :, 0]

        cv2.imshow("left", left_img)
        cv2.imshow("right", right_img)
        imgL, imgR = test_transform(left_img_mono, right_img_mono)

        # imgL, imgR = sample['left'], sample['right']
        imgL = imgL.cuda()
        imgR = imgR.cuda()
        
        with torch.no_grad():
            time_start = time.perf_counter()
            pred = model(imgL.unsqueeze(0).cuda(), imgR.unsqueeze(0).cuda()) # input should be [1, 1, H, W]
            inference_time += time.perf_counter() - time_start
        print(torch.max(pred))
        pred2 = pred[0].data.cpu().numpy()
        # print('pred',pred.shape)
        dispNormalised = pred2 / np.max(pred2)
        # print(f"disp - shape {disp.shape}, max {np.max(disp)}, min {np.min(disp)}")
        cv2.imshow("Disparity", dispNormalised)

        vis.update(pred2, left_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cam.release()
            break

    print('=> Mean inference time for %d images: %.3fs' %
        (framecount, inference_time / framecount))


if __name__ == "__main__":
    main()
