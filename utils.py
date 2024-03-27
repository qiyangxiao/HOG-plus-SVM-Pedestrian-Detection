import cv2
import numpy as np
import os
import random
import time
from PIL import Image
from eval import loadModel



def cv2HOG(winSize:tuple=(64, 128), blockSize:tuple=(16, 16), blockStride:tuple=(8, 8), 
           cellSize:tuple=(8, 8), nbins:int=9, derivAperture:int=1, 
           winSigma:int=-1, histogramNormType:int=0, L2HysThreshold:float=0.2,
           gammaCorrection:bool=True, nlevels:int=64, useSignedGradients:bool=True):
    '''
    定义 Opencv 自带的 HOG 算子
    parameters
    ---
    winSize: the window size of the image block
    blockSize: the block size of windows
    blockStride: step size of the block moving
    cellSize: the size of the gradient hist cell
    nbins: the bin count of the gradient direction hist
    derivAperture: aperture sizes used in gradient calculations
    winSigma: standard deviation of Gaussian smoothing
    histogramNormType: histogram Normalization Type
    L2HysThreshold: thresholds for L2Hys normalization type
    gammaCorrection: whether to use gamma correction
    nlevels: number of gray levels used to calculate the gradient
    useSignedGradients: whether to use signed gradient
    ---
    returns
    a cv2 HOGDescriptor
    '''
    return cv2.HOGDescriptor(winSize, blockSize, blockStride,
                             cellSize, nbins, derivAperture,
                             winSigma, histogramNormType, L2HysThreshold,
                             gammaCorrection, nlevels, useSignedGradients)

def crop(img:np.array, width:int=64, height:int=128):
    '''
    将图像进行 width * height 的中心切割
    parameters
    ---
    img: numpy array of an image
    width: the width of the croped image
    height: the height of the croped image
    ---
    returns
    the cut image numpy array
    '''
    ori_h = img.shape[0]
    ori_w = img.shape[1]
    if width > ori_w or height > ori_h:
        return False
    
    row_idx = round((ori_h - height) / 2)
    col_idx = round((ori_w - width) / 2)

    return img[row_idx:(row_idx+height), col_idx:(col_idx+width)]

def clip(img:np.array, left:int, top:int, 
         width:int=64, height:int=128):
    '''
    截取图像的任意一块儿
    parameters
    ---
    img: numpy array of an image
    left: the column index of the cliped image
    top: the row index of the cliped image
    width: the width of the cliped image
    height: the height of the cliped image
    ---
    returns
    the cut image numpy array
    '''
    ret = img[top:top+height, left:left+width]
    return ret

def loadImage(filepath:str, resize:bool=True):
    '''
    替代 cv2.imread() ，可用于处理特殊的 RGBA 图像
    parameters
    ---
    filepath: path to the image file
    ---
    returns
    the image numpy array in BGR color format
    '''
    img = Image.open(filepath)
    img_rgb = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if resize == True:
        resized_img_bgr = crop(img_bgr)
        return resized_img_bgr
    
    return img_bgr

def loadCutImage(filepath:str, width:int, height:int, count:int=10):
    '''
    随机裁剪负样本图像为 (width, height)
    parameters
    ---
    filepath: path to the negative image file
    width: required width value for training
    height: required height value for training
    count: how many cut images an input image will be cut
    ---
    a list contains all the random cut image numpy arrays
    '''
    img = Image.open(filepath)
    img_rgb = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h = img_bgr.shape[0]
    w = img_bgr.shape[1]

    max_left = w - width
    max_top = h - height

    cut_list = []
    for i in range(count):
        left = random.randint(0, max_left)
        top = random.randint(0, max_top)
        cut_img = clip(img_bgr, left, top, width, height)
        cut_list.append(cut_img)

    return cut_list
  
def slide(modelfile, imgfile:str, min_width:int=48, width_scale:float=1.25, 
          coord_step:int=16, area_width:int=64, area_height:int=128, 
          ratio:int=2, threshold:int=0.99):
    '''
    使用滑动窗口遍历，截取图中的行人
    parameters
    ---
    imgfile: path to the input image file
    min_width: the minimal width of the window
    width_scale: multiples of each window expansion
    coord_step: step size of the coordinate change
    area_width: the width of the resized window area
    area_height: the height of the resized window area
    ratio: the ratio of height-to-width
    ---
    returns
    window_list: a list contains all windows for boxing
    hog_list: the HOG values list corresponding to window_list
    prob_list: the probability values list corresponding to hog_list
    '''
    model = loadModel(modelfile)
    img = loadImage(imgfile, resize=False)
    height = img.shape[0]
    width = img.shape[1]

    w = min_width
    h = int(w * ratio)

    window_list = []
    hog_list = []
    prob_list = []
    hog = cv2HOG()
    while w < width and h < height:
        # 双层 for 循环移动窗口
        for left in range(0, width - w, coord_step):
            for top in range(0, height - h, coord_step):
                patch = clip(img, left, top, w, h)
                resized_patch = cv2.resize(patch, (area_width, area_height))
                hist = hog.compute(resized_patch)
                prob = model.predict_proba(hist.reshape(1, -1))
                if prob[0][1] >= threshold:
                    window_list.append((left, top, w, h))
                    hog_list.append(hist)
                    prob_list.append(prob[0][1])
        # 增大窗口
        w = int(w * width_scale)
        h = int(w * ratio)

    return window_list, hog_list, prob_list

def boxOut(imgfile:str, outfile:str, windows:tuple):
    '''
    生成框选出的图片
    imgfile: the original image file
    outfile: the image file after boxing out
    window: the required box
    ---
    returns
    the boxed out image numpy array
    '''

    img = loadImage(imgfile, resize=False)

    for win in windows:
        left = win[0]
        top = win[1]
        width = win[2]
        height = win[3]
        cv2.rectangle(img, (left, top), (left+width, top+height), (0, 0, 255), 2)

    cv2.imwrite(outfile, img)
    return img

def cv2NMS(windows:list, scores:list, score_threshold:float=0.5,
           nms_threshold:float=0.4):
    '''
    openCV 封装的 NMS 算法，优化框选窗口
    parameters
    ---
    windows: the window list for opitmization
    scores: the corresponding score list
    score_threshold: threshold to filter low scores
    nms_threshold: threshold to filter big IoUs
    ---
    returns
    the filtered window list
    '''
    indices = cv2.dnn.NMSBoxes(windows, scores, score_threshold, nms_threshold)

    selected_windows = [windows[i] for i in indices]

    return selected_windows

def makeDetection(model:str, detectpath:str, outpath:str):
    '''
    使用训练好的模型进行批量测试，框出图像中的行人
    parameters
    ---
    model: path to the trained model
    detectpath: path to the images for detection
    outpath: path to the results
    '''
    files = os.listdir(detectpath)
    for file in files:
        start_time = time.perf_counter()
        filepath = os.path.join(detectpath, file)
        filename = file.split('.')[0]
        extension = file.split('.')[1]
        newfile = f"{filename}_boxed.{extension}"
        window_list, _, prob_list = slide(model, filepath, min_width=64)
        result = cv2NMS(window_list, prob_list, score_threshold=0.99, nms_threshold=0.3)
        boxOut(filepath, os.path.join(outpath, newfile), result)
        end_time = time.perf_counter()
        print(f'Successfully processed image {newfile}, cost time: {(end_time - start_time):.2f}s')
