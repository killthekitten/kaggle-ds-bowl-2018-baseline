import numpy as np
import pandas as pd
from skimage import morphology
from skimage.morphology import binary_closing, binary_opening, disk, binary_dilation

def run_length_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    run_lengths = ' '.join([str(r) for r in run_lengths])
    return run_lengths

def numpy2encoding(predicts, img_name):
    """predicts: [H, W, N] instance binary masks
    注意： 中间可能有洞
    """
    ImageId = []
    EncodedPixels = []
    for i in range(predicts.shape[2]): 
        rle = run_length_encoding(predicts[:,:,i])
        ImageId.append(img_name)
        EncodedPixels.append(rle)    
    return ImageId, EncodedPixels

def numpy2encoding_no_overlap(predicts, img_name):
    """predicts: [H, W, N] instance binary masks
    注意： 中间可能有洞
    remove overlapping parts
    """
    sum_predicts = np.sum(predicts, axis=2)
    sum_predicts[sum_predicts>=2] = 0
    sum_predicts = np.expand_dims(sum_predicts, axis=-1)
    predicts = predicts * sum_predicts
    
    ImageId = []
    EncodedPixels = []
    for i in range(predicts.shape[2]): 
        rle = run_length_encoding(predicts[:,:,i])
        if len(rle)>0:
            ImageId.append(img_name)
            EncodedPixels.append(rle)    
    return ImageId, EncodedPixels

def numpy2encoding_no_overlap2(predicts, img_name, scores):
    """predicts: [H, W, N] instance binary masks
    注意： 中间可能有洞
    overlapping parts are given to the instance of highest score (i.e. DETECTION CONFIDENCE)
    """
    # refine your masks here !
#    predicts = np.apply_along_axis(refineMasks, 2, predicts)
#    for i in range(predicts.shape[2]-1):
#        predicts[:,:,i] = refineMasks(predicts[:,:,i])
#    
    sum_predicts = np.sum(predicts, axis=2)
    rows, cols = np.where(sum_predicts>=2)
    
    for i in zip(rows, cols):
        instance_indicies = np.where(np.any(predicts[i[0],i[1],:]))[0]
        highest = instance_indicies[0]
        predicts[i[0],i[1],:] = predicts[i[0],i[1],:]*0
        predicts[i[0],i[1],highest] = 1
    
    ImageId = []
    EncodedPixels = []
    for i in range(predicts.shape[2]): 
        rle = run_length_encoding(predicts[:,:,i])
        if len(rle)>0:
            ImageId.append(img_name)
            EncodedPixels.append(rle)    
    return ImageId, EncodedPixels

def refineMasks(mask):
    return binary_dilation(mask, disk(1))


def write2csv(file, ImageId, EncodedPixels):
    df = pd.DataFrame({ 'ImageId' : ImageId , 'EncodedPixels' : EncodedPixels})
    df.to_csv(file, index=False, columns=['ImageId', 'EncodedPixels'])



def clean_img(x):
    """http://blog.csdn.net/haoji007/article/details/52063306
        closing 先膨胀再腐蚀，可用来填充孔洞
        opening 先腐蚀再膨胀，可以消除小物体或小斑块
    https://www.kaggle.com/kmader/nuclei-overview-to-submission: 
        remove single pixels, connect nearby regions
    """
    return binary_opening(binary_closing(x, disk(1)), disk(3))
