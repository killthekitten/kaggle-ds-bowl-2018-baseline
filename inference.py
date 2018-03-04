import model as modellib
import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm
from inference_config import inference_config
from bowl_dataset import BowlDataset
from utils import rle_encode, rle_decode, rle_to_string

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

dataset_test = BowlDataset()
dataset_test.load_bowl('stage1_test')
dataset_test.prepare()

output = []
sample_submission = pd.read_csv('stage1_sample_submission.csv')

for image_id in tqdm(sample_submission.ImageId):
    image_path = os.path.join('stage1_test', image_id, 'images', image_id + '.png')
    
    original_image = cv2.imread(image_path)
    results = model.detect([original_image], verbose=0)
    r = results[0]
    
    masks = r['masks']
    
    count = masks.shape[-1]
    occlusion = np.logical_not(masks[:, :, -1]).astype(np.uint8)
    
    for i in range(count - 2, -1, -1):
        mask = masks[:, :, i] * occlusion
        mask_rle = rle_to_string(rle_encode(mask))
        
        # Sanity check
        try:
            rle_decode(mask_rle, original_image.shape[:-1])
            output.append([image_id, mask_rle])
            occlusion = np.logical_and(occlusion, np.logical_not(masks[:, :, i]))
        
        except Exception as e:
            print(e)
            print(image_id)
            print('---')
        
output_df = pd.DataFrame(output, columns=['ImageId', 'EncodedPixels'])
output_df.to_csv('submission.csv', index=False, encoding='utf-8')
        