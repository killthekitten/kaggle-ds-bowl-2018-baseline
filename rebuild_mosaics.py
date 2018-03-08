import numpy as np
import pandas as pd
import cv2
import os
from glob import glob
from scipy.misc import imsave
from tqdm import tqdm


TRAIN_DIR = "stage1_train"
TEST_DIR = "stage1_test"
TRAIN_MOSAICS_DIR = os.path.join(TRAIN_DIR, "mosaics")
TEST_MOSAICS_DIR = os.path.join(TEST_DIR, "mosaics")


def combine_images(img_ids, source_dir):
    """
    Loads 4 images and combines a mosaic img usin them as follows:

        2 3
        0 1
    """
    images = [cv2.imread(os.path.join(source_dir, img_id, "images", img_id + ".png")) for img_id in img_ids]
    down = np.hstack([images[0], images[1]])
    up = np.hstack([images[2], images[3]])
    full = np.vstack([up,down])

    return full


def load_masks(img_id, source_dir):
    """
    Loads all masks related to the `img_id` image and stacks them into a single numpy array.
    """
    mask_paths = glob(os.path.join(source_dir, img_id, "masks", "**.png"))
    masks = [cv2.imread(mask_path, 0) for mask_path in mask_paths]
    masks = np.stack(masks, axis=-1)
    masks = np.where(masks > 128, 1, 0)

    return masks


def map_layers_left_to_right(mask, center, left_half_idx, right_half_idx):
    """
    Selects two adjacent 1-pixel stripes on the edges of left and right halves, then calculates
    how many pixels touch each other for each layer pair. Returns a map of scores.
    """
    result = np.zeros((left_half_idx.shape[0], right_half_idx.shape[0]))

    for left_i, left_id in enumerate(left_half_idx):
        for right_i, right_id in enumerate(right_half_idx):
            result[left_i, right_i] = \
                np.logical_and(
                    mask[:, center - 1, left_id],
                    mask[:, center, right_id]).sum()

    return result


def map_layers_top_to_bottom(mask, center, top_half_idx, bottom_half_idx):
    """
    Same as `map_layers_left_to_right`, but top to bottom.
    """
    result = np.zeros((top_half_idx.shape[0], bottom_half_idx.shape[0]))

    for top_i, top_id in enumerate(top_half_idx):
        for bottom_i, bottom_id in enumerate(bottom_half_idx):
            result[top_i, bottom_i] = \
                np.logical_and(
                    mask[center - 1, :, top_id],
                    mask[center, :, bottom_id]).sum()

    return result


def merge_layers(mask, intersection_map, survivor_half_idx, merged_half_idx):
    """
    Merges `merged_half_idx` channels of the mask into `survivor_half_idx`. If no intersection
    was found between layers, layers are kept.
    """
    merged_idx = []

    for i in range(survivor_half_idx.shape[0]):
        if intersection_map[i].max() == 0:
            continue

        survivor_id = survivor_half_idx[i]
        merged_id = merged_half_idx[intersection_map[i].argmax()]

        mask[:, :, survivor_id] = np.logical_or(mask[:, :, survivor_id], mask[:, :, merged_id])
        merged_idx.append(merged_id)

    return np.delete(mask, merged_idx, axis=-1), len(merged_idx)


def merge_layers_on_edges(mask):
    """
    First merges layers left halves to right halves, then top to bottom.
    """

    lr_center = int(mask.shape[1] / 2)
    left_half_idx = np.argwhere(mask[:, lr_center - 1, :].sum(axis=0) > 0).flatten()
    right_half_idx = np.argwhere(mask[:, lr_center, :].sum(axis=0) > 0).flatten()
    lr_map = map_layers_left_to_right(mask, lr_center, left_half_idx, right_half_idx)
    mask, deleted_layers_count_ltr = merge_layers(mask, lr_map, left_half_idx, right_half_idx)

    tb_center = int(mask.shape[0] / 2)
    top_half_idx = np.argwhere(mask[tb_center - 1, :, :].sum(axis=0) > 0).flatten()
    bottom_half_idx = np.argwhere(mask[tb_center, :, :].sum(axis=0) > 0).flatten()
    tb_map = map_layers_top_to_bottom(mask, tb_center, top_half_idx, bottom_half_idx)
    mask, deleted_layers_count_ttb = merge_layers(mask, tb_map, top_half_idx, bottom_half_idx)

    return mask, deleted_layers_count_ltr + deleted_layers_count_ttb


def combine_masks(img_ids, source_dir):
    """
    Loads 4 masks and combines a mosaic img usin them as follows (_terribly slow_):
        2 3
        0 1
    """
    masks = [load_masks(img_id, source_dir) for img_id in img_ids]
    w = masks[0].shape[1] + masks[1].shape[1]
    channels = sum([mask.shape[-1] for mask in masks])

    return np.dstack([
        np.vstack([
            np.zeros((masks[0].shape[0], w, masks[0].shape[-1])),
            np.hstack([masks[0], np.zeros(masks[0].shape)])
        ]),
        np.vstack([
            np.zeros((masks[1].shape[0], w, masks[1].shape[-1])),
            np.hstack([np.zeros(masks[1].shape), masks[1]])
        ]),
        np.vstack([
            np.hstack([masks[2], np.zeros(masks[2].shape)]),
            np.zeros((masks[2].shape[0], w, masks[2].shape[-1]))
        ]),
        np.vstack([
            np.hstack([np.zeros(masks[3].shape), masks[3]]),
            np.zeros((masks[3].shape[0], w, masks[3].shape[-1]))
        ])
    ]).astype(bool)


def extract_mosaic_ids_from_df(df):
    """
    Regroups Emil's df in a way to be used with `combine_masks` and `combine_images` (watch the order!)
    """
    mosaics = \
        df[~df.mosaic_position.isnull()] \
            .sort_values(["mosaic_idx", "mosaic_position"]) \
            .groupby("mosaic_idx") \
            .agg({ "img_id": lambda x: list(x) })
    non_mosaics = df[df.mosaic_position.isnull()]

    return mosaics, non_mosaics


if __name__ == "__main__":
    if not os.path.exists(TRAIN_MOSAICS_DIR):
        os.mkdir(TRAIN_MOSAICS_DIR)

    if not os.path.exists(TEST_MOSAICS_DIR):
        os.mkdir(TEST_MOSAICS_DIR)

    train_df = pd.read_csv("share_train_df.csv")
    test_df = pd.read_csv('share_test_df.csv')

    mosaics, non_mosaics = extract_mosaic_ids_from_df(train_df)
    test_mosaics, test_non_mosaics = extract_mosaic_ids_from_df(test_df)

    print("Generating train mosaics:")
    for i, row in tqdm(mosaics.iterrows()):
        mosaic = combine_images(row["img_id"], TRAIN_DIR)
        cv2.imwrite(os.path.join(TRAIN_MOSAICS_DIR, str(i) + ".png"), mosaic)
        mask_mosaic = combine_masks(row["img_id"], TRAIN_DIR)
        mask_mosaic, deleted_layers_count = merge_layers_on_edges(mask_mosaic)
        np.save(os.path.join(TRAIN_MOSAICS_DIR, str(i) + ".npy"), mask_mosaic.astype(bool))
        
    print("Copying non-mosaic train images to the same place:")
    for i, row in tqdm(non_mosaics.iterrows()):
        non_mosaic = cv2.imread(os.path.join(TRAIN_DIR, row["img_id"], "images", row["img_id"] + ".png"))
        cv2.imwrite(os.path.join(TRAIN_MOSAICS_DIR, str(row["mosaic_idx"]) + ".png"), non_mosaic)

    print("Generating test mosaics:")
    for i, row in tqdm(test_mosaics.iterrows()):
        mosaic = combine_images(row["img_id"], TEST_DIR)
        cv2.imwrite(os.path.join(TEST_MOSAICS_DIR, str(i) + ".png"), mosaic)

    print("Copying non-mosaic test images to the same place:")
    for i, row in tqdm(test_non_mosaics.iterrows()):
        non_mosaic = cv2.imread(os.path.join(TEST_DIR, row["img_id"], "images", row["img_id"] + ".png"))
        cv2.imwrite(os.path.join(TEST_MOSAICS_DIR, str(row["mosaic_idx"]) + ".png"), non_mosaic)
