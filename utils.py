import os
import json
import cv2 as cv
import numpy as np
import pandas as pd
from config import PATH_TO_IMAGES, PATH_TO_PREPROCESSED_DATA


def f(x):
    return 1 / (1 + np.exp(-x))


def to_corners(boxes):
    corner_boxes = np.zeros((boxes.shape[0], 4), dtype=int)
    for i in range(len(boxes)):
        corner_boxes[i][0] = boxes[i][0]  # x1
        corner_boxes[i][1] = boxes[i][1]  # y1            
        corner_boxes[i][2] = boxes[i][0] + boxes[i][2]  # x2
        corner_boxes[i][3] = boxes[i][1] + boxes[i][3]  # y2
    return corner_boxes


def extract_labels(dataset="train"):
    path_to_data = PATH_TO_IMAGES[dataset] + "/_annotations.coco.json"
    with open(path_to_data, "r") as f:
        data = json.load(f)
    df_boxes = pd.DataFrame(data['annotations']) \
        .set_index("id") \
        .drop(columns=["segmentation", "iscrowd"])
    df_images = pd.DataFrame(data['images']) \
        .drop(columns=['height', 'width', 'date_captured', "license"]) \
        .rename(columns={"id": "image_id"})
    df = df_boxes.merge(df_images, on='image_id', how='inner') \
        .rename(columns={"category_id": "is_occupied"})
    df["is_occupied"] -= 1
    return df


def extract_images(labels_df, size=100, new_img_size=96,
                   save=True, path_to_save=PATH_TO_PREPROCESSED_DATA,
                   dataset='train', custom_indices=None, output_step=0.1):
    """
  description:
  - This function take batch of images and extract crop images of parking lots.

  parameters:
  - labels_df: dataset with coordinates of boxes and images file.
  - size: size of batch with images from which to extract crop images
  - new_img_size: new resolution of image new_img_size*new_img_size
  - dataset: from which dataset to take images ("train", "valid", "test")
  - save: boolean, save results on disk or not
  - path_to_save: where to save results
  - custom_indices: indices of image for which extract crop images
  - output_step: printing and saving progress after each "output_step" size part is done

  return:
  - batch of images in form of numpy array
  - target labels for images
  """
    if custom_indices:
        img_id_set = custom_indices
    else:
        img_id_set = np.random.choice(labels_df.image_id.unique(), size, replace=False)
    images = np.empty((1, new_img_size, new_img_size, 3), dtype=np.uint8)
    target = np.empty(1, dtype=np.uint8)
    progress_counter = 0

    for img_id in img_id_set:
        progress_counter += 1
        if progress_counter % int(output_step * size) == 0:
            print(f"\33[30mProgress: {int(100 * progress_counter / size)}%")
            print(f"\33[30mTrain images shape: {images.shape}")
            print(f"\33[30mTrain target shape: {target.shape}")

        sample = labels_df.loc[labels_df.image_id == img_id]
        boxes = sample.bbox.to_numpy()
        boxes = to_corners(boxes)
        occupance = sample.is_occupied.to_numpy()
        target = np.concatenate((target, occupance), axis=0)
        img_name = sample.file_name.iloc[0]
        img = cv.imread(os.path.join(PATH_TO_IMAGES[dataset], img_name))

        for i in range(boxes.shape[0]):
            crop_img = img[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2], :]
            res_img = cv.resize(crop_img, (new_img_size, new_img_size))
            res_img_exp = np.expand_dims(res_img, axis=0)
            images = np.concatenate((images, res_img_exp), axis=0)

        if save and (progress_counter % int(output_step * size) == 0):
            np.save(os.path.join(path_to_save, f'{dataset}_images.npy'), images)
            np.save(os.path.join(path_to_save, f'{dataset}_target.npy'), target)
            print(f"\33[32mImages and target data succesfully saved in: {path_to_save}")

    return images, target


def predict(model, df, path, new_img_size=96, threshold=0.5):
    img_name = df.file_name.iloc[0]
    img = cv.imread(os.path.join(path, img_name))
    boxes = df.bbox.to_numpy()
    boxes = to_corners(boxes)
    images = np.empty((0, new_img_size, new_img_size, 3))

    for i in range(boxes.shape[0]):
        boxes[i] = np.maximum(boxes[i], 0)
        crop_img = img[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2], :]
        res_img = cv.resize(crop_img, (new_img_size, new_img_size))
        res_img_exp = np.expand_dims(res_img, axis=0)
        images = np.concatenate((images, res_img_exp), axis=0)

    pred = model(images, training=False).numpy().flatten()
    # print(np.round(f(pred), 4))
    pred = np.where(f(pred) >= threshold, 1, 0)
    for i in range(len(boxes)):
        box = boxes[i]
        if pred[i] == 1:
            cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), img.shape[0] // 300)
        elif pred[i] == 0:
            cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), img.shape[0] // 300)

    return img, (pred == 0).sum(), len(pred)
