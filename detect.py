import argparse
import pandas as pd
import cv2 as cv
import tensorflow as tf
from utils import predict

parser = argparse.ArgumentParser(description="This script detect free spaces on parking lot")
parser.add_argument('markup_path', type=str, help='Path to the markup of parking lots')
parser.add_argument("--image_folder", type=str, default="custom_tests", help="Folder in which image is placed")
parser.add_argument("--output_path", type=str, default="custom_tests/output.jpg", help="Path where output image will be saved")
parser.add_argument("--model_path", type=str, default="saved_models/my_model", help="Path to model")
args = parser.parse_args()

# Load the image file
labels = pd.read_pickle(args.markup_path)

# Detect free spaces
model = tf.keras.models.load_model(args.model_path)
img, free_lots, all_lots = predict(model, labels, path=args.image_folder, threshold=0.06)

text = f"Lots free: {free_lots}/{all_lots}"

fontScale = img.shape[1]//800
color = (0, 255, 255)  # yellow
thickness = img.shape[1]//300
org = (0, fontScale*30)

img = cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, fontScale,
                 color, thickness, cv.LINE_AA)

cv.imwrite(args.output_path, img)
