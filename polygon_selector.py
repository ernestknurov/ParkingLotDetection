import cv2
import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="Polygon selector script",
                                 epilog="""- Click left button of mouse and pull for drawing a rectangle.
                                           \n- Type 'd' to delete last rectangle.\n- Type 'q' to save and exit.""")
parser.add_argument('file_directory', type=str, help='Directory in which file is located')
parser.add_argument('file_name', type=str, help='Name of file')
parser.add_argument("--output_path", type=str, default="test.pkl", help="Path where polygons will be saved")
args = parser.parse_args()

# Load the image file
path = args.file_directory
img_name = args.file_name
full_path = os.path.join(path, img_name)
img = cv2.imread(full_path)

# Create a window to display the image in
# cv2.namedWindow("image")
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 1200, 800)
cv2.imshow("image", img)


# scale_factor = img.shape[1] / 1200.0

# Define a function to adjust the rectangle coordinates
def adjust_rect(rect, factor):
    return [int(x / factor) for x in rect]

# Define a dataframe to hold the rectangles
rectangles = pd.DataFrame(columns=['bbox', 'file_name'])

# Define a variable to keep track of the current rectangle
current_rect = None


# Define a function to be called whenever the mouse is clicked
def mouse_callback(event, x, y, flags, params):
    global current_rect

    # If the left mouse button is clicked, start a new rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        current_rect = [x, y]

    # If the left mouse button is released, finish the rectangle
    elif event == cv2.EVENT_LBUTTONUP:
        x1, y1 = current_rect
        x2, y2 = x, y
        # Create a rectangle from the starting and ending points
        rect = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
        # rect = adjust_rect(rect, scale_factor)
        rectangles.loc[rectangles.shape[0]] = [rect, img_name]
        # Draw the rectangle on the image
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (96, 96, 96), img.shape[0]//300)
        cv2.imshow("image", img)

    # If the right mouse button is clicked, cancel the current rectangle
    elif event == cv2.EVENT_RBUTTONDOWN:
        current_rect = None


# Set the mouse callback function for the window
cv2.setMouseCallback("image", mouse_callback)

# Wait for the user to finish selecting rectangles
while True:
    cv2.imshow("image", img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("d"):
        # Delete the last drawn rectangle
        if len(rectangles) > 0:
            rectangles.drop(index=rectangles.shape[0]-1, inplace=True)
        img = cv2.imread(full_path)
        for rect in rectangles.bbox.to_numpy():
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (96, 96, 96), img.shape[0]//300)

# Print the coordinates of the rectangles
rectangles.to_pickle(args.output_path)

# Close the window and exit
cv2.destroyAllWindows()


