# ParkingLotDetection
A system that uses computer vision to identify free parking spots in real-time footage from parking lot cameras.

![image](https://user-images.githubusercontent.com/100434509/230745113-0bd43ab7-1209-4a3a-a5ea-0424f49a9139.png)

## Overview

This project leverages a custom image classifier built on MobileNetv2 to detect available parking spaces efficiently. It's designed to enhance parking management by minimizing the time users spend searching for parking, thereby improving the overall user experience. The PKlot dataset was used for training purposes.

## Getting Started

### Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.7+

### Installation

Clone the project and install the required libraries:

```bash
git clone https://github.com/ernestknurov/ParkingLotDetection.git
cd ParkingLotDetection
pip install numpy pandas opencv-python tensorflow 
```

### Usage

1. Mark parking lots manually using `polygon_selector.py`.
2. For detection, use the `detect.py` script. Ensure you have parking lot markup ready.
3. A tutorial on how to use the application is provided in `tutorial.ipynb`.

```bash
# Example for running detect.py
python detect.py path/to/your/markup.pkl --image_folder path/to/your/images --output_path path/to/save/output.jpg --model_path path/to/your/model
```
