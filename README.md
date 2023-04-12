# ParkingLotDetection
![image](https://user-images.githubusercontent.com/100434509/230745113-0bd43ab7-1209-4a3a-a5ea-0424f49a9139.png)

## Description
This project utilizes computer vision techniques to detect and locate free parking spots in a parking lot. The system is designed to analyze real-time footage from a camera installed in the parking lot, process the images, and identify available parking spots. The project aims to maximize detection accuracy, reasonably reduce the time spent detecting parking spots, improve the efficiency of parking management, and ultimately enhance the user experience.

## Project Structure
In this project, a custom image classifier based on MobileNetv2 was utilized to reduce computational cost. Parking lots are manually marked up using the script **polygon_selector.py**. An example of using this application is provided in the **tutorial.ipynb** file. Use **detect.py** script to process your images (you need markup of parking lots first!). For training was used PKlot dataset.
