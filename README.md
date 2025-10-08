AI-Powered Virtual Surveyor for Infrastructure Mapping


![alt text](https://img.shields.io/badge/Status-Completed-brightgreen)
![alt text](https://img.shields.io/badge/License-MIT-green)

This project is an end-to-end computer vision and geospatial analysis pipeline that automates the process of mapping physical infrastructure. It functions as a "virtual surveyor," programmatically navigating a predefined route, visually identifying target objects using a custom-trained AI model, and calculating their real-world GPS coordinates using data from the Google Street View API.

The system is designed to replace the slow, expensive, and manual process of physical asset surveying with an automated, scalable, and remote alternative.

Table of Contents
Project Overview
Core Features
Project Pipeline
Technical Deep Dive
Technical Stack
Setup & Installation
Usage
Project Roadmap


Manually surveying and mapping assets like utility poles, fire hydrants, or cell towers is a significant operational challenge. This project was built to solve that problem by leveraging three key technologies:

Geospatial Mathematics: To programmatically generate a high-density "scan path" along any given road or route.

Google Street View API: To act as the "eyes" of the system, providing a rich, visual dataset of the real world from a driver's perspective.

Custom Object Detection: A fine-tuned DETR (DEtection TRansformer) model that acts as the "brain," capable of identifying specific target objects within the Street View imagery with high accuracy.

By combining these, the pipeline can take a simple route drawn on Google Maps and produce a detailed CSV file containing the precise predicted latitude and longitude of every utility pole found along that route.

Core Features

Automated Route Interpolation: Generates hundreds of precise GPS waypoints every ~10 meters along a given path using geodesic calculations.

Dynamic Image Acquisition: Systematically calls the Google Street View API at each waypoint, adjusting the camera's heading to scan the environment intelligently.

Custom AI-Powered Detection: Utilizes a custom-trained Hugging Face DETR model for highly accurate object detection (this repository is configured to find utility poles).

Geospatial Triangulation: Implements a novel algorithm to convert a 2D bounding box from an image into a real-world GPS coordinate by estimating distance and angle from the camera.

Scalable Pipeline: The entire process is scripted in Python, allowing for the analysis of long routes and large areas with minimal manual intervention.

Project Pipeline

The project is structured as a multi-stage data processing pipeline. The scripts are designed to be run in sequence:

Data Ingestion & Cleaning (regex_sep_waypoint.py):

Input: A raw, messy CSV file exported from a hand-drawn route in Google Maps.

Process: Uses regular expressions to parse the unstructured text and extract a clean list of primary latitude/longitude waypoints.

Output: A clean CSV (cleaned_google_waypoints_demo_1.csv) that serves as the input for the next stage.

Route Interpolation (way_point_create.py):

Input: The clean list of primary waypoints.

Process: Calculates the geodesic path (distance and bearing) between each waypoint. It then interpolates new GPS coordinates at ~10-meter intervals to create a high-resolution scan path.

Output: A detailed CSV (demo_1_cleaned_way_point_test_florid.csv) containing hundreds of sequential scan points.

Survey & Detection (deploy_model.py / long_test.py):

Input: The high-resolution scan path.

Process: This is the core engine. It iterates through each scan point, calls the Google Street View API, runs the object detection model on the resulting image, and performs the geospatial triangulation for any detected objects.

Output: The final dataset (predicted_lat_and_long.csv), containing the timestamp and predicted GPS coordinates of each successfully identified asset.

(Note: model_build_first.py contains the script for training the object detection model, and long_test.py is a detailed version of the deployment script used for calibration and testing different parameters.)

Technical Deep Dive
Object-to-GPS Triangulation

The most complex part of this project is converting a detected object's bounding box into a GPS coordinate. This is achieved through a multi-step calculation within deploy_model.py:

Distance Estimation: The distance to the object is approximated using a pinhole camera model formula: Distance = (Focal Length * Real Object Height) / Object Height in Pixels. This requires calibrating an estimate for the focal length of Google's cameras and the average height of the target object. A multiplier (d_multiply) is used for fine-tuning.

Angle Calculation: The camera's direction of travel (heading) is known from the geodesic calculation. The horizontal position of the bounding box's center relative to the image's center gives an angular offset. The final bearing to the object is heading + angular_offset.

Coordinate Projection: With a starting point (the camera's GPS), a distance, and a bearing, the Haversine formula is used to project the final latitude and longitude of the detected object.

code
Python
download
content_copy
expand_less
# A simplified look at the core triangulation logic
# 1. Estimate Distance
distance = (google_camera_focal_length * pic_height) / object_height_in_pixels
calibrated_distance = distance * d_multiply # Apply calibration factor

# 2. Calculate Angle
angle_from_camera = ((box_center_x - 320) * PIXELS_TO_DEGREES_RATIO) + camera_heading

# 3. Project New Coordinate using Haversine formula
# ... (complex math using geographiclib or similar) ...
lat_end = degrees(lat2)
long_end = degrees(lon2)
Technical Stack

Language: Python 3.x

AI / Machine Learning:

PyTorch: The backend for the deep learning model.

Hugging Face Transformers: For training and deploying the DETR object detection model.

Geospatial Calculations:

geographiclib: For accurate geodesic (distance and bearing) calculations on the WGS84 ellipsoid.

math, numpy: For the core trigonometric and mathematical operations.

Data Handling:

pandas: For managing and exporting data to CSV files.

Pillow (PIL): For image manipulation and drawing bounding boxes.

APIs & Web:

requests: For making HTTP requests to the Google Street View API.

Setup & Installation

Clone the Repository:

code
Bash
download
content_copy
expand_less
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Set Up a Virtual Environment (Recommended):

code
Bash
download
content_copy
expand_less
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Dependencies:
It is highly recommended to install PyTorch first, following the official instructions for your specific hardware (CPU or CUDA): https://pytorch.org/get-started/locally/

Then, install the rest of the libraries:

code
Bash
download
content_copy
expand_less
pip install transformers Pillow pandas geographiclib requests

Get a Google Maps API Key:

You will need a Google Cloud Platform account with the Street View Static API enabled.

Generate an API key from the GCP console.

Open deploy_model.py and long_test.py and replace the placeholder api_key variable with your actual key.

Download/Train the AI Model:

This repository does not include the final trained model files.

You must train your own model using the model_build_first.py script and your own annotated dataset.

Once trained, update the paths in deploy_model.py and long_test.py to point to your model's checkpoint directory.

Usage

Create and Clean Your Route:

Go to Google My Maps and draw a route in your area of interest.

Export the route as a KML/KMZ file, then convert it to a CSV.

Place this file in the root directory (e.g., as Untitled map- demo_2.csv).

Run the cleaning script: python regex_sep_waypoint.py.

Generate the Scan Path:

Run the interpolation script: python way_point_create.py. This will create the detailed waypoint file.

Deploy the Surveyor:

Run the main deployment script: python deploy_model.py (or long_test.py for more detailed output and parameter tuning).

The script will begin iterating through the waypoints, downloading images, and performing inference. Detected object locations will be saved to the final output CSV.

Project Roadmap

Front-End Interface: Build a simple web UI using Streamlit or Flask to allow users to upload a route file, monitor progress, and view the final results on an interactive map.

Multi-Class Object Detection: Expand the training dataset and model to detect other types of infrastructure (e.g., fire hydrants, traffic lights, manhole covers).

Improved Distance Triangulation: Implement a stereo vision approach by taking two Street View images from slightly different positions to calculate depth and distance more accurately.

Database Integration: Store the results in a PostGIS-enabled database to allow for more complex spatial queries and analysis.
