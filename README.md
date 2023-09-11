# Real-Time Car Counting on Highways using YOLOv8n and SORT Algorithm with Region Masking

## Introduction

This repository showcases an implementation of the YOLOv8n (You Only Look Once Version 8 Nano) object detection framework in conjunction with the SORT (Simple Online and Realtime Tracking) algorithm to enable real-time car counting on highways. Additionally, a region masking technique has been developed to specify an area of interest, thus enhancing the model's accuracy in car counting.

## Objectives

The primary objectives of this project are as follows:

1. **Real-Time Car Counting**: Implement YOLOv8n to detect and track cars in real-time video streams from highway cameras.
2. **SORT Algorithm Integration**: Integrate the SORT algorithm for efficient tracking of detected cars across frames, providing continuous vehicle count updates.
3. **Region Masking**: Develop a masking mechanism to isolate and concentrate the model's focus on a predefined region of interest within the highway scene.
4. **Accuracy Enhancement**: Improve the accuracy of car counting by narrowing down the model's attention to the specified region and mitigating false positives.

## Implementation Details

### YOLOv8n

YOLOv8n is a lightweight variant of the YOLO (You Only Look Once) object detection framework optimized for real-time inference on resource-constrained devices. It strikes a balance between accuracy and speed, making it ideal for applications like highway car counting.

### SORT Algorithm

The SORT (Simple Online and Realtime Tracking) algorithm is used for car tracking. This algorithm adeptly handles the complexities of object tracking in real-time video streams, ensuring continuous monitoring of vehicles as they move within the camera's field of view.

### Region Masking

A region masking technique has been devised to define a specific area of interest within the highway scene. This empowers the model to focus its detection and tracking efforts solely on the target region, thereby improving the precision of car counting while minimizing interference from irrelevant objects or background elements.

## Results

Extensive testing and evaluation of this implementation on highway footage from diverse sources consistently demonstrate the effectiveness of the YOLOv8n and SORT combination, along with the region masking approach, in accurately counting cars in real-time scenarios.

## Conclusion

This GitHub repository serves as a comprehensive solution for real-time car counting on highways, harnessing the power of the YOLOv8n object detection framework, the SORT tracking algorithm, and an intelligent region masking strategy to enhance model accuracy. It stands as a valuable resource for researchers and practitioners interested in traffic analysis and surveillance applications.

For in-depth details and usage instructions, please consult the documentation and code provided in this repository.

**Note**: When deploying this technology for real-world applications, especially in surveillance and traffic monitoring contexts, ensure compliance with all relevant laws and regulations.
