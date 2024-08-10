# Yolo-V5-glass-cup-detection

## Aim and Objectives

## Aim

 Aim of this project is to develop a robust and efficient object detection model capable of accurately identifying and classifying glasses and cups in various environments using the YOLOv5 architecture.

## Objectives:


•	Collect and preprocess a diverse dataset of images containing glasses and cups.
•	Implement the YOLOv5 object detection model for the task.
•	Train the model using the collected dataset.
•	Evaluate the model's performance and fine-tune it for optimal accuracy.
•	Deploy the model for real-time detection applications.

## Abstract

This project presents a deep learning-based approach for detecting glasses and cups in images using the YOLOv5 architecture. The model is trained on a dataset curated from various sources, ensuring a wide range of scenarios and variations. The trained model demonstrates high accuracy and efficiency, making it suitable for applications in smart kitchens, inventory management, and automated cleaning systems.

## Introduction

Object detection has become a pivotal component in various computer vision applications. Detecting everyday objects such as glasses and cups can enhance the functionality of automated systems in homes and industries. This project leverages the YOLOv5 model, known for its speed and accuracy, to achieve reliable glass and cup detection.

## Literature Review

Recent advancements in object detection models, particularly YOLO (You Only Look Once) series, have revolutionized real-time object detection tasks. YOLOv5, the latest in the series, offers significant improvements in terms of speed and accuracy compared to its predecessors. This section reviews various object detection techniques and the evolution of the YOLO architecture.

## Methodology

The methodology involves several steps:
1. **Data Collection and Preprocessing:** Gather images containing glasses and cups from various sources. Annotate the images for object detection tasks.
2. **Model Selection:** Choose the YOLOv5 architecture due to its balance of speed and accuracy.
3. **Training:** Train the YOLOv5 model using the preprocessed dataset. Use data augmentation techniques to improve model robustness.
4. **Evaluation:** Evaluate the model's performance on a validation set and fine-tune hyperparameters for optimal results.
5. **Deployment:** Deploy the trained model for real-time detection tasks.

## Installation

Follow these steps to set up the environment and install the necessary dependencies:

1. Clone the YOLOv5 repository:
    ```sh
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    ```

2. Install the required dependencies:
    ```sh
    pip install -qr requirements.txt
    pip install roboflow
    ```

## Running the glass-cup detection Model

1. Set up the dataset environment variable:
    ```python
    import os
    os.environ["DATASET_DIRECTORY"] = "/content/datasets"
    ```

2. Download the dataset using Roboflow:
    ```python
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("sparrow-ai").project("cup-_glass-data")
    dataset = project.version(1).download("yolov5")
    ```

3. Train the YOLOv5 model:
    ```sh
    python train.py --img 640 --batch 16 --epochs 100 --data {path_to_dataset}/data.yaml --weights yolov5s.pt
    ```
## Demo 




https://github.com/pooja-anil-thakur/glass-cup-detection/assets/173771593/d9b07e32-8f69-40b4-8cb9-57d3965a5729

### Link :- https://youtu.be/hHIPhJS_aEc

## Advantages

- **High Accuracy:** The YOLOv5 model provides precise detection of glasses and cups.
- **Real-Time Detection:** Capable of processing images and videos in real-time.
- **Scalability:** Suitable for various applications, from small home setups to large industrial systems.

## Applications

- **Smart Kitchens:** Automated detection of glasses and cups for inventory management.
- **Robotic Systems:** Integration with robots for automated cleaning and organization.
- **Retail:** Enhanced inventory tracking and management in retail environments.

## Future Scope

- **Improved Accuracy:** Further fine-tuning the model and incorporating more diverse datasets.
- **Edge Deployment:** Optimizing the model for deployment on edge devices.
- **Integration with Other Systems:** Combining with other AI systems for enhanced automation

## Conclusion

This project successfully demonstrates the potential of the YOLOv5 architecture for detecting glasses and cups. With high accuracy and real-time performance, the model is suitable for various practical applications, offering significant benefits in automation and efficiency.



## References
1] Roboflow:- https://roboflow.com/

2] Google images





