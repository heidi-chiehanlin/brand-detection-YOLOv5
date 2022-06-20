# Vehicle logo detection based on YOLOv5 model
*Created by Chieh-An (Heidi) Lin, Feb. 2022.*


## Project Summary
This project focus on car brand recognition. 18 Brands, each with 50 training images were provided.
The complete solution can be found in the [training notebook](/Logo_Recognition_Customized_YOLOv5.ipynb), with the steps listed below:
1. Data Preperation: Using [makesense.ai](https://www.makesense.ai/) to create multiple labels on image.
2. Data Spliting: Assigning images to train/validate/test dataset and organize them in according directories.
3. Model Training: Fitting training images to `YOLOv5s`
4. Model Evaluation: Finding the weights that gives the best accuracy in testing dataset
5. Insights & Observation: [W&B analysis report](https://wandb.ai/heidi-chiehanlin/YOLOv5/reports/Logo-Detection-of-18-Car-Brands--VmlldzoxNTI4NDQ2?accessToken=jtupqhnc9mqci10knaq5mq09pdbl6dl5gdi5sbiavmat4s1tq0hy7i2j26k9ogm7)
<br><br>

## Breif Introduction to YOLO v5

YOLO is a family of object detection architectures and models pretrained on the COCO dataset, famous for its speed and accuracy.
[Glenn Jocher](https://www.linkedin.com/in/glenn-jocher/) introduced PyTorch based version of YOLO and released YOLO v5 ([Github](https://github.com/ultralytics/yolov5)/[Documentation](https://docs.ultralytics.com/)).

![architecture](/The-network-architecture-of-Yolov5.png)

[image source](https://www.researchgate.net/publication/349299852_A_Forest_Fire_Detection_System_Based_on_Ensemble_Learning)

As a **single-stage** object detector, it has three important parts:
1. Backbone: CSPNet. Extracts features from given input images.
2. Neck: PANet. Generates feature pyramids to help model generalize well on objects with different size and scales.
3. Head: Performs final predictions. Three outputs are generated for detection of different sizes.

As for the activation function, YOLO v5 uses Leaky ReLU and Sigmoid activation function.
([reference](https://towardsai.net/p/computer-vision/yolo-v5%E2%80%8A-%E2%80%8Aexplained-and-demystified))
<br><br>

## Model Training
Notebook can be found here: [training customized YOLOv5](/Logo_Recognition_Customized_YOLOv5.ipynb).
Or run it on Google Colab: [link to training notebook on Google Colab.](https://colab.research.google.com/drive/1--XIoBcOkmj8pB_MUB0pXlh6zi3W-OiC?usp=sharing)
<br><br>

## Performance Evaluation
Model logging was done on Weights and Biases. [link to performance report.](https://wandb.ai/heidi-chiehanlin/YOLOv5/reports/Logo-Detection-of-18-Car-Brands--VmlldzoxNTI4NDQ2?accessToken=jtupqhnc9mqci10knaq5mq09pdbl6dl5gdi5sbiavmat4s1tq0hy7i2j26k9ogm7)
<br><br>

## Discovery & Future Improvement

#### 1. Dataset quality
- Increasing images & instances per class.
- Increasing the variety of images, i.e. logos of different angles/orientation, logos on a car from different lighting/shades/backgrounds... etc.

#### 2. Train
In this project, I choose to train custom data on pretrained `yolov5s` with 50 epochs.
Without limited time and computation power,it might be worth to try:

- Using [larger pretrained models](https://github.com/ultralytics/yolov5#pretrained-checkpoints)
- More epochs
- Training from scratch (starting with 300 epochs)


#### 3. Metrics
> metrics don't capture the whole story is that these evaluations are done for fixed thresholds for class id probability, IOU, and objectness confidence, all of which can be tuned for each model to get very different results. ([source](https://wandb.ai/cayush/yoloV5/reports/How-are-your-YOLOv5-models-doing---VmlldzoyNjM3MTY))

Defining metrics that aligns with real world business goal is necessary and it's the direction for model iteration and optimization.
Which one is more critical to the problem we are solving, precision, recall, or mAP? If none of them are good enough, then it's time to design our customized metric.
<br><br>


## Appendix: Terminology

**Precision Recall Curve**

$$Precision: \frac{TP}{(TP+FP)} = \frac{TP}{total\ positive\ predictions}$$
$$Recall: \frac{TP}{(TP+FN)} = \frac{TP}{total\ true\ cases}$$
<br><br>

**IoU (Intersection over Union)**

The amount of overlap between ground truth box & prediction box. $$IoU = \frac{area\ of\ overlap}{area\ of\ union}$$

Increasing IoU requires model to make closer prediction to the true bounding box;
Then, with higher IoU (more overlap), it's easier for model to be considered correct.
<br><br>

**AP (Average Precision)**

AP is the weighted mean of precisions achieved at various recall points; It's the area under the PR curve.
([reference](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173))
<br><br>

**mAP (Mean Average Precision)**

> In YOLOv5, mAP is the average AP over multiple IoU, which makes it an comprehensive performance metric.
A better mAP indicates that the model is doing better in every sense. ([reference](https://blog.roboflow.com/mean-average-precision/))

**mAP@[.5, .95]** = average mAP over different IoU thresholds, from 0.5 to 0.95, step 0.05 (0.5, 0.55, 0.6, ..., 0.95).
