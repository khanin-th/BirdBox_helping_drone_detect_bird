# BirdBox_helping_drone_detect_bird
Detect bird given image 

## Metrics
1. __False Negative__ is important, because being blind to bird when it is present can harm the drone. The process will aim to reduce __False Negative__, and this essentially leads to high __Recall__, which can be used to compare models (higher __Recall__ is better)

## Base Architecture:
Light weights pre-trained object detection models are considered. The initlial screening was ranking the models based on Speed (ms) in ascending order.
1. SSD MobileNet v2 320x320
2. CenterNet MobileNetV2 FPN 512x512
3. EfficientDet D0 512x512

Source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

## Training Data:
[NABirds](https://dl.allaboutbirds.org/nabirds) data set provided by the [Cornell Lab of Ornithology](https://www.birds.cornell.edu/home)

## Future Work:
1. Include samples WITHOUT bird for training the model to recognize the cases bird are not present.
2. Have samples of bird flapping wings which are better samples for the sitation where drone will encounter birds in operation.
3. Use the detected area to predict most likely next movement of the bird, i.e. is the bird flying toward drone or flying away from drone.
