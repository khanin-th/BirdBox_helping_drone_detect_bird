# BirdBox_helping_drone_detect_bird
Detect bird given image 

## Metrics
1. __False Negative__ is important, because being blind to bird when it is present can harm the drone. The process will aim to reduce __False Negative__, and this essentially leads to high __Recall__, which can be used to compare models (higher __Recall__ is better)

## Models:
1. SSD Mobilenet
2. Center Mobilenet
3. EfficientDet
 
## Future Work:
1. Include samples without bird for training the model to recognize the cases bird are not present.
2. Use the detected area to predict most likely next movement of the bird, i.e. is the bird flying toward drone or flying away from drone.
