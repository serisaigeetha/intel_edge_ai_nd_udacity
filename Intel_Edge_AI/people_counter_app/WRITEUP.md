# Project Write-Up

## Introduction

The model used for this project is faster_rcnn_inception_v2_coco, selected from Tensorflow Object Detection Model Zoo.

## Explaining Custom Layers

The Intel OpenVINO toolkit supports neural network model layers in multiple frameworks including TensorFlow*, Caffe*, MXNet*, Kaldi* and ONYX*. A list of known layers is available for each of the frameworks. The layers which are not part of this list are custom layers and hence Model Optimizer classifies them as custom.

Inference Engine needs to know the targeted device so that it perform inference accordingly. IE uses device specific plugins where each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. 

Custom Layer CPU Extension is a a compiled shared library (.so or .dll binary) needed by the CPU Plugin for executing the custom layer on the CPU. CPU extensions can be found in opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/.

Following member functions are used from IECore Class reference to handle the custom layers :

1. query_network() : Queries the plugin with specified device name what network layers are supported in the current configuration

2. add_extension() : Loads extension library to the plugin with a specified device name.

 
The main reason for handling custom layers is to generate a valid Intermediate Representation.


## Comparing Model Performance

The size of the model pre-coversion was 55MB and post-conversion was 52MB. Not too much difference here.
The inference time of the model pre-conversion was 112ms and post-conversion was 9.70ms. There is a significant improvement is speed post-conversion using OpenVINO toolkit.

## Assess Model Use Cases

Some of the potential use cases are:
1. During the current situation of pandemic, most of the places like grocery stores, universites etc are limiting the number of people present inside a closed space. People counter app can used to alert the authorities if number of people exceed a threshold.
2. Can be used to check for burglary in Museums during night time/non-visiting hours where high value paintings/artifacts are stored. 

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...
1. If the lighting is not enough to detect a person, we may miss alerts in some use cases such as burglary.
2. If the accuracy is not good enough, all the statistics collected will be wrong and hence it will lead to loss of information.
3. Camera focal length/image size will help in getting more accurate results by having giving quality inputs.


