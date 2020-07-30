# Computer Pointer Controller

This project uses a gaze detection model to control the mouse pointer of the computer. Gaze Estimation model is used to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. Following figure shows the flow of the project: \
<img src="bin/pipeline.png" alt="Computer Pointer Controller Flow Chart"/>

## Project Set Up and Installation
This project is implemented on __Ubuntu 16.04 LTS__ using [OpenVINO Toolkit 2019_R3](https://docs.openvinotoolkit.org/2019_R3/_docs_install_guides_installing_openvino_linux.html).\

The model required for this project can downloaded from OpenVino Zoo using the below commands:
```
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name face-detection-adas-binary-0001  
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name gaze-estimation-adas-0002 
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name head-pose-estimation-adas-0001  
python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name landmarks-regression-retail-0009
```
Following is the directory structure for the project :
```
project
│   README.md
│   requirments.txt    
│
|---bin
│     demo.mp4
│   
|---models  
│   │---  face-detection-adas-0001
│   │---  gaze-estimation-adas-0002
│   │---  head-pose-estimation-adas-0001
│   |---  landmarks-regression-retail-0009
|
|---src
|     face_detection.py
|     gaze_estimation.py
|     head_pose_estimation.py
|     landmarks_detection.py
|     main.py
|     mouse_controller.py
```
After installing the OpenVINO software, it is important to initilize the following command:
```
source /opt/intel/openvino/bin/setupvars.sh
```
Create a virtual environement for this project using the following commands :
```
virtualenv -p python3.6 comp_pointer_env
source comp_pointer_env/bin/activate
pip install -r requirements.txt
```


## Demo

Run the below commands to run the demo.
1. FP32 precision
```
python src/main.py -i bin/demo.mp4 -d CPU -pt 0.4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -fd models/face-detection-adas-0001/FP32/face-detection-adas-0001 -fl models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 -hp models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 -ge models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002
```

2. FP16 precision
```
python src/main.py -i bin/demo.mp4 -d CPU -pt 0.4 -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -fd models/face-detection-adas-0001/FP16/face-detection-adas-0001 -fl models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 -hp models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -ge models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002
```


## Benchmarks

__FP32:__\
  Total Model Load Time = 0.3403s\
  Total Inference Time = 753.7s\
  FPS = 0.7894s

__FP16:__\
Total Model Load Time = 0.3703s\
Total Inference Time = 753.8s\
FPS = 0.7893s

## Results
Significant difference is not there between FP32 and FP16 interms of inference time or loading time. We could probably see more differences if we use INT8 models. Since INT8 models were not available for all the models used in this project, comparision is not shown here. Usually low precision models occupy less space and hence less computionally intensive. This in turn leads to less inference time.