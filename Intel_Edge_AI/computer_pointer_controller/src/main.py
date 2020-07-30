
import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys
import logging as log
from argparse import ArgumentParser

from face_detection import FaceDetect
from landmarks_detection import FacialLandmarks
from gaze_estimation import GazeEstimate
from head_pose_estimation import HeadposeEstimate

from mouse_controller import MouseController


CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file. Use CAM to use webcam stream")
    

    parser.add_argument("-fd", "--facedetector", required=False, type=str, default=None,
                        help="Path to an xml file with a trained face detector model.")
    parser.add_argument("-fl", "--facelandmark", required=False, type=str, default=None,
                        help="Path to an xml file with a trained face landmarks detector model.")
    parser.add_argument("-hp", "--headpose", required=False, type=str, default=None,
                        help="Path to an xml file with a trained head pose detector model.")
    parser.add_argument("-ge", "--gaze", required=False, type=str, default=None,
                        help="Path to an xml file with a trained gaze detector model.")

                                                       
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")

    return parser

def main():
    log.basicConfig(filename='log_stats.log',level=log.DEBUG)
    args = build_argparser().parse_args()
    model_fd = args.facedetector
    model_fl = args.facelandmark
    model_hp = args.headpose
    model_ge = args.gaze
    device = args.device
    video_file = args.input    
    threshold = args.threshold

    start_model_load_time=time.time()
    
    fd = FaceDetect(model_fd, device, threshold)
    fd.load_model()
    fl = FacialLandmarks(model_fl, device)
    fl.load_model()
    hp = HeadposeEstimate(model_hp, device)
    hp.load_model()
    ge = GazeEstimate(model_ge, device)
    ge.load_model()
    
    total_model_load_time = time.time() - start_model_load_time

    if args.input == 'CAM':
        input_stream = cv2.VideoCapture(0)

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "input file doesn't exist"
        
    cap = cv2.VideoCapture(input_stream)
    
    if input_stream != 0 :
        cap.open(input_stream)
    else : 
        log.error("ERROR! Unable to open video source")
        
    width = int(cap.get(3))
    height = int(cap.get(4))    
    
    frame_counter=0
    start_inference_time=time.time()
    total_time = 0
    try:
        while(cap.isOpened()):        
            
            ### TODO: Read from the video capture ###
            ret, frame = cap.read()
            if(ret == False):
                break
            else :    
                frame_counter = frame_counter+1
                coords, cropped_image = fd.predict(frame)

                pitch, rolls, yaw =hp.predict(cropped_image)
                #yaw, pitch, rolls = fl.predict(image)
                # log.info("yaw")
                # log.info(yaw)
                # log.info("pitch")
                # log.info(pitch)
                # log.info("rolls")
                # log.info(rolls)
                left_eye, right_eye,out_image = fl.predict(cropped_image)
                #log.info(image)
                # log.info("eye1")
                # log.info(left_eye)
                # log.info("eye2")
                # log.info(right_eye)
                #log.
                x,y, z = ge.predict(left_eye,right_eye,[yaw, pitch, rolls],out_image)
                # log.info("mouse_coor")
                # log.info(x)
                # log.info("gaze_vec")
                # log.info(y)
                total_time = total_time + time.time() - start_inference_time
                try :
                    cv2.imshow('output', frame)  
                except cv2.error as e:  
                    print('Invalid frame!')
                if cv2.waitKey(50) & 0xFF == ord('q'):
                    break  
                mc = MouseController('high','fast')
                mc.move(x,y)
                

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
   
        cap.release()
        cv2.destroyAllWindows()

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = frame_counter / total_inference_time    
        log.info('Total Model Load Time = ' + str(total_model_load_time))
        log.info('Total Inference Time = ' + str(total_inference_time))
        log.info('FPS = ' + str(fps))
    except Exception as e:
        log.error("Could not run Inference: ", e)

if __name__ == '__main__':
    
    main()