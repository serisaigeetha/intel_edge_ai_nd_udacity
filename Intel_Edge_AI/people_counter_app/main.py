"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
from datetime import datetime
import socket
import json
import cv2
import time 

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
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
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
 
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
   
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.cpu_extension , args.device)
    
    
    net_input_shape = infer_network.get_input_shape()
    
    
    #print(net_input_shape)
    
    ### TODO: Handle the input stream ###
     # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0

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
    
   
        
    total_count = 0
    ref_time = datetime.now()
    last_detect_count = 0
    current_time = 0
    frame_count_left = 0
    frame_count_enter = 0
    ### TODO: Loop until stream is over ###
    while(cap.isOpened()):        
        
        ### TODO: Read from the video capture ###
        ret, frame = cap.read()
        if(ret == False):
           break
        else :    
          #  print(frame.shape)
           
        ### TODO: Pre-process the image as needed ###
            p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)
            #print(p_frame.shape)
            
            input_shapes = {'image_tensor': p_frame,'image_info': p_frame.shape[1:]}
            
            #print("shape[1:]",p_frame.shape[1:])

        ### TODO: Start asynchronous inference for specified request ###
            #infer_time=time.time()
            infer_network.exec_net(input_shapes)
            #print("Time taken to run inference is: {} seconds".format(time.time()-infer_time))

        ### TODO: Wait for the result ###
            
            if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
                result = infer_network.get_output()
                #print("Time taken to run inference output is: {} seconds".format(time.time()-infer_time))
                # print(result.shape)
                # print(result[0][0].shape)
                # print(result[0].shape)
               
               
            ### TODO: Extract any desired stats from the results ###
                current_detec_count = 0                
                for box in result[0][0]: # Output shape is 1x1x100x7
                    conf = box[2]
                    if conf >= args.prob_threshold:
                        xmin = int(box[3] * width)
                        ymin = int(box[4] * height)
                        xmax = int(box[5] * width)
                        ymax = int(box[6] * height)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),  (0, 255, 0), 2)
                        current_detec_count = current_detec_count + 1
                    

            ### TODO: Calculate and send relevant information on ###                          
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###            
            ### Topic "person/duration": key of "duration" ###                

                if(current_detec_count < last_detect_count):  
                    frame_count_left = 1                     
                elif (current_detec_count > last_detect_count):                      
                    frame_count_enter = 1
                    
                else :
                    if(current_detec_count == 0):                        
                        if( (frame_count_left >= 1) and (frame_count_left < 5)) :
                            frame_count_left = frame_count_left + 1
                        elif (frame_count_left == 5) :     
                            time_diff = int(time.time() - current_time)
                            #int(datetime.now().timestamp() - current_time  )                        
                            total_count = total_count + 1
                            client.publish("person/duration", json.dumps({"duration": time_diff}))
                            client.publish("person", json.dumps({"total": total_count})) 
                            frame_count_left = 0    
                            
                    elif(current_detec_count == 1) :
                         if( (frame_count_enter >= 1) and (frame_count_enter < 3)) :
                            frame_count_enter = frame_count_enter + 1
                        
                         elif (frame_count_enter == 3) :  
                            current_time =  time.time() 
                            frame_count_enter = 0            
             
                client.publish("person", json.dumps({"count": current_detec_count}))                
                   
                last_detect_count = current_detec_count     


        ### TODO: Send the frame to the FFMPEG server ###
            frame = cv2.resize(frame, (768, 432))
            sys.stdout.buffer.write(frame)
            sys.stdout.flush()
     
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

   
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
