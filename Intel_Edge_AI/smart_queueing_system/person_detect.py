

import numpy as np
import time
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import sys

class Queue:

    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d


class PersonDetect:


    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device
        self.threshold=threshold
        self.exec_network = None

        try:
            self.core = IECore()
            self.model = self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):

        #print("load_model starts")
        #supported_layers = self.core.query_network(network=self.model, device_name=self.device)        
        #unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]           
        #if len(unsupported_layers) != 0:
        #    print("Unsupported layers found: {}".format(unsupported_layers))
            #self.core.add_extension(CPU_EXTENSION, device_name=self.device)


        self.exec_network = self.core.load_network(self.model, device_name=self.device, num_requests=1)

        #print("model is loaded")
        
    def predict(self, image):
        
        #print("predict starts")
        p_image = self.preprocess_input(image)
        input_shapes = {self.input_name: p_image}
        self.exec_network.start_async(request_id=0, inputs=input_shapes)
        self.infer_status = self.exec_network.requests[0].wait(-1)       

        if self.infer_status == 0:        
            self.result = self.exec_network.requests[0].outputs[self.output_name]
            #image_predict = self.draw_outputs(self.result,image)
        
            #print("predict ends")
            return self.draw_outputs(self.result,image)

        
        
    
    def draw_outputs(self, coords, image):
        
        #print("draw_output starts")
 
        #self.current_detec_count = 0   
        detect_coor = []
        width = image.shape[1]
        height = image.shape[0]   
        #print("result[0][0] is:",coords[0][0] )
        #print(coords.shape)

        for box in coords[0][0]: # Output shape is 1x1x100x7
                    conf = box[2]
                    if conf >= self.threshold:
                        xmin = int(box[3] * width)
                        ymin = int(box[4] * height)
                        xmax = int(box[5] * width)
                        ymax = int(box[6] * height)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax),  (0, 255, 0), 2)
                        #self.current_detec_count = current_detec_count + 1
                        detect_coor.append((xmin,ymin,xmax,ymax))
        
        #print("draw_output ends")
        return detect_coor, image

    def preprocess_outputs(self, outputs):

        

        return None
            
        
        
    def preprocess_input(self, image):
        
        #print("preprocess starts")
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        #print("preprocess ends")
        return p_frame

def main(args):

    model=args.model
    device=args.device
    video_file=args.video
    max_people=args.max_people
    threshold=args.threshold
    output_path=args.output_path
    #print("All args")
    #print(model)
    #print(device)
    #print(video_file)
    #print(threshold)
    #print(output_path)

    start_model_load_time=time.time()
    pd= PersonDetect(model, device, threshold)
    pd.load_model()
    total_model_load_time = time.time() - start_model_load_time

    queue=Queue()

    try:
        queue_param=np.load(args.queue_param)
        for q in queue_param:
            queue.add_queue(q)
    except:
        print("error loading queue param file")

    try:
        cap=cv2.VideoCapture(video_file)
    except FileNotFoundError:
        print("Cannot locate video file: "+ video_file)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(os.path.join(output_path, 'output_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    counter=0
    start_inference_time=time.time()

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1

            coords, image= pd.predict(frame)
            #print("main_predict completed")
            #print("main: coords", coords)
            num_people= queue.check_coords(coords)
            #print("num_of_people", num_people)
            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
            out_text=""
            y_pixel=25

            for k, v in num_people.items():
                out_text += f"No. of People in Queue {k} is {v} "
                if v >= int(max_people):
                    out_text += f" Queue full; Please move to next Queue "
                cv2.putText(image, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                out_text=""
                y_pixel+=40
            out_video.write(image)

        total_time=time.time()-start_inference_time
        total_inference_time=round(total_time, 1)
        fps=counter/total_inference_time

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(str(total_inference_time)+'\n')
            f.write(str(fps)+'\n')
            f.write(str(total_model_load_time)+'\n')

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference: ", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--max_people', default=2)
    parser.add_argument('--threshold', default=0.60)
    
    args=parser.parse_args()
    

    main(args)