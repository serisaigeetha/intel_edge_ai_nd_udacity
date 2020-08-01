'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from openvino.inference_engine import IENetwork, IECore
import logging as log
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"



class GazeEstimate:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        log.basicConfig(filename='log_stats.log',level=log.DEBUG)
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device=device

        self.exec_network = None

        try:
            self.core = IECore()            
            self.model = IENetwork(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=  [i for i in self.model.inputs.keys()]
        self.input_shape= self.model.inputs[self.input_name[1]].shape
        self.output_name=[o for o in self.model.outputs.keys()]



    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

        supported_layers = self.core.query_network(network=self.model, device_name=self.device)        
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]           
        if len(unsupported_layers) != 0:
            log.info("Unsupported layers found: {}".format(unsupported_layers))
        self.core.add_extension(CPU_EXTENSION, device_name=self.device)

        self.exec_network = self.core.load_network(self.model, device_name=self.device, num_requests=1)



        
    def predict(self, left_eye, right_eye, hp_angles,image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''

        left_eye_img = image[left_eye[0][1]:left_eye[0][3], left_eye[0][0]:left_eye[0][2]]  
        right_eye_img =  image[right_eye[0][1]:right_eye[0][3], right_eye[0][0]:right_eye[0][2]]

        left_eye_image = self.preprocess_input(left_eye_img)
        right_eye_image = self.preprocess_input(right_eye_img)
   
        input_shapes = {'left_eye_image': left_eye_image,
                                                         'right_eye_image': right_eye_image,
                                                         'head_pose_angles': hp_angles}
        self.exec_network.start_async(request_id=0, inputs=input_shapes)
        self.infer_status = self.exec_network.requests[0].wait(-1)       

        if self.infer_status == 0:        
            self.result = self.exec_network.requests[0].outputs[self.output_name[0]]
            #return self.draw_outputs(self.result,left_eye_image,right_eye_image)
            return self.result[0][0],self.result[0][1],self.result[0][2]



    # def draw_outputs(self, output, left_eye_image,right_eye_image):
        
    #     x = int(output[0][0]*10)
    #     y = int(output[0][1]*10)
    #     z = int(output[0][2]*10)

    #     cv2.line(left_eye_image, (x-z, y-z), (x+z, y+z), (255, 0, 255),5)
    #     cv2.line(left_eye_image, (x-z, y-z), (x+z, y+z), (255, 0, 255),5)

    #     return output[0][0],output[0][1],output[0][2]




    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame

