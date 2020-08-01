'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from openvino.inference_engine import IENetwork, IECore
import logging as log

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
class FacialLandmarks:
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

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))


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

        

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        p_image = self.preprocess_input(image)
        input_shapes = {self.input_name: p_image}
        self.exec_network.start_async(request_id=0, inputs=input_shapes)
        self.infer_status = self.exec_network.requests[0].wait(-1)       

        if self.infer_status == 0:        
            self.result = self.exec_network.requests[0].outputs[self.output_name]
            return self.draw_outputs(self.result,image)


        
    
    def draw_outputs(self, coords, image):
        

        left_eye = []
        right_eye = []
        width = image.shape[1]
        height = image.shape[0]   

        left_eye_xmin = int(coords[0][0][0][0] * width) - 20
        left_eye_ymin = int(coords[0][1][0][0] * height) - 20

        left_eye_xmax = int(coords[0][0][0][0] * width) + 20
        left_eye_ymax = int(coords[0][1][0][0] * height) + 20

        right_eye_xmin = int(coords[0][2][0][0] * width) - 20
        right_eye_ymin = int(coords[0][3][0][0] * height) - 20

        right_eye_xmax = int(coords[0][2][0][0] * width) + 20
        right_eye_ymax = int(coords[0][3][0][0] * height) + 20
        cv2.rectangle(image, (left_eye_xmin, left_eye_ymin), (left_eye_xmax, left_eye_ymax),  (0, 255, 0), 2)
        cv2.rectangle(image, (right_eye_xmin, right_eye_ymin), (right_eye_xmax, right_eye_ymax),  (0, 255, 0), 2)
                       
        left_eye.append((left_eye_xmin,left_eye_ymin,left_eye_xmax,left_eye_ymax))
        right_eye.append((right_eye_xmin,right_eye_ymin,right_eye_xmax,right_eye_ymax))

        return left_eye,right_eye,image


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame


