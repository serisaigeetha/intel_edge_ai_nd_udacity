'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from openvino.inference_engine import IENetwork, IECore
import logging as log
#log.warning('model_name : %s', model_name)
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
        #log.info('model_structure : %s', self.model_structure)
        #log.info('model_weights : %s', self.model_weights)

        try:
            self.core = IECore()            
            self.model = IENetwork(model=self.model_structure, weights=self.model_weights)#self.core.read_network(model=self.model_structure, weights=self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        #log.info("self model")
        #log.info(self.model)
        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        #self.output_shape=self.model.outputs[self.output_name].shape  
        #log.info("out_info")  
        #log.info(self.model.outputs.keys())
        #log.info(self.output_shape)
        #log.info("self.output_name")
        #log.info(self.output_name)
        #log.info("facial landmarks model is initialized")    

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        #log.info("face landmarks load_model starts")
        supported_layers = self.core.query_network(network=self.model, device_name=self.device)        
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]           
        if len(unsupported_layers) != 0:
            log.info("Unsupported layers found: {}".format(unsupported_layers))
        self.core.add_extension(CPU_EXTENSION, device_name=self.device)

        self.exec_network = self.core.load_network(self.model, device_name=self.device, num_requests=1)
        #log.info("face landmarks model is loaded")
        

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        #log.info("face landmarks predict starts")
        p_image = self.preprocess_input(image)
        input_shapes = {self.input_name: p_image}
        self.exec_network.start_async(request_id=0, inputs=input_shapes)
        self.infer_status = self.exec_network.requests[0].wait(-1)       

        if self.infer_status == 0:        
            self.result = self.exec_network.requests[0].outputs[self.output_name]
            #log.info(self.exec_network.requests[0].outputs)
            #log.info(self.exec_network.requests[0].outputs[self.output_name])
            #log.info(self.result)
            #log.info(self.result.shape)

            
            return self.draw_outputs(self.result,image)


        
    
    def draw_outputs(self, coords, image):
        
        #logging.info("draw_output starts")
 
        #self.current_detec_count = 0   
        left_eye = []
        right_eye = []
        width = image.shape[1]
        height = image.shape[0]   
        #log.info(width)
        #log.info(height)
        #logging.info("result[0][0] is:",coords[0][0] )
        #logging.info(coords.shape)
        #log.info("coords landmark detect")
        #log.info(coords)
        #print(coords[0][0][0][0],coords[0][1][0][0],coords[0][2][0][0],coords[0][3][0][0] )
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
                        #self.current_detec_count = current_detec_count + 1
        left_eye.append((left_eye_xmin,left_eye_ymin,left_eye_xmax,left_eye_ymax))
        right_eye.append((right_eye_xmin,right_eye_ymin,right_eye_xmax,right_eye_ymax))
        #log.info("left_eye")
        #log.info(left_eye)
        #log.info("right eye")
        #log.info(left_eye)
        #log.info(right_eye)
        #log.info("face landmarks output successful")
        return left_eye,right_eye,image


    def check_model(self):
        return None

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        #log.info("preprocess out shape")
        #log.info(p_frame.shape)
        #log.info("facial landmarks preprocess ends")
        return p_frame

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        return None
