#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.out = None
        self.infer_status = None
        self.plugin = None
        self.net = None
        self.exec_network = None
        self.input_blob = None
        self.output_blob = None
        

    def load_model(self,  model, CPU_EXTENSION, DEVICE):
        ### TODO: Load the model ###
        
        self.plugin = IECore()
        model_bin = os.path.splitext(model)[0] + ".bin"
        self.net = IENetwork(model=model, weights=model_bin)
        
           
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(network=self.net, device_name="CPU")
        
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        
        ### TODO: Add any necessary extensions ###
        if len(unsupported_layers) != 0:
            #print("Unsupported layers found: {}".format(unsupported_layers))
            self.plugin.add_extension(CPU_EXTENSION, "CPU")
                  
        self.exec_network = self.plugin.load_network(self.net, "CPU")

        ### TODO: Return the loaded inference plugin ###
        
        self.input_blob = next(iter(self.net.inputs))
        #print("input_blob", self.input_blob)
        #print(self.net.inputs)
        #print(self.net.inputs['image_tensor'].shape)
        
        self.output_blob = next(iter(self.net.outputs))
        
        #print("output blob", self.output_blob)
        #print(self.net.outputs)
        
        #print("IR successfully loaded into Inference Engine.")
   
        
        return
        
    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        #print (self.net.inputs[self.input_blob].shape)
        return self.net.inputs['image_tensor'].shape 

    def exec_net(self,input_shapes):
        ### TODO: Start an asynchronous request ###
        #print("entered exec_net")
        self.exec_network.start_async(request_id=0, inputs=input_shapes)
            #inputs={'image_tensor': image})

        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        #print("entered wait")
        self.infer_status = self.exec_network.requests[0].wait(-1)
        #print(self.infer_status)

        return self.infer_status

    def get_output(self):
        ### TODO: Extract and return the output results
        out = self.exec_network.requests[0].outputs[self.output_blob]
        ### Note: You may need to update the function parameters. ###
        return out
