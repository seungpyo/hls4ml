from __future__ import print_function
import numpy as np
import os
import yaml
import sys
import torch
import pickle
import re

from hls4ml.model import HLSModel

class PyTorchDataReader:
    def __init__(self, config):
        self.config = config

        if not torch.cuda.is_available():
            self.torch_model = torch.load(config['PytorchModel'], map_location=lambda storage, loc: storage)
        else:
            self.torch_model = torch.load(config['PytorchModel'])

        self.state_dict = self.torch_model.state_dict()
    
        if 'InputImageSize' in config:
            self.input_image_size = config['InputImageSize'].split('x')
            self.input_image_size = tuple([int(x) for x in self.input_image_size])
    
    def get_weights_data(self, layer_name, var_name):
        if var_name == 'kernel':
            var_name = 'weight'
        data = None
        if var_name in ['weight', 'bias']:
            data = self.state_dict[layer_name + '.' + var_name].numpy().transpose()

        return data


def pytorch_to_hls(yamlConfig):

    ######################
    ##  Do translation
    ######################

    print('Interpreting Model')
    reader = PyTorchDataReader(yamlConfig)

    core_layers = ['Linear', 'Conv2d']
    skip_layers = ['Dropout', 'Flatten']
    activation_layers = ['ReLU', 'Sigmoid', 'Tanh', 'SELU', 'LeakyReLU', 'Softmax', 'Softplus', 'Softsign']
    supported_layers = core_layers + skip_layers + activation_layers

    #This is a list of dictionaries to hold all the layer info we need to generate HLS
    layer_list = []
    image_size = reader.input_image_size
    
    #Loop through layers
    print('Topology:')
    modelstr = repr(reader.torch_model).split('\n')
    
    for pytorch_layer in modelstr:
        layer_match = re.match(r'\((.*)\): (\w+)\((.*)\)', pytorch_layer.strip())
        if layer_match is None:
            continue
        
        layer_idx  = layer_match.group(1)
        layer_type = layer_match.group(2)
        layer_spec = layer_match.group(3)

        # #Dictionary to fill in and append to layer_list
        layer={}

        #layer_type = matchname.group(1)
        if layer_type not in supported_layers:
            raise Exception('Unsupported layer {}'.format(layer_type))

        if layer_type == 'Linear':
            layer['class_name'] = 'Dense'
            layer['name'] = layer_idx

            dense_spec = re.match(r'in_features=(\d+), out_features=(\d+).*', layer_spec)
            if dense_spec is None:
                raise Exception('Unable to interpret Linear layer ({})'.format(layer_spec))

            # #Get number of inputs and outputs
            layer['n_in'] = int(dense_spec.group(1))
            layer['n_out'] = int(dense_spec.group(2))

            current_shape = [layer['n_in'], layer['n_out']]
            print('Layer index: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], current_shape))

        elif layer_type == 'Conv2d':
            layer['class_name'] = 'Conv2D'
            layer['name'] = layer_idx
            layer_spec += ','
            conv2d_pattern = r'(?P<n_chan>\d+), (?P<n_filt>\d+), kernel_size=\((?P<filt_height>\d+), (?P<filt_width>\d+)\), stride=\((?P<stride_height>\d+), (?P<stride_width>\d+)\),\s*(?:padding=\((?P<padding_height>\d+), (?P<padding_width>\d+)\),)?\s*(?:dilation=\((?P<dilation_height>\d+), (?P<dilation_width>\d+)\),)?.*'
            conv2d_regex = re.compile(conv2d_pattern)
            # layer_match = re.findall(conv2d_regex, layer_spec)
            layer_match = re.search(conv2d_regex, layer_spec)
            if layer_match is None:
                print('Parsing failed for layer repr:')
                print(layer_spec)

            conv2d_specs = ['n_chan', 'n_filt', 'filt_height', 'filt_width']
            conv2d_specs_optional = {
                'stride_height': '1', 'stride_width': '1',
                'padding_height': '0', 'padding_width': '0',
                'dilation_height': '1', 'dilation_width': '1',
            }
            for spec in conv2d_specs:
                layer[spec] = layer_match.group(spec)
            for spec in conv2d_specs_optional:
                layer[spec] = layer_match.group(spec)
                if layer[spec] is None:
                    layer[spec] = conv2d_specs_optional[spec]
            for spec in conv2d_specs:
                layer[spec] = int(layer[spec])
            for spec in conv2d_specs_optional:
                layer[spec] = int(layer[spec])
            layer['n_in'] = layer['n_chan'] * layer['filt_height'] * layer['filt_width']
            layer['n_out'] = layer['n_filt']
            print('Parsing succeed for layer repr:')
            print(layer_spec)
            print('layer:')
            print(layer)

        elif layer_type in activation_layers:
            layer['activation'] = layer_type.lower()
            if layer['activation'] == 'Softmax':
                layer['class_name'] = 'Softmax'
            else:
                layer['class_name'] = 'Activation'
            layer['name'] = layer['activation'] + '_' + str(layer_idx)

        layer_list.append(layer)

    input_layer = {}
    input_layer['name'] = 'input1'
    input_layer['class_name'] = 'InputLayer'
    input_layer['input_shape'] = [layer_list[0]['n_in']]
    layer_list.insert(0, input_layer)


    #################
    ## Generate HLS
    #################

    reader = PyTorchDataReader(yamlConfig)
    print('Creating HLS model')
    hls_model = HLSModel(yamlConfig, reader, layer_list)
    return hls_model
