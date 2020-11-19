from __future__ import print_function
import numpy as np
import os
import yaml
import sys
import torch
import pickle
import re
import math

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
            self.input_image_size = [int(x) for x in self.input_image_size]
    
    def get_weights_data(self, layer_name, var_name):
        if var_name == 'kernel':
            var_name = 'weight'
        data = None
        if var_name in ['weight', 'bias']:
            try:
                data = self.state_dict[layer_name + '.' + var_name].numpy().transpose()
            except KeyError as kerr:
                # Deal with non-existent bias
                if var_name == 'bias':
                    data = None
                else:
                    raise kerr
        return data


class ImageSizeTracker:
    '''
    Feature map size formula is based on:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    Parameters:
        initial_size: list(int, int)
    I chose queue to implement BFS on multi-branch CNN.
    '''

    def __init__(self, initial_size):
        self.initial_size = list(initial_size)
        self.queue = [list(initial_size)]

    def compute_image_size(self, prev_size, kernel_size, stride, padding, dilation):
        assert(len(kernel_size) == 4)
        assert(len(stride) == 2)
        assert(len(padding) == 2)
        assert(len(dilation) == 2)
        print('prev_size = ', prev_size)
        cur_size = list(self.queue[-1]) if prev_size is None else prev_size
        # We only deal with CHW or NCHW format
        dim_list = [1, 2] if len(prev_size) == 3 else [2, 3]
        for dim, img_dim in enumerate(dim_list):
            cur_size[img_dim] = math.floor(((cur_size[img_dim] + 2 * padding[dim] - dilation[dim] * (kernel_size[dim+2] - 1) - 1) / stride[dim]) + 1)
            if cur_size[img_dim] <= 0:
                raise Exception(
                    "Non-positive dimension size={} yielded by kernel_size={}, stride={}, padding={}, dilation={}, image_size={}".
                    format(cur_size, kernel_size, stride, padding, dilation, self.queue))
        
        if len(prev_size) == 3:
            cur_size[0] = kernel_size[1]
        else:
            cur_size[1] = kernel_size[1]
        return cur_size

    def head(self):
        return self.queue[0]

    def pop(self):
        return self.queue.pop(0)

    def push(self, size):
        self.queue.append(size)


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
    image_size_tracker = ImageSizeTracker(reader.input_image_size)
    print('image_size', image_size_tracker)
    
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
            # Parse conv2d layer kernel spec
            layer_spec += ','
            conv2d_pattern = r'(?P<n_chan>\d+), (?P<n_filt>\d+), kernel_size=\((?P<filt_height>\d+), (?P<filt_width>\d+)\), stride=\((?P<stride_height>\d+), (?P<stride_width>\d+)\),\s*(?:padding=\((?P<padding_height>\d+), (?P<padding_width>\d+)\),)?\s*(?:dilation=\((?P<dilation_height>\d+), (?P<dilation_width>\d+)\),)?.*'
            conv2d_regex = re.compile(conv2d_pattern)
            # layer_match = re.findall(conv2d_regex, layer_spec)
            layer_match = re.search(conv2d_regex, layer_spec)
            if layer_match is None:
                raise Exception('Unable to interpret Conv2D layer ({})'.format(layer_spec))
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

            # Update image height and width in CNN
            current_shape = [layer[k] for k in conv2d_specs]
            print('Layer index: {}, layer type: {}, current shape: {}'.format(layer['name'], layer['class_name'], current_shape))
            
            prev_size = image_size_tracker.pop()
            cur_size = image_size_tracker.compute_image_size(
                prev_size, 
                [layer['n_chan'], layer['n_filt'], layer['filt_height'], layer['filt_width']],
                [layer['stride_height'], layer['stride_width']],
                [layer['padding_height'], layer['padding_width']],
                [layer['dilation_height'], layer['dilation_width']])
            image_size_tracker.push(cur_size)
            layer['in_height'] = prev_size[0]
            layer['in_width'] = prev_size[1]
            layer['out_height'] = cur_size[0]
            layer['out_width'] = cur_size[1]

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
    # input_layer['input_shape'] = [layer_list[0]['n_in']]
    input_layer['input_shape'] = image_size_tracker.initial_size
    layer_list.insert(0, input_layer)


    #################
    ## Generate HLS
    #################

    reader = PyTorchDataReader(yamlConfig)
    print('Creating HLS model')
    hls_model = HLSModel(yamlConfig, reader, layer_list)
    return hls_model
