#*******************************************************************************
# Copyright (c) 2021-2025
# School of Electrical, Computer and Energy Engineering, Arizona State University
# Department of Electrical and Computer Engineering, University of Wisconsin-Madison
# PI: Prof. Yu Cao, Prof. Umit Y. Ogras, Prof. Jae-sun Seo, Prof. Chaitali Chakrabrati
# All rights reserved.
#
# This source code is part of SIAM - a framework to benchmark chiplet-based IMC 
# architectures with synaptic devices(e.g., SRAM and RRAM).
# Copyright of the model is maintained by the developers, and the model is distributed under
# the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License
# http://creativecommons.org/licenses/by-nc/4.0/legalcode.
# The source code is free and you can redistribute and/or modify it
# by providing that the following conditions are met:
#
#  1) Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2) Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Developer list:
#		Gokul Krishnan Email: gkrish19@asu.edu
#		Sumit K. Mandal Email: skmandal@wisc.edu
#
# Acknowledgements: Prof.Shimeng Yu and his research group for NeuroSim
#*******************************************************************************/

import os
import numpy as np
from subprocess import call
from pathlib import Path
import time
import shutil

from Interconnect.noc_estimation import interconnect_estimation
from Interconnect.nop_estimation import nop_interconnect_estimation
from NoP_hardware import *

# Globabal variables for the Simulation
num_classes = 10

# NoC Parameters
quantization_bit = 8
weight_length = quantization_bit
input_length = quantization_bit
bus_width = 32
netname = 'ResNet-20'
xbar_size = 128
chiplet_size = 16 # Number of IMC tiles inside chiplet
num_chiplets = 25 # Total number of Chiplets
type = 'Homogeneous_Design'
scale = 100

# NoP Parameters - Extracted from GRS Nvidia 
ebit = 43.2 # pJ
area_per_lane = 5304.5 #um2
clocking_area = 10609 #um2
n_lane = 32
n_bits_per_chiplet = 4.19E+06 #Automate this in next version
scale_nop = 10


def write_matrix_weight(input_matrix, filename):
    cout = input_matrix.shape[-1]
    weight_matrix = input_matrix.reshape(-1, cout)
    np.savetxt(filename, weight_matrix, delimiter=",", fmt='%10.5f')

def write_matrix_activation_conv(input_matrix, fill_dimension, length, filename):
    filled_matrix_b = np.ones([input_matrix.shape[2], input_matrix.shape[1] * length], dtype=np.str)
    filled_matrix_bin, scale = dec2bin(input_matrix[0, :], length)
    for i, b in enumerate(filled_matrix_bin):
        filled_matrix_b[:, i::length] = b.transpose()
    np.savetxt(filename, filled_matrix_b, delimiter=",", fmt='%s')

def write_matrix_activation_fc(input_matrix, input_length, fill_dimension, length, filename):
    filled_matrix_b = np.ones([input_matrix.shape[1], length], dtype=np.str)
    filled_matrix_bin, scale = dec2bin(input_matrix[0, :], length)
    for i, b in enumerate(filled_matrix_bin):
        filled_matrix_b[:, i] = b
    np.savetxt(filename, filled_matrix_b, delimiter=",", fmt='%s')

def stretch_input(input_matrix, input_length, window_size = 5):
    input_shape = input_matrix.shape
    my_file = Path("./to_interconnect/ip_activation.csv")
    if my_file.is_file():
        with open('./to_interconnect/ip_activation.csv', 'a') as f: #Dumps file for the ip_activation for the interconnect simulator.
            f.write(str(input_shape[3]*input_shape[2]*input_shape[1]*input_length))
            f.write("\n")
            f.close()
    else:
        with open('./to_interconnect/ip_activation.csv', 'w') as f: #Dumps file for the ip_activation for the interconnect simulator.
            f.write(str(input_shape[3]*input_shape[2]*input_shape[1]*input_length))
            f.write("\n")
            f.close()
    # print("input_shape", input_shape)
    item_num_1 = ((input_shape[2]) - window_size + 1) * ((input_shape[3])-window_size + 1)
    item_num = max(item_num_1, 1)
    if (item_num_1==0):
        output_matrix = np.ones((input_shape[0],item_num,input_shape[1]*(window_size-1)*(window_size-1)))
    else:
        output_matrix = np.ones((input_shape[0],item_num,input_shape[1]*(window_size)*window_size))
    iter = 0
    for i in range( max(input_shape[2]-window_size + 1, 1) ):
        for j in range( max(input_shape[3]-window_size + 1, 1) ):
            for b in range(input_shape[0]):
                if (item_num_1==0):
                    output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size-1,j: j+window_size-1].reshape(input_shape[1]*(window_size-1)*(window_size-1))
                else:
                    output_matrix[b,iter,:] = input_matrix[b, :, i:i+window_size,j: j+window_size].reshape(input_shape[1]*window_size*window_size)
            iter += 1
    return output_matrix


def dec2bin(x,n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0/(2**(n-1))
    x_int = x/delta

    base = 2**(n-1)

    y[x_int>=0] = 0
    y[x_int< 0] = 1
    rest = x_int + base*y
    out.append(y.copy())
    scale_list.append(-base*delta)
    for i in range(n-1):
        base = base/2
        y[rest>=base] = 1
        y[rest<base]  = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)

    return out,scale_list


def main():

    IN = []
    W = []
    print("Starting the parsing of the network")
    # delete directories if these exist
    if not os.path.exists('./layer_record'):
        os.makedirs('./layer_record')
    output_path = './layer_record/'

    if os.path.exists('./layer_record/trace_command.sh'):
        os.remove('./layer_record/trace_command.sh')

    if os.path.exists('./to_interconnect'):
        shutil.rmtree('./to_interconnect')
    os.makedirs('./to_interconnect')
    
    if os.path.exists('./Final_Results'):
        shutil.rmtree('./Final_Results')
    os.makedirs('./Final_Results')

    f = open('./layer_record/trace_command.sh', "w")
    
    # Read the NetWork.csv file
    network_params = np.loadtxt('./SIAM/NetWork.csv', dtype=int, delimiter=',')
    num_layers = network_params.shape[0]

    # Create input matrix and weight matrix
    # Ideally if should be extracted from the network itself frm Pytroch or TensorFlow. Need to add this.
    # In interest of the different sturctures we can have higher flexibility

    # Regular Code
    for layer_idx in range(0, num_layers):            
        params_row = network_params[layer_idx]
        temp_array_IN = np.ones(shape=(1, network_params[layer_idx][2], \
                                        network_params[layer_idx][1], \
                                            network_params[layer_idx][0]), dtype='int8')
        IN.append(temp_array_IN)
        

        if (layer_idx == (num_layers-1)):
                temp_array_W = np.ones(shape=(network_params[layer_idx][4], \
                                                    network_params[layer_idx][3], network_params[layer_idx][2], \
                                                        num_classes), dtype='int8')
        else:
            temp_array_W = np.ones(shape=(network_params[layer_idx][4], \
                                                network_params[layer_idx][3], network_params[layer_idx][2], \
                                                    network_params[layer_idx][5]), dtype='int8')
        W.append(temp_array_W)
    f.write('./SIAM/main ./SIAM/NetWork.csv '+str(weight_length)+' '+str(input_length)+' ')

    # Debug Line
    #f.write('gdb --args ./SIAM/main ./SIAM/NetWork.csv '+str(weight_length)+' '+str(input_length)+' ')

    for i,(input,weight) in enumerate(zip(IN,W)):
        input_file_name = 'input_layer' + str(i) + '.csv'
        weight_file_name = 'weight_layer' + str(i) + '.csv'
        f.write(output_path + weight_file_name+' '+output_path + input_file_name+' ')
        write_matrix_weight(weight, output_path + weight_file_name)
        if len(weight.shape) > 2:
            k = weight.shape[0]
            write_matrix_activation_conv(stretch_input(input, input_length, k), None, input_length, output_path + input_file_name)
        else:
            write_matrix_activation_fc(input, input_length, None, input_length, output_path + input_file_name)
    f.close()

    # # Estimation of computation performance
    print("Starting the Estimation of the Performance")
    start = time.time()
    call(["/bin/bash", "./layer_record/trace_command.sh"])
    # # start = time.time()
   
    # perform cycle accurate noc simulation
    interconnect_estimation(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale)

    # NoP Estimation
    nop_interconnect_estimation(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale_nop)

    # Calculate and Dump NoP Hardware Cost
    Nop_area, NoP_energy = NoP_hardware_estimation(ebit, area_per_lane, clocking_area, n_lane, num_chiplets, n_bits_per_chiplet)
    area_file = open('/home/gkrish19/SIAM_Integration/Final_Results/area_chiplet.csv', 'a')
    area_file.write('Total NoP Driver area is' + '\t' + str(Nop_area) + '\t' + 'um^2')
    area_file.close()

    energy_file = open('/home/gkrish19/SIAM_Integration/Final_Results/Energy_chiplet.csv', 'a')
    energy_file.write('Total NoP Driver Energy is' + '\t' + str(NoP_energy) + '\t' + 'pJ')
    energy_file.close()

    end = time.time()
    print("The SIAM sim time is:", (end - start))

    
if __name__ == "__main__":
    main()
