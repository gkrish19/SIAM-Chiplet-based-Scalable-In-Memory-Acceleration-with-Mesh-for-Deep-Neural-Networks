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

#NoC estimation of SIAM tool
import os, re, glob, sys, math, shutil
import numpy as np
import pandas as pd
from subprocess import call
from pathlib import Path
import math

from Interconnect.generate_traces_noc import generate_traces_noc
from Interconnect.run_booksim_noc import run_booksim_noc


#Take all below parameters as argument
# quantization_bit = 8
# bus_width = 32
# netname = 'VGG-19_45.3M'
# xbar_size = 256
# chiplet_size = 9
# num_chiplets = 144
# type = 'Homogeneous_Design'
# scale = 100



def interconnect_estimation(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale):

    generate_traces_noc(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale)

    print('Trace generation for NoC is finished')
    print('Starting to simulate the NoC trace')


    trace_directory_name = type + str(num_chiplets) + '_cnt_size_' + str(chiplet_size) + '_scale_' + str(scale) + '/'
    trace_directory_full_path = '/home/gkrish19/SIAM_Integration/Interconnect/' + netname + '_NoC_traces' + '/' + trace_directory_name
    
    results_directory_name = trace_directory_name
    results_directory_full_path = '/home/gkrish19/SIAM_Integration/Final_Results/NoC_Results_' + netname + '/' + results_directory_name
                
    run_booksim_noc(trace_directory_full_path)
    if (not os.path.exists(results_directory_full_path)):
    	os.makedirs(results_directory_full_path)
    
    
    os.system('mv /home/gkrish19/SIAM_Integration/Interconnect/logs/ ' + results_directory_full_path)


# interconnect_estimation(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale)
