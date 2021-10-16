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
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:22:09 2021

"""


import os, re, glob, sys, math
import timeit

from Interconnect.generate_traces_nop import generate_traces_nop
from Interconnect.run_booksim_mesh_chiplet_nop import run_booksim_mesh_chiplet_nop

start = timeit.default_timer()

# chiplet_size = 25
# num_chiplet = 144
# scale = 1
# bus_width = 4
# netname = 'VGG19_homogeneous_NoP_traces'

def nop_interconnect_estimation(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale):
    
    generate_traces_nop(quantization_bit, bus_width, netname, xbar_size, chiplet_size, num_chiplets, type, scale)

    print('Trace generation for NoP is finished')
    print('Starting to simulate the NoP trace')
    
    trace_directory_name = str(type) + '_' + str(num_chiplets) + '_cnt_size_' + str(chiplet_size) + '_scale_' + str(scale) + '_bus_width_' + str(bus_width)
    trace_directory_full_path = '/home/gkrish19/SIAM_Integration/Interconnect/' + netname + '_NoP_traces' + '/' + trace_directory_name
    
    results_directory_name = 'results_' + trace_directory_name
    results_directory_full_path = '/home/gkrish19/SIAM_Integration/Final_Results/NoP_Results_' + 'results_' + netname + '/' + results_directory_name
    
    os.system('pwd')
    
    # os.system('python3 run_booksim_mesh_chiplet_nop.py ' + trace_directory_full_path + ' ' + str(bus_width))
    run_booksim_mesh_chiplet_nop(trace_directory_full_path, bus_width)
    
    if (not os.path.exists(results_directory_full_path)):
    	os.makedirs(results_directory_full_path)
    
    
    # os.system('mv /home/gkrish19/SIAM_Integration/Interconnect/logs_NoP/ ' + results_directory_full_path)
    		
    
                    
