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

#!/usr/bin/python
# python run_booksim.py <directory to injection file>


import os, re, glob, sys, math
import numpy

# # Extract command line arguments
# trace_file_dir = sys.argv[1] #directory name
# bus_width = sys.argv[2] #bus width


def run_booksim_mesh_chiplet_nop(trace_file_dir, bus_width):


    #os.chdir(trace_file_dir)
    #mesh_sizes_per_layer = pd.readcsv('mesh_sizes_per_layer.csv')
    mesh_size_file_name = trace_file_dir + '/nop_mesh_size.csv'
    mesh_size = int(numpy.loadtxt(mesh_size_file_name))
    #os.chdir('..')
    
    
    # Initialize file counter
    file_counter = 0
    
    # Create directory to store config files
    os.system('mkdir -p /home/gkrish19/SIAM_Integration/Interconnect/logs_NoP/configs')
    
        
    # Get a list of all files in directory
    files = glob.glob(trace_file_dir + '/*.txt')
    file_counter = 0
    total_latency = 0
    total_area = 0
    total_power = 0
    
    # Iterate over all files
    for file in files :
    
        # print('[ INFO] Processing file ' + file + ' ...')
    
        # Extract file name without extension and absolute path from filename
        run_name = os.path.splitext(os.path.basename(file))[0]
        run_id = run_name.strip('trace_file_chiplet_')
    
    
        # trace file
        trace_file_name = 'trace_file_chiplet_' + str(file_counter) + '.txt'
    
    
        # Open read file handle of config file
        fp = open('/home/gkrish19/SIAM_Integration/Interconnect/mesh_config_trace_based_nop', 'r')
    
        # Set path to config file
        config_file = '/home/gkrish19/SIAM_Integration/Interconnect/logs_NoP/configs/chiplet_' + str(file_counter) + '_mesh_config'
    
        # Open write file handle for config file
        outfile = open(config_file, 'w')
    
        # Iterate over file and set size of mesh in config file
        for line in fp :
    
            line = line.strip()
    
            # Search for pattern
            matchobj = re.match(r'^k=', line)
    
            # Set size of mesh if line in file corresponds to mesh size
            if matchobj :
                line = 'k=' + str(mesh_size) + ';'
    
            # Search for pattern
            matchobj1 = re.match(r'^channel_width = ', line)
    
            # Set size of mesh if line in file corresponds to mesh size
            if matchobj1 :
                line = 'channel_width = ' + str(bus_width) + ';'
            
            # Write config to file
            outfile.write(line + '\n')
    
        # Close file handles
        fp.close()
        outfile.close()
    
        # Set path to log file for trace files
        log_file = '/home/gkrish19/SIAM_Integration/Interconnect/logs_NoP/chiplet_'  + str(run_id) + '.log'
    
        # Copy trace file
        os.system('cp ' + file + ' trace_file.txt')
    
        # Run Booksim with config file and save log
        booksim_command = '/home/gkrish19/SIAM_Integration/Interconnect/booksim ' + config_file + ' > ' + log_file
        os.system(booksim_command)
    
        # Grep for packet latency average from log file
        latency = os.popen('grep "Trace is finished in" ' + log_file + ' | tail -1 | awk \'{print $5}\'').read().strip()
    
        # print('[ INFO] Latency for Chiplet : ' + str(run_id) + ' is ' + latency +'\n')
        total_latency = total_latency + int(latency)
    
    
        power = os.popen('grep "Total Power" ' + log_file + ' | tail -1 | awk \'{print $4}\'').read().strip()
    
        # print('[ INFO] Power for Chiplet : '  + str(run_id) + ' is ' + power +'\n')
        
        total_power = total_power + float(power)
    
    
        area = os.popen('grep "Total Area" ' + log_file + ' | tail -1 | awk \'{print $4}\'').read().strip()
    
        # print('[ INFO] Area for Chiplet : ' + str(run_id) + ' is ' + area +'\n')
    
        total_area = total_area + float(area)

        # Increment file counter
        file_counter += 1
    
    # Open output file handle to write latency
    outfile_area = open('/home/gkrish19/SIAM_Integration/Interconnect/logs_NoP/booksim_area.csv', 'a')
    outfile_area.write(str(total_area/file_counter) + '\n')
    outfile_area.close()

    area_file = open('/home/gkrish19/SIAM_Integration/Final_Results/area_chiplet.csv', 'a')
    area_file.write('Total NoP area is' + '\t' + str(total_area/file_counter) + '\t' + 'um^2' + '\n')
    area_file.close()
        
    # Open output file handle to write latency
    outfile_latency = open('./logs_NoP/booksim_latency.csv', 'a')
    outfile_latency.write(str(total_latency) + '\n')
    outfile_latency.close()

    latency_file = open('/home/gkrish19/SIAM_Integration/Final_Results/Latency_chiplet.csv', 'a')
    latency_file.write('Total NoP latency is' +'\t' + str(total_latency*4e-9) +'\t' + 's' + '\n')
    latency_file.close()
    
    # Open output file handle to write latency
    outfile_power = open('./logs_NoP/booksim_power.csv', 'a')
    outfile_power.write(str(total_power/file_counter) + '\n')
    outfile_power.close()

    power_file = open('/home/gkrish19/SIAM_Integration/Final_Results/Energy_chiplet.csv', 'a')
    power_file.write('Total NoP power is' +'\t' + str(total_power/file_counter) +'\t' + 'mW' + '\n')
    power_file.close()  