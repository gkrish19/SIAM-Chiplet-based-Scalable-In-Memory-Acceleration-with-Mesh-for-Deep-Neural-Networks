# SIAM: Chiplet-based Scalable In-Memory Acceleration with Mesh for Deep Neural Networks

A comprehensive tool that allows for system-level performance estimation of chiplet-based In-Memory computing (IMC) architectures. 
This work was developed by the researchers at Arizona State University and University of Wisconsin-Madison. The PIs invovled are Prof. Yu Cao, Prof. Umit Y. Ogras, Prof. Jae-sun Seo, and Prof. Chaitali Chakrabarti.

SIAM consists of 4 main engines: Partition and Mapping Engine, Circuits and NoC Engine, NoP Engine, and DRAM Engine. This tool incorporates device, circuit, architecture, and algorithm properties of IMC. The first three engines are combined into a single codebase while the DRAM engine is provided as a separate GUI. 

The figure below shows the overall block diagram of SIAM:
![image](https://user-images.githubusercontent.com/39285763/137603062-8f04f99d-d874-4e8e-8d6d-4462215067e9.png)

The overall arhcitecture utilized within SIAM is shown below:
![image](https://user-images.githubusercontent.com/39285763/137603105-b8dd95a5-4acb-4a11-84c0-2055450924be.png)


SIAM supports two different parition schemes to generate either a homogeneous strucure or a custom structure, as shown below:
![image](https://user-images.githubusercontent.com/39285763/137603110-9d655699-61bb-43cc-ac4b-189e89bf6f9e.png)


The current version of SIAM is a beta version that we are currently testing and improving. We have an NoC-mesh, and NoP-mesh with the driver and interconnect properties as that in the Nvidia GRS driver [R1]. The interconnect parasitics have been incorporated into the NoP by using the PTM models [R2]. The current mapping within SIAM follows that in [R3]. The current version of the tool does not include software accuracy estimation. It will be added in future versions. SIAM has been calibrated with the results to that in SIMBA [R4].

Features incorporated in the beta version:

1. In-Memory Computing Circuits and Architecture
2. Network-on-package - Mesh structure
3. Network-on-chip - Mesh
4. DRAM Estiamtion
5. Chiplet Architecture customization
6. NoP driver and interconnect from Nvidia GRS
7. Support for a wide range of DNNs (refer to Network folder)

Folder/File Definitions:

SIAM: The main codebase for the IMC and parition and mapping engine
VAMPIRE: The overall DRAM engine GUI file. This can be run using the final_gui.py file within it
Interconnect: NoC and NoP estiamtion folder
to_interconnect: Files generated by SIAM parition engine for the NoC and NoP estimation
main.py: The top file used to run the whole tool
RAMULATOR_final: The DRAM latency generator tool. Does not require any edits
hardware_estimation_v2: Dummy file for the main function. Used for debug purpose only
Networks: Network structures for different DNNs for the user

To change the network structure please choose the required one from the Networks folder and add it to the SIAM folder. For any new network structure, please edit the network.csv file in the following order across columns: IFM_size, IFM_size, N_IFM, Kx, Ky, NOFM, 0/1 if not followed/followed by pooling, layer-wsie sparsity.

A demo video is added to the repository (through dropbox) to help users get started. A detailed user manual will be posted shortly.
To run the NoC and NoP files you must enter the Interconnect folder through the terminal and type chmod +x booksim.
To run the gui for DRAM use the final_gui.py within the VAMPIRE folder. 
First time user must make file for the vampire, ramulator, and the SIAM tool.

Some of the paths have been hard coded into the main_run.py and python files within the Interconnect folder. The user must change them to desired folder for use. This will be fixed in the next version of SIAM.

Dependencies:

1. gcc 7 
2. Python 3.5 or higher

The files are hosted in a dropbox folder shared via a link that can used to download the whole repository. This is done as larger fiels exist in the whole framework that limit the upload capabilities. 

If you are using this tool, you are required to cite the following work:

[1] Gokul Krishnan, Sumit K. Mandal, Manvitha Pannala, Chaitali Chakrabarti, Jae-Sun Seo, Umit Y. Ogras, and Yu Cao. "SIAM: Chiplet-based Scalable In-Memory Acceleration with Mesh for Deep Neural Networks." ACM Transactions on Embedded Computing Systems (TECS) 20, no. 5s (2021): 1-24.

Additional work to be cited:

[2] Gokul Krishnan, Sumit K. Mandal, Chaitali Chakrabarti, Jae-sun Seo, Umit Y. Ogras, and Yu Cao. "Interconnect-aware area and energy optimization for in-memory acceleration of DNNs." IEEE Design & Test (2020). 

[3] Sumit K. Mandal, Gokul Krishnan, Chaitali Chakrabarti, Jae-Sun Seo, Yu Cao, and Umit Y. Ogras. "A Latency-Optimized Reconfigurable NoC for In-Memory Acceleration of DNNs." IEEE Journal on Emerging and Selected Topics in Circuits and Systems 10, no. 3 (2020): 362-375.

References:

[R1] Walker J. Turner, et al. "Ground-referenced signaling for intra-chip and short-reach chip-to-chip interconnects." 2018 IEEE Custom Integrated Circuits Conference (CICC). IEEE, 2018.

[R2] S Sinha, et. al, “Exploring Sub-20nm FinFET Design with Predictive Technology Models.” IEEE DAC 2012.

[R3] Gokul Krishnan, Sumit K. Mandal, Chaitali Chakrabarti, Jae-sun Seo, Umit Y. Ogras, and Yu Cao. "Interconnect-aware area and energy optimization for in-memory acceleration of DNNs." IEEE Design & Test (2020). 

[R4] Yakun Sophia Shao, et al. "Simba: Scaling deep-learning inference with multi-chip-module-based architecture." Proceedings of the 52nd Annual IEEE/ACM International Symposium on Microarchitecture. 2019.

[R5] Pai-Yu Chen, Xiaochen Peng, and Shimeng Yu. "NeuroSim: A circuit-level macro model for benchmarking neuro-inspired architectures in online learning." IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems 37, no. 12 (2018): 3067-3080.

For any queries please contact the following:

1. Gokul Krishnan: gkrish19@asu.edu
2. Sumit K. Mandal: skmandal@wisc.edu

The model is made publicly available on a non-commercial basis. Copyright of the model is maintained by the developers, and the model is distributed under the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License (https://creativecommons.org/licenses/by-nc/4.0/legalcode).


Acknowledgements:
This work was supported in part by C-BRIC, one of the six centers in JUMP, a Semiconductor Research Corporation program sponsored by DARPA.
