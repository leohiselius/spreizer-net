# spreizer-net
### Implementation in Brian2 of anisotropic spiking neural network

Before running, make sure the files "perlin.py", "torus_distance.py" and "params.py" as well as the folders "spike_monitors", "state_monitors", "figures" lie in the same folder as "spreizer_net.py".

To run a simulation, create an instance of the SpreizerNet-class. The only required arguments are the respective sizes of the excitatory and inhibitory populations, respectively. Note that these number must be square. Then, call the necessary methods, like so:

    sn = SpreizerNet(14400, 3600)  
    sn.set_seed()  
    sn.populate()  
    sn.connect()  
    sn.set_initial_potentials()  
    sn.connect_spike_monitors()  
    sn.run_sim()  
    sn.save_monitors('example_simulation')  
    
The spike monitors are saved to the folder "spike_monitors" when save_monitors('simulation_name') is called. For spike monitors, four files per simulation are saved: "spike_monitors/simulation_name_e_t.npy", "spike_monitors/simulation_name_e_i.npy", "spike_monitors/simulation_name_i_t.npy", "spike_monitors/simulation_name_i_i.npy". These are the spike times and spike indices for the e and i population, respectively. Remember to change the name of the simulation if you wish to save spike monitors for multiple simulations: saving with the same name will overwrite previous saves. If state monitors are connected, they will also be saved when calling save_monitors('simulation_name'), to the folder "state_monitors".

See comments in code on connecting external input.

Parameters can be changed in the file "params.py". Note that the parameter values in "params.py" can be changed from within another file.

The file "excitability_matrix.py" generates plots of the EI-landscape along with a flow map in which the perlin directions are visualized, scaled with the relative excitability at a given location. 

When either sigma_e or sigma_i are smaller than ~0.08, the effective connection probability will decrease. Running excitability_matrix() will provide details for how much Je and Ji should be increased in order to keep n_connections * synapse_strength constant.

If you have any questions, please email me on leohi@kth.se
