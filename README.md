# spreizer-net
### Implementation in Brian2 of anisotropic spiking neural network

Before running, make sure the files "perlin.py", "torus_distance.py" and "params.py" as well as the folders "saves", "figures" lie in the same folder as "spreizer_net.py". In "saves", the folders "spike_monitors" and "state_monitors" should exist.

To run a simulation, create an instance of the SpreizerNet-class. Then, call the necessary methods, like so:

    sn = SpreizerNet()  
    sn.set_seed()  
    sn.populate()  
    sn.connect()  
    sn.set_initial_potentials()  
    sn.connect_spike_monitors()  
    sn.run_sim()  
    sn.save_monitors('example_simulation')  
    
The spike monitors are saved to the folder "spike_monitors" when save_monitors('simulation_name') is called. For spike monitors, four files per simulation are saved: "spike_monitors/simulation_name_e_t.npy", "spike_monitors/simulation_name_e_i.npy", "spike_monitors/simulation_name_i_t.npy", "spike_monitors/simulation_name_i_i.npy". These are the spike times and spike indices for the e and i population, respectively. Remember to change the name of the simulation if you wish to save spike monitors for multiple simulations: saving with the same name will overwrite previous saves. If state monitors are connected, they will also be saved when calling save_monitors('simulation_name'), to the folder "state_monitors".

See comments in code on connecting external input.

Parameters can be changed in the file "params.py". Be cautious when changing the parameter values in "params.py" from within another file: values in "params.py" which are dependent on other values will not change correspondingly!

The file "excitability_matrix.py" generates plots of the EI-landscape along with a flow map in which the perlin directions are visualized, scaled with the relative excitability at a given location. 

If you have any questions, please email me on leohi@kth.se
