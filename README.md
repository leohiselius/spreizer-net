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

See the documentation for connecting external input.

Parameters can be changed in the file "params.py". Note that the values of the dictionary in "params.py" can be changed from within another file.

The file "excitability_matrix.py" generates plots of the EI-landscape along with a kind of flow map. 

When either sigma_e or sigma_i are smaller than ~0.08, the effective connection probability will decrease. Running excitability_matrix will provide details for how much Je and Ji should be increased in order to keep n_connections * synapse_strength constant.

If you have any questions, please email me on leohi@kth.se
