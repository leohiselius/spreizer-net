# Copyright 2021 Leo Hiselius
# The MIT License

from brian2 import *
import numpy as np
from scipy import interpolate
import warnings
from perlin import generate_perlin
from torus_distance import torus_distance
import params

class SpreizerNet:  

    def __init__(self, n_pop_e, n_pop_i):
        """Initializer

        Args:
            n_pop_e (int): Number of neurons in the e population. Must be a square number.
            n_pop_i (int): Number of neurons in the i population. Must be a square number.
        """   
        assert (int(np.sqrt(n_pop_e)) == np.sqrt(n_pop_e)) and \
            (int(np.sqrt(n_pop_i)) == np.sqrt(n_pop_i)), \
                'n_pop_e or n_pop_i is not a square number!'

        self.network = Network()
        self.network_dimensions = {'n_pop_e' : n_pop_e, 'n_pop_i' : n_pop_i, 
                                    'n_row_e' : int(np.sqrt(n_pop_e)), 'n_col_e' : int(np.sqrt(n_pop_e)),
                                    'n_row_i' : int(np.sqrt(n_pop_i)), 'n_col_i' : int(np.sqrt(n_pop_i))}
        self.neuron_groups = {'e' : None, 'i' : None}
        self.synapses = {'ee' : None, 'ie' : None, 'ei' : None, 'ii' : None}
        self.spike_monitors = {'e' : None, 'i' : None}
        self.state_monitors = {'e' : None, 'i' : None}
        self.spike_generator = None
        self.spike_generator_synapses = None
        
    def set_seed(self, seed_value=0):
        """Sets the seed for any stochastic elements of the simulation.

        Args:
            seed_value (int, optional): The seed value. Defaults to 0.
        """        
        seed(seed_value)
        np.random.seed(seed_value)
    
    def populate(self):     
        """Fills the network with e and i neurons on evenly spaced [0,1]x[0,1]-grid.
        """        
        if params.neuron_params['sigma_gwn'] == 0:
            integration_method = 'exact'
        else:
            integration_method = 'euler'
        
        # Instantiate the excitatory and inhibitory networks
        self.neuron_groups['e'] = NeuronGroup(self.network_dimensions['n_pop_e'], params.neuron_eqs, threshold='v>Vt',
                                   reset='v=Vr', refractory=params.neuron_params['tau_ref'], method=integration_method)
        
        self.neuron_groups['i'] = NeuronGroup(self.network_dimensions['n_pop_i'], params.neuron_eqs, threshold='v>Vt',
                                   reset='v=Vr', refractory=params.neuron_params['tau_ref'], method=integration_method)

        # Neuron parameters
        for param in params.neuron_params:
            self.neuron_groups['e'].namespace[param] = params.neuron_params[param]
            self.neuron_groups['i'].namespace[param] = params.neuron_params[param]

        # Place neurons on evenly spaced grid [0,1]x[0,1]. i neurons are shifted to lay in between e neurons.
        n_row_e = float(self.network_dimensions['n_row_e'])
        n_row_i = float(self.network_dimensions['n_row_i'])
        n_col_e = float(self.network_dimensions['n_col_e'])
        n_col_i = float(self.network_dimensions['n_col_i'])
        self.neuron_groups['e'].x = '(i // n_col_e) / n_col_e'
        self.neuron_groups['e'].y = '(i % n_row_e) / n_row_e'
        self.neuron_groups['i'].x = '(i // n_col_i) / n_col_i + 1/(2*n_col_e)'
        self.neuron_groups['i'].y = '(i % n_row_i) / n_row_i + 1/(2*n_row_e)'

        # Add to network attribute
        self.network.add(self.neuron_groups['e'])
        self.network.add(self.neuron_groups['i'])

    def connect(self, allow_multiple_connections=True, perlin_seed_value=0):
        """Connects neurons with synapses.

        Args:
            allow_multiple_connections (bool, optional): If True, multiple connections can be made where probability 
                                                            of connecting is greater than one. Defaults to True.
            perlin_seed (int, optional): Seed passed to generate_perlin(). Defaults to 0.
        """      
        
           
        # Generate perlin map
        perlin_map = generate_perlin(int(np.sqrt(self.network_dimensions['n_pop_e'])), params.perlin_scale,
            seed_value=perlin_seed_value, save=True)

        idx = 0
        for i in range(self.network_dimensions['n_col_e']):
            for j in range(self.network_dimensions['n_row_e']):
                self.neuron_groups['e'].x_shift[idx] = params.grid_offset / self.network_dimensions['n_col_e'] * np.cos(perlin_map[i, j])
                self.neuron_groups['e'].y_shift[idx] = params.grid_offset / self.network_dimensions['n_row_e'] * np.sin(perlin_map[i, j])
                idx += 1
      
        # Define synapses
        self.synapses['ee'] = Synapses(self.neuron_groups['e'], on_pre='ke += Je')
        self.synapses['ie'] = Synapses(self.neuron_groups['e'], self.neuron_groups['i'], on_pre='ke += Je')
        self.synapses['ei'] = Synapses(self.neuron_groups['i'], self.neuron_groups['e'], on_pre='ki += Ji')
        self.synapses['ii'] = Synapses(self.neuron_groups['i'], on_pre='ki += Ji')

        # Synapse parameters
        for param in params.synapse_params:
            self.synapses['ee'].namespace[param] = params.synapse_params[param]
            self.synapses['ie'].namespace[param] = params.synapse_params[param]
            self.synapses['ei'].namespace[param] = params.synapse_params[param]
            self.synapses['ii'].namespace[param] = params.synapse_params[param]
        
        # Make synapses
        synapse_names = ['ee', 'ie', 'ei', 'ii']
        if allow_multiple_connections:
            for syn_name in synapse_names:
                p_con = params.p_con[syn_name]
                n_con = '(int(' + p_con + ')+1)'                    # ceil the likelihood. This is number of connections
                p_con += '/' + n_con                                # divide the likelihood with its ceil
                self.synapses[syn_name].connect(p=p_con, n=n_con)   # n_con connections are made with p=p_con
        else:
            self.synapses['ee'].connect(p=params.p_con['ee'])
            self.synapses['ie'].connect(p=params.p_con['ie'])
            self.synapses['ei'].connect(p=params.p_con['ei'])
            self.synapses['ii'].connect(p=params.p_con['ii'])
            
        # Synapse delay
        self.synapses['ee'].delay = self.synapses['ie'].delay = \
             self.synapses['ei'].delay = self.synapses['ii'].delay = 'synapse_delay'

        # Add to network attribute
        self.network.add(self.synapses['ee'])
        self.network.add(self.synapses['ie'])
        self.network.add(self.synapses['ei'])
        self.network.add(self.synapses['ii'])

    def print_average_projections(self):
        """Prints the average number of outgoing projections for the e and i population, respectively.
        """        
        e_out = np.add(self.synapses['ee'].N_outgoing_pre, self.synapses['ie'].N_outgoing_pre)
        i_out = np.add(self.synapses['ei'].N_outgoing_pre, self.synapses['ii'].N_outgoing_pre)
        print('Number of outgoing projections for e network is ' +
              str(np.mean(e_out)) + ' +/- ' + str(np.std(e_out)))
        print('Number of outgoing projections for i network is ' +
              str(np.mean(i_out)) + ' +/- ' + str(np.std(i_out)))

    def set_initial_potentials(self, start_at_rest=True):
        """Sets the initial membrane potentials for all neurons

        Args:
            start_at_rest (bool, optional): Controls if neurons start at rest or uniformly random between Vr and Vt. 
                                            Defaults to True.
        """        
        if start_at_rest:      
            self.neuron_groups['e'].v = 'Vr'
            self.neuron_groups['i'].v = 'Vr'
        else:
            self.neuron_groups['e'].v = 'Vr + rand() * (Vt - Vr)'
            self.neuron_groups['i'].v = 'Vr + rand() * (Vt - Vr)'

    def prepare_external_input(self, samples, times, sigma_spike=1):
        """Reshapes flattened input data to n_row_e x n_col_e and returns the list of indices and 
            their respective spike times

        Args:
            samples ([np.array]): list of flattened input data.
            times ([ms]): the corresponding spike times (in ms) of the samples.
            sigma_spike (int, optional): standard deviation (in ms) of spike timing. Defaults to 1.

        Returns:
            spike_indices [int]: list of neuron indices
            spike_times [ms] : list of corresponding spike times for all neurin indices in spike_indices
        """        
        
        # Takes 1d input vector of length M^2 and outputs length N^2
        # Reshapes vector to MxM image, interpolates, resamples, reshape back to vector
        # Code below written by Andrew Lehr.
        M = int(np.sqrt(np.shape(samples[0])[0]))
        N = self.network_dimensions['n_row_e']
        spike_indices = []
        spike_times = []

        for t, v in enumerate(samples):
            time = times[t]
            v = np.rot90(v.reshape(M,M), k=3)
            x = np.arange(M)
            y = np.arange(M)
            xx, yy = np.meshgrid(x, y)
            f = interpolate.interp2d(x, y, v, kind='linear') 
            xnew = np.linspace(1, M, N)
            ynew = np.linspace(1, M, N)
            znew = f(xnew, ynew)
            znew = np.where(znew.reshape(N*N,) > 0)[0]
        # End of Andrew Lehr's code.

            for idx in znew:
                spike_indices.append(idx)
                spike_times.append(max(time/ms+sigma_spike*np.round(np.random.randn(),1),0)*ms)

        return spike_indices, spike_times
        
    def connect_external_input(self, spike_idcs, spike_times):
        # TODO DISCLAIMER: external input should probably be implemented as a current injection as opposed to this.
        """Generate external input (e.g. MNIST) at specific times
        Args:
            spike_idcs ([int]): the corresponding indices
            spike_times ([int]): the corresponding spike_times. Must be same length as spike_idcs.
        """        
        self.spike_generator = SpikeGeneratorGroup(self.network_dimensions['n_pop_e'], spike_idcs, spike_times)

        # When spike_generator fires, the corresponding neuron(s) of neuron_groups['e'] fires
        self.spike_generator_synapses = Synapses(self.spike_generator, self.neuron_groups['e'], on_pre='v=Vt+1*mV')
        self.spike_generator_synapses.namespace['Vt'] = params.neuron_params['Vt']

        # Connect one to one
        self.spike_generator_synapses.connect('i==j')
        self.network.add(self.spike_generator)
        self.network.add(self.spike_generator_synapses)
            
    def connect_spike_monitors(self):
        """Connect spike monitors to neuron_groups.
        """        
        self.spike_monitors['e'] = SpikeMonitor(self.neuron_groups['e'], record=True)
        self.spike_monitors['i'] = SpikeMonitor(self.neuron_groups['i'], record=True)
        self.network.add(self.spike_monitors['e'])
        self.network.add(self.spike_monitors['i'])

    def connect_state_monitors(self, e_idcs, i_idcs):
        """Connect state monitors to neuron_groups.

        Args:
            e_idcs ([int]): indices of neurons to be recorded
            i_idcs ([int]): indices of neurons to be recorded
        """              
        self.state_monitors['e'] = StateMonitor(self.neuron_groups['e'], 'v', record=e_idcs)
        self.state_monitors['i'] = StateMonitor(self.neuron_groups['i'], 'v', record=i_idcs)
        self.network.add(self.state_monitors['e'])
        self.network.add(self.state_monitors['i'])

    def store_network(self):
        """Stores network. restore() restores to this point.
        """        
        self.network.store()

    def run_sim(self, simulation_time=1*second):
        """Runs the simulation.

        Args:
            simulation_time (second, optional): duration of the simulation. Defaults to 1*second.
        """   
        # Print warning if the spike_monitors are not connected
        if self.spike_monitors['e'] is None or self.spike_monitors['i'] is None:
            warnings.warn('Simulation running without SpikeMonitors!')

        # Run simulation
        self.network.run(simulation_time, report='stderr', report_period=2*second)

    def save_monitors(self, simulation_name):
        """Saves any existing montitors (spike and state) to folders spike_monitors and state_monitors, respectively.

        Args:
            simulation_name (str): The name of the simulation.
        """        
        # Save spike_monitors
        if self.spike_monitors['e'] is not None: 
            np.save('saves/spike_monitors/'+simulation_name+'_e_i', self.spike_monitors['e'].i[:])
            np.save('saves/spike_monitors/'+simulation_name+'_e_t', self.spike_monitors['e'].t[:])
        if self.spike_monitors['i'] is not None:
            np.save('saves/spike_monitors/'+simulation_name+'_i_i', self.spike_monitors['i'].i[:])
            np.save('saves/spike_monitors/'+simulation_name+'_i_t', self.spike_monitors['i'].t[:])

        # Save state_monitors
        if self.state_monitors['e'] is not None:
            np.save('saves/state_monitors/'+simulation_name+'_e_v', self.state_monitors['e'].v[:])
            np.save('saves/state_monitors/'+simulation_name+'_e_t', self.state_monitors['e'].t[:])
        if self.state_monitors['i'] is not None:
            np.save('saves/state_monitors/'+simulation_name+'_i_v', self.state_monitors['i'].v[:])
            np.save('saves/state_monitors/'+simulation_name+'_i_t', self.state_monitors['i'].t[:])

    def restore_network(self):
        """Restores all attributes of the current class instance to the point where store_network() was called.
        """        
        if self.spike_generator in self.network.sorted_objects:
            self.network.remove(self.spike_generator)
            self.network.remove(self.spike_generator_synapses)
        self.network.restore(restore_random_state=True)   
