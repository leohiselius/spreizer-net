from brian2 import *
import numpy as np

# Neuron parameters
neuron_params = {
    'Cm' : 250 * pF,            # capacitance
    'gL' : 25 * nsiemens,       # conductance
    'EL' : -70 * mV,            # leak (rest) potential
    'mu_gwn' : 350 * pA,        # constant background current
    'sigma_gwn' : 100 * pA,      # std of stochastic background current
    'tau_e' : 5 * ms,           # e time constant
    'tau_i' : 5 * ms,           # i time constant
    'tau_ref' : 2 * ms,         # refractory time constant
    'Vt' : -55 * mV,            # Threshold potential
    'Vr' : -70 * mV             # Reset potential
    }
neuron_params['tau_m'] = neuron_params['Cm'] / neuron_params['gL']

if neuron_params['sigma_gwn'] == 0:
    neuron_eqs = '''
        dv/dt = (-gL * (v - EL) + Ie + Ii + mu_gwn) / Cm  : volt (unless refractory)
        '''
else:
    neuron_eqs = '''
        dv/dt = (-gL * (v - EL) + Ie + Ii + mu_gwn) / Cm + (sigma_gwn/Cm) * sqrt(2*tau_m) * xi : volt (unless refractory)
        '''

neuron_eqs += '''
            dIe/dt = (ke - Ie) / tau_e : ampere
            dke/dt = -ke / tau_e : ampere
            dIi/dt = (ki - Ii) / tau_i : ampere
            dki/dt = -ki / tau_i : ampere
            x : 1
            y : 1
            x_shift : 1
            y_shift : 1
            '''

# Synapse parameters
synapse_params = {
    'Je' : np.e * 10 * pA,      # excitatory synaptic current
    'g' : 4,                   # ratio of recurrent inhibition and excitation
    'sigma_e' : 0.075,          # excitatory connectivity width
    'sigma_i' : 0.1,        # inhibitory connectivity width
    'synapse_delay' : 1 * ms   # synapse delay
    }

synapse_params['Ji'] = -synapse_params['g'] * synapse_params['Je']
synapse_params['p_max_e'] = 0.05 / (2 * pi * synapse_params['sigma_e']**2)
synapse_params['p_max_i'] = 0.05 / (2 * pi * synapse_params['sigma_i']**2)

# Perlin scale and grid offset
perlin_scale = 6
grid_offset = 1
