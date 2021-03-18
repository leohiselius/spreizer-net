import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from perlin import generate_perlin

def gaussian_2d_fast(size, amp, mu_x, mu_y, sigma):
    x = np.arange(0, 1, 1/size[0])
    y = np.arange(0, 1, 1/size[1])
    xs, ys = np.meshgrid(x,y)

    dxs = np.minimum(np.abs(xs-mu_x), 1-np.abs(xs-mu_x))
    dys = np.minimum(np.abs(ys-mu_y), 1-np.abs(ys-mu_y))

    heat_map = amp*np.exp(-(dxs**2+dys**2)/(2*sigma**2))
    return heat_map

def excitability_matrix(sigma_e, sigma_i, perlin_scale, grid_offset,
                        p_e=0.05, p_i=0.05, we=0.22, g=4,
                        n_row_e=120, n_row_i=60, 
                        expected_connectivity=True, is_plot=True):

    n_pop_e = n_row_e**2
    n_pop_i = n_row_i**2

    mu_gwn = 350 * 1e-12 # Ampere
    gL = 25 * 1e-9       # Siemens

    p_max_e = p_e / (2 * np.pi * sigma_e**2)
    p_max_i = p_i / (2 * np.pi * sigma_i**2)

    # Two landscapes: e and i. The contribution of each neuron is stored separately in the n_row_e**2 matrices
    e_landscape = np.zeros((n_row_e**2, n_row_e, n_row_e))
    i_landscape = np.zeros((n_row_i**2, n_row_e, n_row_e))
    perlin = generate_perlin(n_row_e, perlin_scale, seed_value=0)
    x = np.arange(0,1,1/n_row_e)
    y = np.arange(0,1,1/n_row_e)
    X, Y = np.meshgrid(x,y)
    U = np.cos(perlin)
    V = np.sin(perlin)

    # Excitatory
    mu_xs = np.arange(0,1,1/n_row_e)
    mu_ys = np.arange(0,1,1/n_row_e)
    counter = 0
    for i, mu_x in enumerate(mu_xs):
        for j, mu_y in enumerate(mu_ys):  
            x_offset = grid_offset / n_row_e * np.cos(perlin[i,j])
            y_offset = grid_offset / n_row_e * np.sin(perlin[i,j])
            mh = gaussian_2d_fast((n_row_e, n_row_e), p_max_e, mu_x+x_offset, mu_y+y_offset, sigma_e)

            #clip probabilities at 1
            e_landscape[counter] = np.minimum(mh, np.ones(mh.shape))
            counter += 1

    # Inhibitory
    mu_xs = np.arange(1/n_row_e,1+1/n_row_e,1/n_row_i)
    mu_ys = np.arange(1/n_row_e,1+1/n_row_e,1/n_row_i)
    counter = 0
    for mu_x in mu_xs:
        for mu_y in mu_ys:
            mh = gaussian_2d_fast((n_row_e, n_row_e), p_max_i, mu_x, mu_y, sigma_i)

            #clip probabilities at 1
            i_landscape[counter] = np.minimum(mh, np.ones(mh.shape))
            counter += 1

    # in total there should be n_pop_e * (n_pop_e * p_max_e) = 10 368 000 e-connections
    # and n_pop_i * (n_pop_e * 0.05) = 2 592 000 i-connections
    num_e_connections = np.sum(e_landscape)
    num_i_connections = np.sum(i_landscape)
    e_calibration = n_pop_e * n_pop_e * p_e / num_e_connections
    i_calibration = n_pop_i * n_pop_e * p_i / num_i_connections
    print('e_calibration is ', e_calibration)
    print('i_calibration is ', i_calibration)

    if expected_connectivity:
        # calculate expected number of connections
        e_landscape = n_row_e**2*np.mean(e_landscape, axis=0)
        i_landscape = n_row_i**2*np.mean(i_landscape, axis=0)

    else:   # we sample
        sample_e_landscape = np.zeros((n_row_e, n_row_e))
        for i in range(n_row_e):
            for j in range(n_row_e):
                neuron = e_landscape[:, i, j]
                random_numbers = np.random.random(n_row_e**2)
                num_connected = len(np.where(random_numbers<neuron)[0])
                sample_e_landscape[i, j] = num_connected

        sample_i_landscape = np.zeros((n_row_e, n_row_e))
        for i in range(n_row_e):
            for j in range(n_row_e):
                neuron = i_landscape[:, i, j]
                random_numbers = np.random.random(n_row_i**2)
                num_connected = len(np.where(random_numbers<neuron)[0])
                sample_i_landscape[i, j] = num_connected

        e_landscape = sample_e_landscape
        i_landscape = sample_i_landscape

    # Now we fill a landscape with physical units (mV)
    rest_pot = -70 # mV
    thres_pot = -55 # mV
    ext_pot = mu_gwn / gL * 1e3 #mV 
    no_activity_pot = rest_pot + ext_pot   # -56 mV when mu_gwn = 350 pA

    landscape = no_activity_pot * np.ones((n_row_e, n_row_e))

    # Synapse strengths
    we = we * e_calibration  #mV
    wi = -g * we * i_calibration / e_calibration #mV

    landscape += we * e_landscape
    landscape += wi * i_landscape

    # scale X and Y quiver according to values in ei_landscape. first normalize landscape

    norm_landscape = np.copy(landscape)
    norm_landscape -= np.amin(norm_landscape)
    norm_landscape /= np.amax(norm_landscape)

    U = 0.5*np.multiply(U, norm_landscape)
    V = 0.5*np.multiply(V, norm_landscape)

    if is_plot:
        # Plot
        plt.figure(figsize=(8,8))
        if expected_connectivity:
            mode = 'Expected '
        else:
            mode = 'Sampled '
        plt.title(mode+'EI landscape')
        plt.imshow(landscape, origin='lower', extent=[0,1,0,1])
        norm = mpl.colors.Normalize(vmin=round(np.amin(landscape)), vmax=round(np.amax(landscape)))
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm), label='mV')
        plt.quiver(X, Y, U, V, units='xy', scale=50)
        plt.suptitle(r'$\sigma_e=$'+str(sigma_e)+r', $\sigma_i=$'+str(sigma_i)+', perlin scale='+str(perlin_scale)+', g='+str(g),
            fontsize=15)
        plt.show()

        # Plot binary landscape (below/above threshold)
        above_thres = np.where(np.reshape(landscape, 14400)>thres_pot)
        binary_landscape = np.zeros(14400)
        binary_landscape[above_thres] = 1
        binary_landscape = np.reshape(binary_landscape,(120, 120))
        plt.figure(figsize=(8,8))
        plt.title(mode+'EI landscape (binary)')
        plt.imshow(binary_landscape, origin='lower', extent=[0,1,0,1])
        plt.quiver(X, Y, U, V, units='xy', scale=50)
        plt.suptitle(r'$\sigma_e=$'+str(sigma_e)+r', $\sigma_i=$'+str(sigma_i)+', perlin scale='+str(perlin_scale)+', g='+str(g),
            fontsize=15)
        plt.show()

    return landscape, X, Y, U, V
