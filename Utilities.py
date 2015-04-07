import numpy as np
import matplotlib.pyplot as plt


# Utilities
def machine_limit_exp(var, const, l=-700.0, h=700.0):
    var_mn = np.abs(var)
    var_mx = np.abs(var)
    const_mn = np.min(np.sign(var) * const)
    const_mx = np.max(np.sign(var) * const)
    if np.abs(var) * const_mn < l:
        var_mn = np.abs(l / const_mn)
    if np.abs(var) * const_mx > h:
        var_mx = np.abs(h / const_mx)
    return np.min([var_mn, var_mx, np.abs(var)]) * np.sign(var)


def compute_increase_vector(width, decay, magnitude):
    dr = width if (width % 2) == 1 else width + 1
    vec = np.zeros((dr, ))
    middle = int((dr-1)/2)
    for i in range(0, middle+1, 1):
        vec[middle + i] = magnitude * (decay ** i)
        vec[middle - i] = magnitude * (decay ** i)
    # print 'vec: ', vec
    return vec


def compute_value_index(array, value):
    try:
        return (np.abs(array-np.array(value)[:, 0])).argmin()
    except (TypeError, IndexError):
        return(np.abs(array-value)).argmin()


def add_value_vector(prev_values, add_vector, index):
    # index += 1
    def compute_indices(ind, mid, van, val):
        miai = -1 * min(0, ind - mid - 1)
        maai = min(val, van - ind + mid + 1)
        mivi = max(0, ind - mid - 1)
        mavi = min(van, ind + mid)
        return miai, maai, mivi, mavi

    middle = int((len(add_vector) - 1)/2)

    val_ret = np.array(prev_values)
    min_add_index, max_add_index, min_val_index, max_val_index = compute_indices(index,
                                                                                 middle,
                                                                                 len(prev_values),
                                                                                 len(add_vector))

    val_ret[min_val_index:max_val_index] += add_vector[min_add_index: max_add_index]
    return val_ret # / val_ret.__abs__().max()


def smooth_1d_sequence(sequence, sigma=15):
    # smoothing functions for more readable plotting
    from scipy.ndimage import gaussian_filter1d
    sequence = np.array(sequence)
    assert len(sequence.shape) <= 2, 'Cannot interpret an array with more than 2 dimensions as a tuple of 1d sequences.'
    # asserting that the data is in the rows and that the array has a second dimension (for the for loop)
    if max(sequence.shape) > min(sequence.shape):
        if sequence.shape[1] > sequence.shape[0]:
            sequence = sequence.T
    else:
        sequence = sequence[None]
    for i in range(sequence.shape[1]):
        val_interpol = np.interp(range(sequence.shape[0]), range(sequence.shape[0]), sequence[:, i])
        sequence[:, i] = gaussian_filter1d(val_interpol, sigma)
    return sequence


def plot_results(data, vertical_limit=4):
    if len(data) > vertical_limit:
        first_dim = np.ceil(len(data)//2)
        second_dim = 2
        ax_indices = []
        for j in range(second_dim):
            for i in range(first_dim):
                ax_indices.append([i, j])
    else:
        first_dim = len(data)
        second_dim = 1
        ax_indices = range(first_dim)

    # creating the indices

    def create_plot(ax, points, name, ylimits, smoothing):
        if smoothing != -1:
            ax.plot(smooth_1d_sequence(points, smoothing))
        else:
            ax.plot(points)
        ax.set_title(name)
        ax.grid(True)
        if ylimits is not None:
            ax.set_ylim(ylimits)

    data_processed = []
    for name in data.keys():
        points = data[name]['values']
        ylimits = None if 'yLimits' not in data[name] else data[name]['yLimits']
        smoothing = 3 if 'smooth' not in data[name] else data[name]['smooth']
        data_processed.append([name, points, ylimits, smoothing])

    fig, ax_all = plt.subplots(first_dim, second_dim)
    for i in range(len(ax_indices)):
        try:
            try:
                axe = ax_all[ax_indices[i][0], ax_indices[i][1]]
            except TypeError:
                axe = ax_all[ax_indices[i]]
            create_plot(axe,
                        data_processed[i][1],
                        data_processed[i][0],
                        data_processed[i][2],
                        data_processed[i][3]
                        )
        except IndexError:
            break

    # show the plots
    plt.show()
