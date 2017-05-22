import numpy as np
import matplotlib.pyplot as p
from mpl_toolkits.mplot3d import Axes3D


def read_file_perso(filename):
    positions = np.loadtxt(filename)
    label = filename.split('_')[1].split('.')[0]

    x = [temp[0] for temp in positions]
    y = [temp[1] for temp in positions]
    z = [temp[2] * 0 for temp in positions]

    return x, y, z, label


def make_3D_plot(ggh_filename, vbf_filename):
    ggh_x, ggh_y, ggh_z, label = read_file_perso(ggh_filename)
    vbf_x, vbf_y, vbf_z, label = read_file_perso(vbf_filename)
    fig = p.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ggh_x, ggh_y, ggh_z, marker='o', c='b', label='GGH events')
    ax.scatter(vbf_x, vbf_y, vbf_z, marker='^', c='r', label='VBF events')
    ax.autoscale_view()
    p.title('3D plot with third dimension '+str(label))
    p.legend(loc=3)
    p.show()

def plot(third_dimension, path='figs/'):
    make_3D_plot(path+'ggh_'+str(third_dimension)+'.txt', path+'vbf_'+str(third_dimension)+'.txt')
