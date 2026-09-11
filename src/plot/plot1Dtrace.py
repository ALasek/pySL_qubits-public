from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc


def plot1Dtrace(params):
    SimName = params['SimName']

    # Read simulation files
    FilePath = './out/' + SimName + '/waves/'
    waves = [f for f in listdir(FilePath) if isfile(join(FilePath, f))]
    waves.sort()

    wave = np.loadtxt(fname = FilePath + waves[0])


    x = np.linspace(float(params["Xa"]), float(params["Xb"]), int(params["Nx"]))
    psi2 = np.zeros((len(waves),len(x)))

    for i in range(0, len(waves)-1):
        wave = np.loadtxt(fname = FilePath + waves[i])
        psi2[i,:] = np.square(wave[:,0]) + np.square(wave[:,1])

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    plt.imshow(np.transpose(psi2), extent=[0,100,-200,200], aspect=0.1, cmap='inferno', vmin=0, vmax=0.01)
    plt.xlabel('t (ps)')
    plt.ylabel('y (nm)')
    fig.suptitle('1D Double Slit Experiment')

    SavePath = './out/' + SimName + '/trace1D.png'
    plt.savefig('trace1D.png', dpi=600)
    plt.show()