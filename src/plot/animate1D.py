from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc

SimName = 'CUDA_1D_Particle_in_a_Box'

def animate1D(params):
    SimName = params['SimName']

    # Read simulation files
    FilePath = './out/' + SimName + '/waves/'
    waves = [f for f in listdir(FilePath) if isfile(join(FilePath, f))]
    waves.sort()


    wave = np.loadtxt(fname = FilePath + waves[0])
    # print(sum(np.square(wave[:,0]) + np.square(wave[:,1])))
    x = np.linspace(float(params["Xa"]), float(params["Xb"]), int(params["Nx"]))

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(float(params["Xa"]), float(params["Xb"])), ylim=(-10*max(np.square(wave[:,0]) + np.square(wave[:,1])), 10*max(np.square(wave[:,0]) + np.square(wave[:,1]))))
    fig.suptitle('1D Particle in a Box')
    plt.xlabel('x (nm)')
    plt.ylabel('ψ')
    plotcols = ["blue","red","black"]
    lines = []
    for index in range(3):
        lobj = ax.plot([],[],lw=2,color=plotcols[index])[0]
        lines.append(lobj)

    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([],[])
        return lines

    # animation function.  This is called sequentially
    def animate(i):
        wave = np.loadtxt(fname = FilePath + waves[i])
        y1 = wave[:,0]
        y2 = wave[:,1]
        y3 = np.square(wave[:,0]) + np.square(wave[:,1])
        ylist = [y1, y2, y3]

        for lnum,line in enumerate(lines):
            line.set_data(x, ylist[lnum]) # set data for each line separately.

        return lines

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=len(waves), interval=1, blit=True)

    SavePath = './out/' + SimName + '/animation.mp4'
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Cambridge Quantum Information Group'))

    anim.save(SavePath, writer=writer)

    plt.show()