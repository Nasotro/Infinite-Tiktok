import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2
from matplotlib.animation import FuncAnimation


def anim1():
    fig = plt.figure()
    im = plt.imshow(np.zeros((100, 100)), cmap='viridis')

    def update(frame):
        octaves = 6
        freq = 16.0 * frame / 100.0 if frame > 0 else 0.1
        noise = np.zeros((100, 100))
        for y in range(100):
            for x in range(100):
                noise[x][y] = pnoise2(x / freq, y / freq, octaves=octaves)
        im.set_array(noise)
        return [im]

    ani = FuncAnimation(fig, func=update, frames=100, interval=30)
    plt.show()


    # ani.save(filename="anim_example.mp4", writer="ffmpeg")

def anim2():
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_xticklabels([])  # Remove radial labels
    ax.set_yticklabels([])  # Remove angular labels
    ax.set_rmax(2)
    ax.grid(False)

    line, = ax.plot([], [])

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        theta = np.linspace(0, 2*np.pi, 1000)
        r = np.sin(i*theta)
        line.set_data(theta, r)
        return line,

    ani = FuncAnimation(fig, animate, init_func=init, frames=np.linspace(1, 50, 10000), blit=True)

    plt.show()

    # ani.save(filename="anim_example.mp4", writer="ffmpeg", fps=300)

def anim3():
    data = np.random.rand(10, 10)
    fig, ax = plt.subplots()

    im = ax.imshow(data, cmap='gray')
    cbar = fig.colorbar(im)
    def animate(i):
        
        size = 256
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0

        noise = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                noise[i][j] = pnoise2(i/size, j/size, octaves=octaves, persistence=persistence, lacunarity=lacunarity, repeatx=size, repeaty=size, base=i)
            im.set_array(noise)
        return [im]

    ani = FuncAnimation(fig, animate, frames=100, blit=True)
    plt.show()
    
def anim4():
    size = 256
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0

    noise = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            noise[i][j] = pnoise2(i/size, j/size, octaves=octaves, persistence=persistence, lacunarity=lacunarity)

    plt.imshow(noise, cmap='gray')
    plt.show()

if __name__ == '__main__':
    # anim1()
    # anim2()
    anim3()
    # anim4()
    