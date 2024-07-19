import numpy as np
import random as r
import pygame as pg
from math import ceil, floor


def main():
    # properties
    width = 8
    height = 8
    res = 7
    scale = 10
    line_length = 5

    def smoothstep(w):
        return 3 * w ** 2 - 2 * w ** 3

    # define interpolation
    def lerp(a_0, a_1, w):
        return a_0 + smoothstep(w) * (a_1 - a_0)
        #  return (1 - w)*a_0 + w * a_1

    def generate_gradient():
        return [[{"pos": np.array([x, y]),
                  "angle": r.random() * 2*np.pi,
                  "angle_vel": r.random() * np.pi/2 - np.pi/4}
                 for x in range(width)]
                for y in range(height)]

    def move_gradient(grad):
        grad["angle"] += grad["angle_vel"]

    def update_gradient(grads):
        [[move_gradient(grad) for grad in row] for row in grads]

    def to_vector(angle):
        return np.array([np.cos(angle), np.sin(angle)])

    def draw_grid(grads):
        grad_vector = [[to_vector(grad["angle"]) for grad in row] for row in grads]
        # Calculate noise value of each point
        for y in range(res * (height - 1)):
            for x in range(res * (width - 1)):
                # Find distance from point to all corners
                cell_x = (x / res) % 1
                cell_y = (y / res) % 1
                # Pair distance vector with corresponding gradient
                grad_pair = [
                    [grad_vector[floor(y / res)][floor(x / res)], np.array([cell_y, cell_x])],
                    [grad_vector[floor(y / res)][ceil(x / res)], np.array([cell_y, cell_x - 1])],
                    [grad_vector[ceil(y / res)][floor(x / res)], np.array([cell_y - 1, cell_x])],
                    [grad_vector[ceil(y / res)][ceil(x / res)], np.array([cell_y - 1, cell_x - 1])]
                ]
                # compute dot-product of pairs
                dot_all = [(np.dot(grad[0], grad[1])) / grid_max for grad in grad_pair]

                # Interpolate all values to one
                top_row = lerp(dot_all[0], dot_all[1], cell_x)
                bot_row = lerp(dot_all[2], dot_all[3], cell_x)

                noise_val = lerp(top_row, bot_row, cell_y)

                # Drawing
                pg.draw.rect(win, (noise_val * 255 if noise_val > 0 else 0, 0,
                                   -noise_val * 255 if noise_val < 0 else 0),
                             (scale * (line_length + x), scale * (line_length + y), scale, scale))

    # Setup
    gradients = generate_gradient()
    grid_max = np.sqrt(2)

    # Pygame Setup
    pg.init()
    win = pg.display.set_mode([scale * (2 * line_length + res * (width - 1)),
                               scale * (2 * line_length + res * (height - 1))])
    print("Resolution: " + str(win.get_width()) + " " + str(win.get_height()))
    # Main loop
    run = True
    while run:

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        update_gradient(gradients)
        draw_grid(gradients)

        pg.display.update()

    pg.quit()


if __name__ == '__main__':
    main()


