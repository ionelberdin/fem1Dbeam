#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from section import ThickLine, Section

# rectangular section
width = 2.5  # m
height = 0.5  # m
t_spar = 0.05  # m
t_skin = 0.03  # m

Nx = 100
Nz = 20

x = np.linspace(0, width, Nx)
z = np.linspace(0, height, Nz)

lines = [
    ThickLine(zip(x, np.zeros(Nx)), t_skin),
    ThickLine(zip(width*np.ones(Nz), z), t_spar),
    ThickLine(zip(x[::-1], height*np.ones(Nx)), t_skin),
    ThickLine(zip(np.zeros(Nz), z[::-1]), t_spar)
]

rectangular_section = Section(lines)


if __name__ == "__main__":
    # draw section
    fig = rectangular_section.plot()
    ax = plt.gca()
    ax.set_xlim((-0.5, 3))
    ax.set_ylim((-0.25, 1.5))
    ann = ax.annotate(r'$I_1 = x$', xy=(1.5, 1), annotation_clip=False)
    plt.show()
