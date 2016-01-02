#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../"))  # noqa

from section import ThickLine, Section

N = 20
a = 0.02
t_spar = 0.0015
t_skin = 0.0005

left_spar_x = np.hstack((
    np.linspace(a, 0, N),
    np.zeros(N),
    np.linspace(0, a, N)
))
left_spar_y = np.hstack((
    a * np.ones(N),
    np.linspace(a, 0, N),
    np.zeros(N)
))

right_spar_x = np.hstack((
    np.linspace(2*a, 3*a, N),
    3 * a * np.ones(N),
    np.linspace(3*a, 2*a, N)
))
right_spar_y = np.hstack((
    np.zeros(N),
    np.linspace(0, a, N),
    a * np.ones(N)
))

lines = [
    ThickLine(zip(left_spar_x, left_spar_y), t_spar),
    ThickLine(zip(np.linspace(a, 2*a, N), np.zeros(N)), t_skin),
    ThickLine(zip(right_spar_x, right_spar_y), t_spar),
    ThickLine(zip(np.linspace(2*a, a, N), a*np.ones(N)), t_skin),
]

section = Section(lines)


def plot():
    section.plot()
    ax = plt.gca()
    ax.axis("equal")
    ax.grid()
    ax.set_xlim((-a/4, 13*a/4))
    ax.set_ylim((-a/4, 5*a/4))
    ax.set_xlabel(r'horizontal dimension ($m$)')
    ax.set_ylabel(r'vertical dimension ($m$)')
    plt.show()


def section_properties():
    # print data
    print section.Ix, section.Iy, section.Iz
    print section.A, section.J
