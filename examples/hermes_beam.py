#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath("../"))  # noqa

from beam import BendingBeam, TwistingBeam
from materials import IsotropicMaterial, aluminum
from analytical_beam_modes import cantilever_bending_modes
from analytical_beam_modes import cantilever_twisting_modes

from hermes_section import section as S

mat = IsotropicMaterial(**aluminum['6061-T6'])

L = 1.
styles = ['r*-', 'g.-', 'bo-']
bstyles = ['k+-', 'r.-', 'g^-', 'b*-', 'cd-', 'mo-']

bending_beam = BendingBeam(L, S, mat)


def style_gen(token, color=0, line=0):
    colors = 'krgb'
    tokens = 'o.+vds*'
    lines = ['-', '-.', '--', ':']
    color = colors[color % len(colors)]
    token = tokens[token % len(tokens)]
    line = lines[line % len(lines)]
    return "{0}{1}{2}".format(color, token, line)


def subplot_modes(modes):
    """
    modes is a list of lists (one for each natural frequency)
    each sublist contains 3-comp tuples with (label, freq, mode)
    """
    n = len(modes)
    print "\nPloting {0} first modes:".format(n)

    # initiate plot
    fig, axarr = plt.subplots(n, sharex=True)

    legend_kwargs = {'loc': 'center left', 'bbox_to_anchor': (1, 0.5)}
    for i, mode_container in enumerate(modes):
        print "\nMode {0}".format(i+1)

        legend = map(lambda x: " ".join((x[0], "(f={0:.2f} Hz)".format(x[1]))),
                     mode_container)
        modes_i = map(lambda x: x[2], mode_container)

        for j, mode in enumerate(modes_i):
            x = np.linspace(0, L, len(mode))
            axarr[i].plot(x, mode, style_gen(j))
        axarr[i].legend(legend, **legend_kwargs)
        axarr[i].grid()

    axarr[n-1].set_xlabel(r"Length of the beam ($y$ axis) in meters")
    axarr[n/2].set_ylabel("Normalized deflection")
    plt.subplots_adjust(right=0.6)


def plot_bending_modes(nodes, n=3):
    print "\nPloting {0} first bending modes:".format(n)
    f, axarr = plt.subplots(n, sharex=True)
    for j, N in enumerate(nodes):
        beam = BendingBeam(L, S, mat, BCs=[0, 1], nodes=N)
        eigs = beam.get_modes()

        for i, (eigval, eigvec) in enumerate(eigs[:n]):
            print "N:", N, ", f=", eigval, "Hz"
            y = eigvec[::2] / eigvec[-2]
            x = np.linspace(0, L, len(y))
            axarr[i].plot(x, y, styles[j])

    axarr[n-1].set_xlabel(r"Length of the beam ($y$ axis) in meters")
    axarr[n/2].set_ylabel("Normalized vertical deflection")

    for i, (f, mode) in enumerate(cantilever_bending_modes(beam, n, points=20)):
        print f, "Hz"
        x = mode.T[0]
        y = mode.T[1]
        axarr[i].plot(x, y, styles[2])
        axarr[i].grid()
    axarr[n/2].legend(["N = 8", "N = 20", "static"],
                      loc='center right',
                      bbox_to_anchor=(1.3, 0.5))

    plt.subplots_adjust(right=0.8)


def plot_twisting_modes(nodes, n=3):
    print "\nPloting {0} first twisting modes:".format(n)
    fig, axarr = plt.subplots(n, sharex=True)
    fs = []
    for j, N in enumerate(nodes):
        beam = TwistingBeam(L, S, mat, BCs=[0], nodes=N)
        modes = beam.get_modes()[:n]

        print "\nN:", N
        for i, (freq, mode) in enumerate(modes):
            print "mode {0}, f = {1:.2f} Hz".format(i+1, freq)

            axarr[i].plot(np.linspace(0, L, len(mode)), mode, styles[j % 3])
        fs.append(map(lambda x: x[0], modes))

    axarr[n-1].set_xlabel(r"Length of the beam ($y$ axis) in meters")
    axarr[n/2].set_ylabel("Normalized angular deflection")
    fi = []
    modes = cantilever_twisting_modes(beam, n, points=20)
    print "\nStatic:"
    for i, (freq, mode) in enumerate(modes):
        print "mode {0}, f = {1:.2f} Hz".format(i+1, freq)
        fi.append(freq)
        x = mode.T[0]
        y = mode.T[1]
        axarr[i].plot(x, y, styles[2])
        axarr[i].grid()
    axarr[n/2].legend(["N = 8", "N = 20", "static"],
                      loc='center right',
                      bbox_to_anchor=(1.3, 0.5))
    plt.subplots_adjust(right=0.8)
    fs = [fi] + fs
    fs = np.array(fs).T
    f0 = fs[:, 0:1] * np.ones((1, 2))
    errs = np.abs(fs[:, 1:] - f0) / f0 * 100
    with open("twisting.out", "w") as outfile:
        for a, b in zip(fs, errs):
            line = "\t".join(["%.2f" % p for p in np.hstack((a, b))])
            outfile.writelines(line + "\n")


def plot_twisting_errors(nodes, n=3):
    print "\nEvaluating Twisting errors for {0} first modes:".format(n)
    plt.figure("twisting_errors")
    freqs_i = [[] for _ in range(n)]
    for N in nodes:
        beam = TwistingBeam(L, S, mat, BCs=[0], nodes=N)
        for i, (freq, mode) in enumerate(beam.get_modes()[:n]):
            freqs_i[i].append(freq)

    modes = cantilever_twisting_modes(beam, n)
    freqs = map(lambda x: x[0], modes)

    for i, (f, f_i) in enumerate(zip(freqs, freqs_i)):
        plt.plot(nodes, abs(f-f_i)/f*100, bstyles[i])

    plt.legend(["$f = %.2f Hz$" % f for f in freqs])
    plt.xlabel(r'Number of nodes')
    plt.ylabel(r'Relative error $\epsilon = |f_n - f|/f*100$ (%)')
    plt.grid()


if __name__ == "__main__":
    nodes = [8, 20]
    n = 6
    legend = ["N = {0}".format(i) for i in nodes] + ["analytic"]
    bending_beam_8 = BendingBeam(L, S, mat, BCs=[0, 1], nodes=8)
    bending_beam_20 = BendingBeam(L, S, mat, BCs=[0, 1], nodes=20)
    bending_modes_8 = map(lambda x: ("N = 8", x[0], x[1][::2]),
                          bending_beam_8.get_modes()[:n])
    bending_modes_20 = map(lambda x: ("N = 20", x[0], x[1][::2]),
                           bending_beam_20.get_modes()[:n])
    bending_modes = cantilever_bending_modes(bending_beam_8, n)
    bending_modes = map(lambda x: ("analytic", x[0], x[1].T[1]),
                        bending_modes)

    subplot_modes(zip(bending_modes_8, bending_modes_20, bending_modes))

    # plot_bending_modes(nodes, n)
    # plot_twisting_modes(nodes, n)
    # plot_twisting_errors(range(8, 101, 2), 6)
    plt.show()
