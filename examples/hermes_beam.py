#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath("../"))  # noqa

from beam import BendingBeam, TwistingBeam
from materials import IsotropicMaterial, aluminum
from analytical_beam_modes import cantilever_axial_modes
from analytical_beam_modes import cantilever_bending_modes
from analytical_beam_modes import cantilever_twisting_modes

from hermes_section import section as S

mat = IsotropicMaterial(**aluminum['6061-T6'])

L = 1.
styles = ['r*-', 'g.-', 'bo-']
bstyles = ['k+-', 'r.-', 'g^-', 'b*-', 'cd-', 'mo-']

bending_beam = BendingBeam(L, S, mat)


def plot_bending_modes(nodes, n=3):
    f, axarr = plt.subplots(n, sharex=True)
    for j, N in enumerate(nodes):
        beam = BendingBeam(L, S, mat, N)
        beam.BCs = [0, 1]
        eigs = beam.get_modes()

        for i, (eigval, eigvec) in enumerate(eigs[:n]):
            print "N:", N, ", f=", eigval, "Hz"
            y = eigvec[::2] / eigvec[-2]
            x = np.linspace(0, L, len(y))
            axarr[i].plot(x, y, styles[j])

    axarr[n-1].set_xlabel(r"Length of the beam ($y$ axis) in meters")
    axarr[n/2].set_ylabel("Normalized vertical deflection")

    for i, (f, mode) in enumerate(cantilever_bending_modes(beam, n)):
        print f, "Hz"
        x = np.linspace(0, L, len(mode))
        axarr[i].plot(x, mode, styles[2])
        axarr[i].grid()
    axarr[n/2].legend(["N = 8", "N = 20", "static"],
                      loc='center right',
                      bbox_to_anchor=(1.25, 0.5))


def plot_twisting_modes(nodes, n=3):
    fig, axarr = plt.subplots(n, sharex=True)
    fs = []
    for j, N in enumerate(nodes):
        beam = TwistingBeam(L, S, mat, N)
        beam.BCs = [0]
        modes = beam.get_modes()

        print "\nN:", N
        for i, (freq, mode) in enumerate(modes):
            print "mode {0}, f = {1} Hz".format(i, freq)

            axarr[i].plot(np.linspace(0, L, len(mode)), mode, styles[j])
        fs.append(map(lambda x: x[0], modes))

    axarr[n-1].set_xlabel(r"Length of the beam ($y$ axis) in meters")
    axarr[3].set_ylabel("Normalized angular deflection")
    fi = []
    for i, f in zip(range(n), beam.cantilever_normal_frequencies):
        print f, "Hz"
        fi.append(f)
        omega = f * 2 * np.pi
        kn = omega * np.sqrt(mat.rho*S.Iy/mat.G/S.J)
        x = np.linspace(0, 1, 20)
        y = np.sin(kn * x)
        axarr[i].plot(x, y / y[-1], styles[2])
        axarr[i].grid()
    axarr[3].legend(["N = 8", "N = 20", "static"],
                    loc='center right',
                    bbox_to_anchor=(1.25, 0.5))
    fs = [fi] + fs
    fs = np.array(fs).T
    f0 = fs[:, 0:1] * np.ones((1, 2))
    errs = np.abs(fs[:, 1:] - f0) / f0 * 100
    with open("twisting.out", "w") as outfile:
        for a, b in zip(fs, errs):
            line = "\t".join(["%.2f" % p for p in np.hstack((a, b))])
            outfile.writelines(line + "\n")


def plot_bending_errors(nodes, n=3):
    eigs = []
    for N in nodes:
        beam = TwistingBeam(L, S, mat, N)
        beam.BCs = [0]
        eigs.append(beam.get_modes())
    fs = [f for i, f in zip(range(n), beam.cantilever_normal_frequencies)]

    for i, f in enumerate(fs):
        plt.plot(nodes,
                 map(lambda x: abs(x[i][0] - f) / f * 100, eigs),
                 bstyles[i])
    plt.legend(["$f = %.2f Hz$" % f for i, f in enumerate(fs)])
    plt.xlabel(r'Number of nodes')
    plt.ylabel(r'Relative error $\epsilon = |f_n - f|/f*100$ (%)')
    plt.grid()


if __name__ == "__main__":
    nodes = [8, 20]
    n = 6
    # plot_twisting_modes(nodes, n)
    # plot_bending_modes(nodes, n)
    plot_bending_errors(range(8, 101, 2), 6)
    plt.show()
