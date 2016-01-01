#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt  # noqa
from matplotlib import cm  # noqa

import numpy as np


def symmetric_matrix(elements):
    """ """
    n = len(elements)
    N = int((-1 + np.sqrt(1 + 9 * n)) / 2)
    if n != (N + 1) * N / 2:
        raise Exception()
    triangular = np.zeros((N, N))
    triangular[np.triu_indices(N)] = np.array(elements)
    return triangular + np.triu(triangular, 1).T


class BasicBeam(object):
    """ """

    def __init__(self, length, section, material, nodes=10):
        self.length = length
        self.section = section
        self.material = material
        self.nodes = nodes
        self.BCs = []

    @property
    def element_K(self):
        """ K: Stiffness Matrix """
        raise NotImplementedError()

    @property
    def element_M(self):
        """ M: Mass Matrix """
        raise NotImplementedError()

    @property
    def full_K(self):
        """ K: Stiffness Matrix """
        element = self.element_K
        return self.assemble_matrices([element for _ in range(self.nodes)])

    @property
    def full_M(self):
        """ M: Mass Matrix """
        element = self.element_M
        return self.assemble_matrices([element for _ in range(self.nodes)])

    @staticmethod
    def assemble_matrices(matrices):
        """
        Assemble all the elemental matrices to form a global matrix
        Input:
            matrices: list of sqare matrices sharing shape (N, N)
        """

        N = len(matrices)
        n = matrices[0].shape[0] / 2
        full_N = n * (N + 1)
        full_matrix = np.zeros((full_N, full_N))
        for i, matrix in enumerate(matrices):
            a = i * n
            b = a + 2 * n
            full_matrix[a:b, a:b] = full_matrix[a:b, a:b] + matrix

        return full_matrix

    def get_eig(self):
        K = self.full_K
        M_inverse = np.linalg.inv(self.full_M)

        # Build problem matrix
        problem_matrix = np.dot(M_inverse, K)

        # Impose Boundary Conditions
        BCs = [x if x >= 0 else len(self.BCs) + x for x in self.BCs]
        problem_matrix = np.delete(np.delete(problem_matrix, BCs, 0), BCs, 1)

        # Calculate eigenvalues & eigenvectors (as a column matrix)
        eigval, eigvec = np.linalg.eig(problem_matrix)

        # add missing values in eigenvectors (the ones affected by BCs)
        BCs = [x-i for i, x in enumerate(BCs)]
        zeros = np.zeros((len(BCs), eigvec.shape[1]))
        eigvec = np.insert(eigvec, BCs, zeros, 0).T

        # make normal modes point upwards at the end of the beam
        for i, x in enumerate(eigvec):
            eigvec[i] = -1 * x if x[-1] < 0 else x

        eig = zip(eigval, eigvec)

        # calculate frequency (eigenvalue is circular frequency squared)
        eig = map(lambda x: (np.sqrt(x[0]) / (2 * np.pi), x[1]), eig)

        # Order by ascending frequency
        eig.sort(key=lambda x: x[0])

        return eig


class BendingBeam(BasicBeam):
    """ """

    @property
    def element_K(self):
        """ """
        L = self.length / (self.nodes - 1.)
        E = self.material.E
        I = self.section.Ix
        triu = [12, 6, -12, 6, 4, -6, 2, 12, -6, 4]
        return E * I / L**3 * symmetric_matrix(triu)

    @property
    def element_M(self):
        """ """
        L = self.length / (self.nodes - 1.)
        A = self.section.A
        rho = self.material.rho
        triu = [156, 22, 54, -13, 4, 13, -3, 156, -22, 4]
        return A * rho * L / 420 * symmetric_matrix(triu)

    @property
    def cantilever_normal_frequencies(self):
        """
        cos(kn*L) * cosh(kn*L) = -1
        cosh(kn*L) >= 1
        cos(kn*L) changes sign every kn*L=pi
        """

        L, S, mat = self.length, self.section, self.material
        A, Ix = S.A, S.Ix
        E, rho = mat.E, mat.rho

        def f(x):
            return np.cos(x) * np.cosh(x) + 1
        EPS = 1e-9

        n = 0
        margin = np.pi / 2

        while True:
            ref = (2 * n + 1) * np.pi / 2
            x0, x1 = ref, ref + (-1)**n * margin
            f0, f1 = f(x0), f(x1)
            if (f0 > 0) == (f1 > 0):
                raise Exception("Wrong guess :(")

            while abs(x0 - x1) / x0 > EPS:
                x2 = (x0 + x1) / 2
                f2 = f(x2)
                if (f2 > 0) == (f0 > 0):
                    x0, f0 = x2, f2
                else:
                    x1, f1 = x2, f2

            x = (x0 + x1) / 2
            margin = abs(ref - x)
            n += 1

            yield (x / L)**2 * np.sqrt(E * Ix / rho / A) / 2 / np.pi


class TwistingBeam(BasicBeam):
    """ """

    @property
    def element_K(self):
        """ """
        L = self.length / (self.nodes - 1.)
        G = self.material.G
        J = self.section.J
        triu = [1, -1, 1]
        return G * J / L * symmetric_matrix(triu)

    @property
    def element_M(self):
        """ """
        L = self.length / (self.nodes - 1.)
        Iy = self.section.Iy
        rho = self.material.rho
        triu = [2, 1, 2]
        return Iy * rho * L / 6 * symmetric_matrix(triu)

    @property
    def cantilever_normal_frequencies(self):
        S, mat = self.section, self.material
        L, J, Iy = self.length, S.J, S.Iy
        G, rho = mat.G, mat.rho
        n = 0
        while True:
            x = (2 * n + 1) * np.pi / 2
            n += 1
            yield (x / L) * np.sqrt(G * J / rho / Iy) / 2 / np.pi


def test_material():
    from materials import IsotropicMaterial, aluminum

    return IsotropicMaterial(**aluminum["6061-T6"])


def test_section():
    from section import Section, ThickLine  # noqa

    x0 = np.linspace(-0.5, 0.5, 100)
    z0 = np.linspace(0.25, 0.25, 100)
    x1 = np.linspace(0.5, 0.5, 100)
    z1 = np.linspace(0.25, -0.25, 100)
    x2 = np.linspace(0.5, -0.5, 100)
    z2 = np.linspace(-0.25, -0.25, 100)
    x3 = np.linspace(-0.5, -0.5, 100)
    z3 = np.linspace(-0.25, 0.25, 100)
    a0 = ThickLine(zip(x0, z0), 0.03)
    a1 = ThickLine(zip(x1, z1), 0.05)
    a2 = ThickLine(zip(x2, z2), 0.03)
    a3 = ThickLine(zip(x3, z3), 0.05)

    return Section([a0, a1, a2, a3])


def test_bending_beam(N=10):
    section = test_section()
    mat = test_material()

    return BendingBeam(10., section, mat, N)


def test_twisting_beam(N=10):
    section = test_section()
    mat = test_material()

    return TwistingBeam(10., section, mat, N)


def test_eig(eig, N=None):
    plt.figure(1)
    # plt.figure(2)
    for omega, mode in eig[:N]:
        plt.figure(1)
        plt.plot(mode[::])
        # plt.figure(2)
        # plt.plot(mode[1::2])
    # plt.figure(3)
    # plt.plot(map(lambda x: x[0], eig))
    plt.show()


def plot_matrix(matrix):
    plt.figure("full_K", figsize=(12, 10), dpi=200)
    Kmax = np.max(np.abs(matrix))
    N = matrix.shape[0]
    plt.xlim = (0, N)
    plt.ylim = (0, N)
    im = plt.imshow(matrix,  # noqa
                    interpolation='None',
                    cmap=cm.bwr,
                    extent=[0, N, 0, N],
                    aspect=0.8,
                    origin='upper',
                    vmin=-Kmax,
                    vmax=Kmax)
    cb = plt.colorbar()
    cb.set_label(r"$K$")

    plt.tight_layout()  # = (0, N)
    plt.axis("equal")
    plt.title(r"Full Beam Stiffness Matrix $K$")
    plt.xlabel(r"columns")
    plt.ylabel(r"rows")
    plt.grid()
    plt.show()


def test_things():
    eig = []
    A, B = 10, 101
    for i in range(A, B):
        beam = test_twisting_beam(i)
        beam.BCs = [0, 1]
        eig.append(beam.get_eig())
    expected = []
    for i, f in zip(range(6), beam.cantilever_normal_frequencies):
        plt.plot(range(A, B), map(lambda x: abs(x[i][0] - f) / f * 100, eig))
        expected.append(f)
    plt.legend(["%.1f" % x for x in expected])
    plt.grid()
    plt.show()
    # test_eig(eig, 6)


def test_cantilever_bending():
    beam = test_bending_beam(10)
    for i, f in zip(range(10), beam.cantilever_normal_frequencies):
        print i, f


def test_cantilever_twisting():
    beam = test_twisting_beam(10)
    for i, f in zip(range(10), beam.cantilever_normal_frequencies):
        print i, f


def main():
    test_things()


if __name__ == "__main__":
    main()
