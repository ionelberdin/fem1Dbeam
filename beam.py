#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np


def symmetric_matrix(elements):
    """
    Build a symmetric square matrix from its
    upper diagonal elements passed as a flat list:
    [[1, 2, 3],
     [ , 4, 5],  =>  [1, 2, 3, 4, 5, 6]
     [ ,  , 6]]

    >>> symmetric_matrix([1, 2, 3, 4, 5, 6])
    [[1, 2, 3],
     [2, 4, 5],
     [3, 5, 6]]
    """

    n = len(elements)
    N = int((-1 + np.sqrt(1 + 9 * n)) / 2)
    if n != (N + 1) * N / 2:
        raise Exception()

    triangular = np.zeros((N, N))
    triangular[np.triu_indices(N)] = np.array(elements)

    return triangular + np.triu(triangular, 1).T


class BasicBeam(object):
    """
    This class doesn't implement any particular case.
    BasicBeam integrates common methods and properties of:
        AxialBeam
        BendingBeam
        TwistingBeam
    Property methods 'K_e' and 'M_e' are intentionally not implemented.
    """

    def __init__(self, length, section, material, BCs=[], nodes=10):
        """
        Use coherent unit system for length, section and material.
        """

        self.length = length
        self.section = section
        self.material = material
        self.nodes = nodes
        self.BCs = BCs

    @property
    def K_e(self):
        """
        [[K_e]]: Element Stiffness Matrix
        This method needs to be implemented by
        classes that inherit from BasicBeam
        """

        raise NotImplementedError()

    @property
    def M_e(self):
        """
        [[M_e]]: Element Mass Matrix
        This method needs to be implemented by
        classes that inherit from BasicBeam
        """

        raise NotImplementedError()

    @property
    def K(self):
        """
        [[K]]: Beam Stiffness Matrix
        """

        element = self.K_e
        return self.assemble_matrices([element for _ in range(self.nodes)])

    @property
    def M(self):
        """
        [[M]]: Beam Mass Matrix
        """

        element = self.M_e
        return self.assemble_matrices([element for _ in range(self.nodes)])

    @staticmethod
    def assemble_matrices(matrices):
        """
        Assemble all element matrices to create the beam full matrix
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

    def get_modes(self):
        """
        Compute eigenvalues and eigenvectors and process them
        to return them as natural frequencies and normal modes.
        """

        K = self.K
        M_inverse = np.linalg.inv(self.M)

        # Build problem matrix
        problem_matrix = np.dot(M_inverse, K)

        # Impose Boundary Conditions
        BCs = [x if x >= 0 else len(self.BCs) + x for x in self.BCs]
        problem_matrix = np.delete(np.delete(problem_matrix, BCs, 0), BCs, 1)

        # Calculate eigenvalues & eigenvectors (as a column matrix)
        eigval, eigvec = np.linalg.eig(problem_matrix)

        # add missing values in eigenvectors (the ones affected by BCs)
        BCs = [x - i for i, x in enumerate(BCs)]
        zeros = np.zeros((len(BCs), eigvec.shape[1]))
        eigvec = np.insert(eigvec, BCs, zeros, 0).T

        # make normal modes point upwards at the end of the beam
        for i, x in enumerate(eigvec):
            eigvec[i] = -1 * x if x[-1] < 0 else x

        modes = zip(eigval, eigvec)

        # calculate frequency (eigenvalue is squared circular frequency)
        modes = map(lambda x: (np.sqrt(x[0]) / (2 * np.pi), x[1]), modes)

        # Order by ascending frequency
        modes.sort(key=lambda x: x[0])

        return modes


class AxialBeam(BasicBeam):
    """
    This class solves the 1D beam axial case.
    Shape functions for this case are linear:
        f1(x) = 1 - x
        f2(x) = x
        x defined in [0, 1]
        f1'(x) = -1
        f2'(x) = 1
    """

    @property
    def K_e(self):
        """
        E * A / L_e * [[ 1, -1],
                      [-1,  1]]
        """

        L_e = self.length / (self.nodes - 1.)
        E = self.material.E
        A = self.section.A
        triu = [1, -1, 1]

        return E * A / L_e * symmetric_matrix(triu)

    @property
    def M_e(self):
        """
        rho * A * L_e / 6 * [[2, 1],
                            [1, 2]]
        """

        L_e = self.length / (self.nodes - 1.)
        A = self.section.A
        rho = self.material.rho
        triu = [2, 1, 2]

        return A * rho * L_e / 6 * symmetric_matrix(triu)


class BendingBeam(BasicBeam):
    """ """

    @property
    def K_e(self):
        """ """
        L_e = self.length / (self.nodes - 1.)
        E = self.material.E
        I = self.section.Ix
        triu = [12, 6, -12, 6, 4, -6, 2, 12, -6, 4]
        return E * I / L_e**3 * symmetric_matrix(triu)

    @property
    def M_e(self):
        """ """
        L_e = self.length / (self.nodes - 1.)
        A = self.section.A
        rho = self.material.rho
        triu = [156, 22, 54, -13, 4, 13, -3, 156, -22, 4]
        return A * rho * L_e / 420 * symmetric_matrix(triu)


class TwistingBeam(BasicBeam):
    """ """

    @property
    def K_e(self):
        """ """
        L = self.length / (self.nodes - 1.)
        G = self.material.G
        J = self.section.J
        triu = [1, -1, 1]
        return G * J / L * symmetric_matrix(triu)

    @property
    def M_e(self):
        """ """
        L = self.length / (self.nodes - 1.)
        Iy = self.section.Iy
        rho = self.material.rho
        triu = [2, 1, 2]
        return Iy * rho * L / 6 * symmetric_matrix(triu)

    @property
    def cantilever_natural_frequencies(self):
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


def test_things():
    eig = []
    A, B = 10, 101
    for i in range(A, B):
        beam = test_twisting_beam(i)
        beam.BCs = [0, 1]
        eig.append(beam.get_eig())
    expected = []
    for i, f in zip(range(6), beam.cantilever_natural_frequencies):
        plt.plot(range(A, B), map(lambda x: abs(x[i][0] - f) / f * 100, eig))
        expected.append(f)
    plt.legend(["%.1f" % x for x in expected])
    plt.grid()
    plt.show()
    # test_eig(eig, 6)


def test_cantilever_bending():
    beam = test_bending_beam(10)
    for i, f in zip(range(10), beam.cantilever_natural_frequencies):
        print i, f


def test_cantilever_twisting():
    beam = test_twisting_beam(10)
    for i, f in zip(range(10), beam.cantilever_natural_frequencies):
        print i, f


def main():
    test_things()


if __name__ == "__main__":
    main()
