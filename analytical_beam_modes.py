import numpy as np
from scipy.optimize import fsolve


def cantilever_axial_modes(beam, N=10):
    """
    Natural frequencies come from solutions of:
        cos(kn*L) = 0
    Modes are given by:
        f(x) = C * sin(kn * x)
        where:
            C is a constant
            kn is sort of a natural frequency
            x is the axial local coordinate of the beam
    Input:
        beam: Beam instance
        N: number of modes requested (defaults to 10)
    Output:
        list of tuples containing:
            [0]: natural frequency
            [1]: normal mode as a Numpy array of 100 elements
    """

    L = beam.length
    mat = beam.material

    physical_constant = np.sqrt(mat.E / mat.rho)

    modes = []

    for n in range(N):
        kn = (2 * n + 1) * np.pi / 2 / L
        freq = kn * physical_constant / (2 * np.pi)
        mode = np.sin(kn * np.linspace(0, L, 100))

        modes.append((freq, mode / mode[-1]))

    return modes


def cantilever_bending_modes(beam, N=10, vertical='z', EPS=1e-9):
    """
    Natural frequencies come from solutions of:
        cos(kn * L) * cosh(kn * L) = -1
    That are obtained for values of kn that make cos(kn *L) < 0 because:
        cosh(kn * L) >= 1 for all values of kn
    Since cos(kn * L) changes sign every kn * L = pi, kn values will
    approach to '(2n + 1) * pi / 2 / L' as n gets higher.

    Modes are given by:
        cantilever_second_order(x, kn, L)
        where:
            C is a constant
            kn is sort of a natural frequency
            x is the axial local coordinate of the beam

    Input:
        beam: Beam instance
        N: number of modes requested (defaults to 10)
        vertical: axis in which vertical displacements happen ('x' or 'z')
        EPS: maximum numerical relative error allowed for kn

    Output:
        list of bicomponent tuples with:
            [0]: natural frequency (in Hertzs)
            [1]: normal mode as a Numpy array of 100 elements
    """

    L = beam.length
    S = beam.section
    mat = beam.material

    if vertical not in ['x', 'z']:
        raise Exception("vertical shall be either 'x' or 'z'")

    I = S.Ix if vertical == 'z' else S.Iz

    physical_constant = np.sqrt(mat.E * I / mat.rho / S.A)

    y = np.linspace(0, L, 100)

    def err(x):
        """ Function to minimize """
        return np.cos(x) * np.cosh(x) + 1

    modes = []

    # set a margin to limit the interval in which solution will be looked for
    margin = np.pi / 2

    for n in range(N):
        ref = (2 * n + 1) * np.pi / 2
        knL = ref + (-1)**n * margin

        if margin > EPS:
            knL = fsolve(func=err, x0=knL, xtol=EPS)[0]
            # update margin to shorten the search range
            margin = abs(ref - knL)

        kn = knL / L
        freq = kn**2 * physical_constant / 2 / np.pi
        mode = cantilever_fourth_order(y, kn, L)

        modes.append((freq, mode / mode[-1]))

    return modes


def cantilever_twisting_modes(beam, N=10):
    """
    Natural frequencies come from solutions of:
        cos(kn*L) = 0
    Modes are given by:
        f(x) = C * sin(kn * x)
        where:
            C is a constant
            kn is sort of a natural frequency
            x is the axial local coordinate of the beam
    Input:
        beam: Beam instance
        N: number of modes requested (defaults to 10)
    Output:
        list of tuples containing:
            [0]: natural frequency
            [1]: normal mode as a Numpy array of 100 elements
    """

    L = beam.length
    S = beam.section
    mat = beam.material

    physical_constant = np.sqrt(mat.G * S.J / mat.rho / S.Iy)

    modes = []

    for n in range(N):
        kn = (2 * n + 1) * np.pi / 2 / L
        freq = kn * physical_constant / (2 * np.pi)
        mode = np.sin(kn * np.linspace(0, L, 100))

        modes.append((freq, mode / mode[-1]))

    return modes


def cantilever_fourth_order(x, kn, L):
    """
    Solutions for equation:
        math:
        [\
            \frac{d^4 f(x)}{d x^4} - k_n^4 f(x) = cte
            f(0) = f'(0) = 0
            f''(L) = f'''(L) = 0
            \]
    F(x) = C * (g1(x) - a / b * g2(x))
    g1(x) = cos(kn * x) - cosh(kn * x)
    g2(x) = sin(kn * x) - sinh(kn * x)
    a = cos(kn * L) + cosh(kn * L)
    b = sin(kn * L) + sinh(kn * L)
    """

    g1 = np.cos(kn * x) - np.cosh(kn * x)
    g2 = np.sin(kn * x) - np.sinh(kn * x)

    a = np.cos(kn * L) + np.cosh(kn * L)
    b = np.sin(kn * L) + np.sinh(kn * L)

    return g1 - a / b * g2
