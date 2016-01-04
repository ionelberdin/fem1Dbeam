import sys

import numpy as np

sys.path.append("../")  # noqa
from section import Section, ThickLine


def asymmetric_section():
    # half circle
    theta = np.linspace(0, np.pi, 100)
    points_a = np.array([[np.sin(x), np.cos(x)] for x in theta])
    a = ThickLine(points_a, 0.03)

    # lower wall
    points_b = np.array([[x, -1] for x in np.linspace(0, 2, 50)])
    b = ThickLine(points_b, 0.1)

    # right wall
    points_c = np.array([[2, x] for x in np.linspace(-1, 1, 100)])
    c = ThickLine(points_c, 0.05)

    # upper wall
    points_d = np.array([[x, 1] for x in np.linspace(2, 0, 50)])
    d = ThickLine(points_d, 0.01)

    return Section(lines=[a, b, c, d])


if __name__ == "__main__":
    # debug and test
    print "Creating asymmetric section"
    section = asymmetric_section()

    print "Section created!"
    print section

    plt, fig = section.plot()

    plt.axis("equal")
    plt.show()
