#!python
# -*- coding:utf-8 -*-

"""
    ^ z
    |      beam
    |===================-> y
   /
  v x

Mx & Mz are flector torques
My is torsion torque
N = Ny is axial force
Qx & Qz are shear forces

"""

from cached_property import cached_property

import matplotlib.pyplot as plt
import numpy as np


def plot_line(points, style="k-", linewidth=1, solid_capstyle='butt'):
    plt.plot(points.T[0], points.T[1], style,
             linewidth=linewidth,
             solid_capstyle=solid_capstyle)


def plot_point(point, style='ko'):
    plt.plot([point[0]], [point[1]], style)


def plot_vectors(point, vectors, style="k-"):
    # ax = plt.gca()
    # X, Y, U, V = zip(*[np.hstack((point, x)) for x in vectors])
    # ax.quiver(X, Y, U, V,
    #          angles='xy', scale_units='xy', scale=1)
    # ax.draw()
    for vector in vectors:
        points = np.vstack((point, point + vector)).T
        plt.plot(points[0], points[1], style)


def steiner(delta, area):
    """ Calculates Stainer contribution to Inertia Tensor
        when moving from some axes placed at the center of gravity
        to other parallel axes.

        delta = np.array(new_axes_origin - cg)

        Returns: np.matrix([[dz**2, dx*dz], [dx*dz, dx**2]]) * area

        To come back to cg-axes, area should be passed with negative sign.
    """
    # [dx, dz] => [[dz, dx]]
    delta = np.asmatrix(delta[::-1])

    return np.dot(delta.T, delta) * area


class Section(object):
    """ Let's start with a simple model.

        Assumptions about the section:
            * It's composed by many ThickLine objects (thin-wall section)
            * It's closed
            * It has only one 'cell' (just one loop)

        Reference system:
            x: horizontal axis
            z: vertical axis
            y: axis perpendicular to the section
    """

    def __init__(self, lines=[]):
        """ Class Constructor """

        self.lines = lines

    def get_ppal_axes(self):
        """ Get Inertia principal axes and values """

        # Since I is a symmetric matrix, let's use eigh (h: Hermitian)
        self.ppal_I, self.ppal_axes = np.linalg.eig(self.I)

    def check_closed(self):
        """ Returns True if all segments which compose the section
            form a closed loop. False otherwise. """

        firsts = [line.points[0] for line in self.lines]
        lasts = [line.points[-1] for line in self.lines[-1:] + self.lines[:-1]]
        diff = np.array(firsts) - np.array(lasts)
        dist = np.linalg.norm(np.max(lasts, axis=0) - np.min(lasts, axis=0))
        compare = np.linalg.norm(diff, axis=1) < dist / 1e6
        return np.all(compare)

    @cached_property
    def cg(self):
        """ Center of Gravity position [x, z] """

        if len(self.lines) == 0:
            return np.array([0, 0])

        cgs = np.array([line.cg for line in self.lines])
        areas = np.array([line.L * line.t for line in self.lines])
        weights = np.array([cg * A for cg, A in zip(cgs, areas)])
        return np.sum(weights, axis=0) / np.sum(areas)

    @cached_property
    def J(self):
        """ Assuming line.t = constant """
        return 4 * self.S ** 2 / np.sum([x.L / x.t for x in self.lines])

    def plot(self):

        fig = plt.figure()
        # plot lines
        min_t = min([x.t for x in self.lines])
        for line in self.lines:
            line.plot(t_scale=1/min_t)
            # plot line.cg for line in lines
            plot_point(line.cg, 'yo')

        # plot section.cg
        plot_point(self.cg, 'ro')

        # plot inertia principal axes
        try:
            self.ppal_axes
        except:
            self.get_ppal_axes()
        max_abs_val = max([abs(x) for x in self.ppal_I])
        scaled_ppal_I = self.ppal_I / max_abs_val
        vec_arrays = np.asarray(self.ppal_axes.T)
        vectors = [x * y for x, y in zip(vec_arrays, scaled_ppal_I)]
        plot_vectors(self.cg, vectors, style='r-')

        Ix = np.array([1, 0]) * self.I[0, 0] / max_abs_val
        Iz = np.array([0, 1]) * self.I[1, 1] / max_abs_val
        plot_vectors(self.cg, [Ix, Iz], 'k-')

        return fig

    @cached_property
    def A(self):
        return sum([x.A for x in self.lines])

    @cached_property
    def S(self):
        """ Section inner surface
            2 * S = integral(r_t(chi) * dchi), along perimeter of closed curve
            r_t = np.cross(r, t)
                r: vector from an arbitrary point to other point in the curve
                t: tangent versor in the curve """

        if not self.check_closed():
            raise Exception(self.errors('section_not_closed'))

        all_points = np.vstack((line.points[1:] for line in self.lines))
        all_points = np.vstack((self.lines[0].points[0], all_points))
        t_vec = np.diff(all_points, axis=0)
        r = all_points[:-1] - self.cg
        cross_vectors = np.cross(r, t_vec)
        return np.sum(cross_vectors) / 2

    @cached_property
    def I(self):
        """ Inertia Matrix """

        I = map(lambda x: x.I + steiner(self.cg - x.cg, x.A), self.lines)
        return reduce(lambda x, y: x + y, I)

    @property
    def Ix(self):
        return self.I[0, 0]

    @property
    def Iy(self):
        """ Moment of inertia in perpendicular direction to the section """

        return self.Ix + self.Iz

    @property
    def Iz(self):
        return self.I[1, 1]

    def errors(self, error_name, *args, **kwargs):

        errors = {
            'section_not_closed': """
                Section is not properly defined.\n
                It needs to be a closed section.\n
                Try plotting the section through the 'plot' method:\n
                \t>>> section.plot()\n
                Where 'section' may be replaced with the name of the
                object in which the Section is stored.\n
                If the plot shows a closed surface, then lines might
                not be ordered in a continuous way. They should be
                ordered so the last point of a line is the same as
                the first of the next line and so on. Finally, the
                last point of the last line needs to be the same
                as the first point of the first line.""",
            'wrong_error_name': """
                No error named '{wrong_name}' was found in Section class."""
        }

        try:
            return errors[error_name].format(*args, **kwargs)
        except:
            return self.errors('wrong_error_name', wrong_name=error_name)


class ThickLine(object):
    def __init__(self, points, t):
        """
            points: np.array of 2D mean-line points stacked as rows
            t: constant thickness
        """

        self.t = t
        self.set_points(np.array(points))
        self.area = self.L * self.t

    @property
    def A(self):
        return self.L * self.t

    @cached_property
    def cg(self):
        """ Calculate Center of Gravity """

        mp, dn = self.mean_points, self.delta_norms
        mp_by_lengths = np.array([x * y for x, y in zip(mp, dn)])

        return np.sum(mp_by_lengths, axis=0) / self.L

    @cached_property
    def I(self):
        """ Inertia tensor [[Ix, Ixz], [Ixz, Iz]] respect ThickLine.cg """

        I = np.asmatrix([[0, 0], [0, 0]])  # initialize tensor
        cg = self.cg  # localize cg to optimize calls inside the loop

        for mp, area in zip(self.mean_points, self.areas):
            # vector from line.cg to one of its mean points
            delta = np.asmatrix((mp - cg)[::-1])  # [[dz, dx]]
            # [[dz**2, dx*dz], [dx*dz, dx**2]]
            delta_mat = np.dot(delta.T, delta)
            # Ix = Ix + dz**2 * area
            # Iz = Iz + dx**2 * area
            # Ixz = Ixz + dx*dz * area
            I = I + delta_mat * float(area)

        return I

    @cached_property
    def L(self):
        return sum(self.delta_norms)

    def plot(self, style='b-', t_scale=1):
        plot_line(self.points, style,
                  linewidth=self.t * t_scale)

    def reverse_line(self):
        self.set_points(self.points[::-1])

    def set_points(self, points):
        self.points = points
        self.mean_points = (self.points[:-1] + self.points[1:]) / 2
        self.deltas = np.diff(self.points, axis=0)
        self.delta_norms = np.linalg.norm(self.deltas, axis=1)
        self.areas = self.delta_norms * self.t


if __name__ == "__main__":
    theta = np.linspace(0, np.pi, 100)
    points_a = np.array([[np.sin(x), np.cos(x)] for x in theta])
    a = ThickLine(points_a, 0.03)
    points_b = np.array([[x, -1] for x in np.linspace(0, 2, 50)])
    b = ThickLine(points_b, 0.1)
    points_c = np.array([[2, x] for x in np.linspace(-1, 1, 100)])
    c = ThickLine(points_c, 0.05)
    points_d = np.array([[x, 1] for x in np.linspace(2, 0, 50)])
    d = ThickLine(points_d, 0.01)

    section = Section(lines=[a, b, c, d])

    plt.figure()
    section.plot()

    plt.axis("equal")
    plt.ylim(np.array(plt.ylim()) * 1.1)
    plt.show()

    print("Section Inner Surface: {0} u^2".format(section.S))
