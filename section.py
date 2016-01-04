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
    vectors = np.array(vectors)
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
        self.calculate_ppal_axes()

    def __str__(self):
        """
        Main details of the section as a summary to be printed.

        This function is called when a section instance is casted to string
        explicitely or implicitely:
        >>> section_instance = Section(some_lines)
        >>> print section_instance
        >>> some_var = str(section_instance)
        """
        text = ["\nSection details:",
                "Area, A = {obj.A} u^2",
                "Inner area, S = {obj.S} u^2",
                "Torsion constant, J = {obj.J} u^2",
                "Center of Gravity, CG = {obj.cg}",
                "Area moments of inertia in XZ axes at CG:",
                "\tI_x = {obj.Ix} u^4",
                "\tI_z = {obj.Iz} u^4",
                "\tI_xz= {obj.Ixz} u^4",
                "Principal area moments of inertia:",
                "\tI_1 = {obj.I_1} u^4 at {obj.ppal_angle_deg} deg of OX",
                "\tI_2 = {obj.I_2} u^4 at {obj.ppal_angle_deg} deg of OZ",
                "Perpendicular moment of inertia through CG:",
                "\tI_y = {obj.Iy} u^4 = I_x + I_z = I_1 + I_2"]

        return "\n".join(text).format(obj=self)

    def calculate_ppal_axes(self):
        """
        Calculate principal axes and area moments of inertia
        """

        # Since I is a symmetric matrix, eigh is used (h: Hermitian)
        eigvals, eigvecs = np.linalg.eigh(self.I)

        # Descending order, higher first
        if eigvals[1] > eigvals[0]:
            eigvecs = eigvecs[:, ::-1]
            eigvals = eigvals[::-1]

        self.I_1 = eigvals[0]
        self.I_2 = eigvals[1]
        vec = np.resize(eigvecs[:, 0], (2,))
        self.ppal_angle = np.arcsin(np.cross([1, 0], vec) / np.linalg.norm(vec))
        self.ppal_angle_deg = self.ppal_angle * 180 / np.pi

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

        xbox = np.array(plt.xlim())
        ybox = np.array(plt.ylim())
        xmargin = np.abs(xbox - self.cg[0])
        ymargin = np.abs(ybox - self.cg[1])
        cos = np.cos(self.ppal_angle)
        sin = np.sin(self.ppal_angle)
        I_1_scale = np.hstack((xmargin / cos, ymargin / sin)) / self.I_1
        I_2_scale = np.hstack((xmargin / sin, ymargin / cos)) / self.I_2
        I_scale = np.min(np.abs(np.hstack((I_1_scale, I_2_scale)))) * 0.7

        # plot inertia principal axes
        I_1 = self.I_1 * I_scale
        I_2 = self.I_2 * I_scale
        vectors = [I_1 * np.array([cos, sin]), I_2 * np.array([-sin, cos])]
        plot_vectors(self.cg, vectors, style='r-')

        theta = np.linspace(0, 2*np.pi, 100)
        I_ellipse = np.vstack(([I_1*np.cos(x), I_2*np.sin(x)] for x in theta))
        I_ellipse = np.dot(I_ellipse, [[cos, sin], [-sin, cos]])
        I_ellipse += np.dot(np.ones((100, 1)), self.cg.reshape((1, 2)))
        I_ellipse = I_ellipse.T
        plt.plot(I_ellipse[0], I_ellipse[1], 'r:')

        # plot moments of inertia of local reference system at CG
        Ix = np.array([1, 0]) * self.Ix * I_scale
        Iz = np.array([0, 1]) * self.Iz * I_scale
        plot_vectors(self.cg, [Ix, Iz], 'k-')

        # invert X axis
        margin = np.array([-0.05, 0.05])
        plt.xlim((xbox * (1 + margin))[::-1])
        plt.ylim((ybox * (1 + margin)))

        print plt.xlim(), plt.ylim()

        return plt, fig

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

    @property
    def Ixz(self):
        return self.I[0, 1]

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
