# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 14:09:36 2015

Data obtained from: http://asm.matweb.com/
ASM: Aerospace Specification Metals Inc.

Titanium alloys: http://www.aerospacemetals.com/titanium-distributor.html
Aluminum alloys: http://www.aerospacemetals.com/aluminum-distributor.html

@author: Ion
"""


class IsotropicMaterial(object):
    def __init__(self, E, nu, rho, name):
        self.name = name
        self.E = E  # Elastic modulus
        self.nu = nu  # Poisson's ratio
        self.G = self.get_G()  # Shear modulus (aka: mu)
        self.rho = rho

    def get_G(self):
        """ 2 * G * (1 + nu) = E
            G = E / (1 + nu) / 2 """
        return self.E / (1. + self.nu) / 2


# materials DB
aluminum = {
    "6061-T6": {'E': 70e9, 'nu': 0.33, 'rho': 2.7e3, 'name': "Aluminum 6061-T6"}
}

steel = {}

titanium = {
    "MTU031": {'name': "Titanium Grade 3, Annealed", 'E': 104e9, 'nu': 0.34}
}


if __name__ == "__main__":
    alum = IsotropicMaterial(**aluminum['6061-T6'])
    print alum.G
