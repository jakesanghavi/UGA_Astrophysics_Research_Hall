from N_Body_Simulator.Body import Body
from astropy.constants import sigma_sb, L_sun
import numpy as np
from astropy import units as u

# Class representation of the planet body type
class Planet(Body):
    def __init__(self, name="Planet", mass=1.0, radius=1.0,
                 position=None, velocity=None, albedo=0.0,
                 semimaj=None, ecc=None, inc=None, longascend=None,
                 argper=None, meananom=None, G=None, totalMass=None):

        if semimaj is not None:
            # Orbital-element constructor
            super().__init__(name, mass, radius)
        else:
            # Cartesian constructor
            super().__init__(name, mass, radius, position, velocity)

        self.type = "Planet"
        self.albedo = albedo
        self.temperature = 0.0
        self.reflectiveLuminosity = 0.0
        self.luminosity = 0.0

    # Getter setters
    def setEquilibriumTemperature(self, temp):
        self.temperature = temp

    def getEquilibriumTemperature(self):
        return self.temperature

    def setReflectiveLuminosity(self, lum):
        self.reflectiveLuminosity = lum

    def setAlbedo(self, alb):
        self.albedo = alb

    def getAlbedo(self):
        return self.albedo

    def setLuminosity(self, lum):
        self.luminosity = lum

    def getLuminosity(self):
        return self.luminosity

    # Return the luminosity of the planet
    def calcLuminosity(self):
        # Convert constants to plain floats
        pi_val = np.pi
        AU_val = (1 * u.au).value
        sigma_val = sigma_sb.value
        Lsun_val = L_sun.value

        # Thermal emission
        self.luminosity = (
            4.0 * pi_val * (self.radius * AU_val)**2 *
            sigma_val * self.temperature**4 / Lsun_val
        )

        # Add reflected starlight
        self.luminosity += self.reflectiveLuminosity
