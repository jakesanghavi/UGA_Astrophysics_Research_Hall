import math
import numpy as np
from astropy import constants as const  # Using astropy constants
from N_Body_Simulator.Body import Body  # assumes Body is in body.py

# Use unitless astropy constants
pi = math.pi
rsol = const.R_sun.value
lsol = const.L_sun.value
stefan = const.sigma_sb.value
hplanck = const.h.value
c = const.c.value
k_B = const.k_B.value


# Class to represent the star body type
class Star(Body):
    def __init__(self, name="Star", mass=1.0, radius=1.0,
                 position=None, velocity=None,
                 Teff=5000.0, nlambda=1000, luminosity=None,
                 semimaj=None, ecc=None, inc=None, longascend=None,
                 argper=None, meananom=None, G=None, totalMass=None):
        """
        Flexible constructor for Star:
        - Either pass position/velocity (Cartesian) OR orbital elements (semimaj, ecc, ...).
        - Luminosity can be computed (Stefan-Boltzmann) or given directly.
        """

        # Initialize as a Body
        super().__init__(name=name, mass=mass, radius=radius,
                         position=position, velocity=velocity,
                         semimaj=semimaj, ecc=ecc, inc=inc,
                         longascend=longascend, argper=argper, meananom=meananom,
                         G=G, total_mass=totalMass)

        self.type = "Star"
        self.Teff = Teff
        self.nlambda = nlambda
        self.I_lambda = np.zeros(nlambda)

        # Spectrum defaults from C++ version
        self.lambda_min = 1e-9
        self.lambda_max = 3e-6

        # Habitable zone bounds
        # These are set in a few lines
        self.innerHZ = 0.0
        self.outerHZ = 0.0

        # Luminosity handling
        if luminosity is not None:
            self.luminosity = luminosity
        else:
            self.calcLuminosityStefanBoltzmann()

        # Habitable zone calculation
        self.calculateSingleHZ()

    # Getter setters
    def setLuminosity(self, lum): self.luminosity = lum
    def getLuminosity(self): return self.luminosity

    def setTeff(self, T): self.Teff = T
    def getTeff(self): return self.Teff

    def setLambdaMin(self, l): self.lambda_min = l
    def getLambdaMin(self): return self.lambda_min

    def setLambdaMax(self, l): self.lambda_max = l
    def getLambdaMax(self): return self.lambda_max

    def setNLambda(self, n):
        self.nlambda = n
        self.I_lambda = np.zeros(n)

    def getNLambda(self): return self.nlambda

    def setInnerHZ(self, r): self.innerHZ = r
    def getInnerHZ(self): return self.innerHZ

    def setOuterHZ(self, r): self.outerHZ = r
    def getOuterHZ(self): return self.outerHZ

    def getILambda(self): return self.I_lambda

    # Create a copy of the star
    # This contains different reference pointers and is safe to change
    def Clone(self):
        return Star(
            name=self.name,
            mass=self.mass,
            radius=self.radius,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            Teff=self.Teff,
            nlambda=self.nlambda,
            luminosity=self.luminosity
        )

    # Return luminosity in units L_sol
    def calcLuminosityStefanBoltzmann(self):
        radstarSI = self.radius * rsol
        self.luminosity = 4.0 * pi * radstarSI**2 * stefan * self.Teff**4
        self.luminosity /= lsol

    # Compute the blackbody spectrum of the star
    def calculateBlackbodySpectrum(self):
        """Compute Planck spectrum between lambda_min and lambda_max."""
        dlambda = (self.lambda_max - self.lambda_min) / float(self.nlambda)

        for i in range(self.nlambda):
            lam = self.lambda_min + i * dlambda
            exponent = math.exp(hplanck * c / (lam * k_B * self.Teff)) - 1.0
            self.I_lambda[i] = (2.0 * hplanck * c**2) / (lam**5 * exponent)

    # Retrieve the peak wavelength via Wien's law in units cm
    def calculatePeakWavelength(self):
        if self.Teff > 0.0:
            return 2.8977685e-1 / self.Teff
        else:
            return 0.0

    # Compute the inner and outer habitable zone in regions AU
    def calculateSingleHZ(self):
        Sinner = 4.19e-8 * self.Teff**2 - 2.139e-4 * self.Teff + 1.268
        Souter = 6.19e-9 * self.Teff**2 - 1.319e-5 * self.Teff + 0.2341

        if Sinner > 0:
            self.innerHZ = math.sqrt(self.luminosity / Sinner)
        else:
            self.innerHZ = 0.0

        if Souter > 0:
            self.outerHZ = math.sqrt(self.luminosity / Souter)
        else:
            self.outerHZ = 0.0
