import math
import numpy as np
from utils import orbital_elements_to_state_vectors

# Template class for all bodies (planet, star, etc.) for the simulation
class Body:
    def __init__(self, name="Body", mass=1.0, radius=1.0,
                 position=None, velocity=None,
                 semimaj=None, ecc=None, inc=None,
                 longascend=None, argper=None, meananom=None,
                 G=None, total_mass=None):

        self.name = name
        self.type = "Body"
        self.mass = mass
        self.radius = radius
        self.collisionBounce = True

        # Cartesian init.
        if position is not None and velocity is not None:
            self.position = np.array(position, dtype=float)
            self.velocity = np.array(velocity, dtype=float)

        # Orbital param init. method
        elif None not in (semimaj, ecc, inc, longascend, argper, meananom, G, total_mass):
            self.position, self.velocity = orbital_elements_to_state_vectors(
                semimaj, ecc, inc, longascend, argper, meananom, G, total_mass
            )
            self.semiMajorAxis = semimaj
            self.eccentricity = ecc
            self.inclination = inc
            self.argumentPeriapsis = argper
            self.longitudeAscendingNode = longascend
            self.meanAnomaly = meananom
        else:
            self.position = np.zeros(3)
            self.velocity = np.zeros(3)

        # These must be defined by the class wrapping this
        self.acceleration = np.zeros(3)
        self.jerk = np.zeros(3)
        self.snap = np.zeros(3)
        self.crackle = np.zeros(3)

        self.semiMajorAxis = getattr(self, "semiMajorAxis", 0.0)
        self.period = 0.0

        self.eccentricityVector = np.zeros(3)
        self.eccentricity = getattr(self, "eccentricity", 0.0)

        self.orbitalAngularMomentum = np.zeros(3)
        self.magOrbitalAngularMomentum = 0.0

        self.inclination = getattr(self, "inclination", 0.0)
        self.trueAnomaly = 0.0
        self.meanAnomaly = getattr(self, "meanAnomaly", 0.0)
        self.eccentricAnomaly = 0.0
        self.argumentPeriapsis = getattr(self, "argumentPeriapsis", 0.0)
        self.longitudePeriapsis = 0.0
        self.longitudeAscendingNode = getattr(self, "longitudeAscendingNode", 0.0)

        self.timestep = 0.0
        self.hostBody = None
        self.hostMass = self.mass

    # Return a safe-to-edit clone of the bdoy
    def nBodyClone(self):
        return Body(self.name, self.mass, self.radius,
                    self.position.copy(), self.velocity.copy())


    # Oribital mechanics methods
    def calcOrbitalAngularMomentum(self):
        self.orbitalAngularMomentum = np.cross(self.position, self.velocity)
        self.magOrbitalAngularMomentum = np.linalg.norm(self.orbitalAngularMomentum)

    def calcEccentricity(self, G, totmass):
        gravparam = G * totmass
        r = np.linalg.norm(self.position)
        v = np.linalg.norm(self.velocity)
        vdotr = np.dot(self.velocity, self.position)

        if r == 0.0:
            self.eccentricityVector = np.zeros(3)
        else:
            self.eccentricityVector = (
                (v**2 * self.position - vdotr * self.velocity) / gravparam
                - self.position / r
            )

        self.eccentricity = np.linalg.norm(self.eccentricityVector)

    def calcOrbitFromVector(self, G, totmass):
        self.calcOrbitalAngularMomentum()
        self.calcEccentricity(G, totmass)

        self.semiMajorAxis = (self.magOrbitalAngularMomentum**2) / (
            G * totmass * (1 - self.eccentricity**2)
        )
        self.period = self.calcPeriod(G, totmass)

        if self.magOrbitalAngularMomentum > 0.0:
            self.inclination = math.acos(
                self.orbitalAngularMomentum[2] / self.magOrbitalAngularMomentum
            )
        else:
            self.inclination = 0.0

        if self.inclination == 0.0:
            self.longitudeAscendingNode = 0.0
            nplane = np.array([self.magOrbitalAngularMomentum, 0.0, 0.0])
            nscalar = np.linalg.norm(nplane)
        else:
            nplane = np.array([
                -self.orbitalAngularMomentum[1],
                self.orbitalAngularMomentum[0],
                0.0
            ])
            nscalar = np.linalg.norm(nplane)
            self.longitudeAscendingNode = math.acos(nplane[0] / nscalar)
            if nplane[1] < 0.0:
                self.longitudeAscendingNode = 2.0 * math.pi - self.longitudeAscendingNode

        r = np.linalg.norm(self.position)

        if self.eccentricity == 0.0 and self.inclination == 0.0:
            self.trueAnomaly = math.acos(self.position[0] / r)
            if self.velocity[0] < 0.0:
                self.trueAnomaly = 2.0 * math.pi - self.trueAnomaly
        elif self.eccentricity == 0.0:
            ndotR = np.dot(nplane, self.position) / (r * nscalar)
            ndotV = np.dot(nplane, self.velocity)
            self.trueAnomaly = math.acos(ndotR)
            if ndotV > 0.0:
                self.trueAnomaly = 2.0 * math.pi - self.trueAnomaly
        else:
            edotR = np.dot(self.eccentricityVector, self.position) / (r * self.eccentricity)
            rdotV = np.dot(self.velocity, self.position)
            self.trueAnomaly = math.acos(edotR)
            if rdotV < 0.0:
                self.trueAnomaly = 2.0 * math.pi - self.trueAnomaly

        if self.eccentricity != 0.0:
            edotn = np.dot(self.eccentricityVector, nplane) / (nscalar * self.eccentricity)
            self.argumentPeriapsis = math.acos(edotn)
            if self.eccentricityVector[2] < 0.0:
                self.argumentPeriapsis = 2.0 * math.pi - self.argumentPeriapsis
            self.longitudePeriapsis = self.argumentPeriapsis + self.longitudeAscendingNode
        else:
            self.argumentPeriapsis = 0.0
            self.longitudePeriapsis = 0.0
            
    def rotateZ(vec, angle):
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s, 0],
                        [s,  c, 0],
                        [0,  0, 1]])
            return R @ vec

    def rotateX(vec, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[1, 0,  0],
                    [0, c, -s],
                    [0, s,  c]])
        return R @ vec        
    
    # Calculate body position and velocity in 3-space
    # given orbital elements
    def calcVectorFromOrbit(self, G, totmass):

        # Calculate distance from CoM using semi-major axis, eccentricity, and true anomaly
        magpos = self.semiMajorAxis * (1.0 - self.eccentricity**2) / (1.0 + self.eccentricity * np.cos(self.trueAnomaly))

        # Position in orbital plane
        position = np.array([
            magpos * np.cos(self.trueAnomaly),
            magpos * np.sin(self.trueAnomaly),
            0.0
        ])

        # Velocity in orbital plane
        semiLatusRectum = abs(self.semiMajorAxis * (1.0 - self.eccentricity**2))
        gravparam = G * totmass

        if semiLatusRectum != 0.0:
            magvel = np.sqrt(gravparam / semiLatusRectum)
        else:
            magvel = 0.0

        velocity = np.array([
            -magvel * np.sin(self.trueAnomaly),
            magvel * (np.cos(self.trueAnomaly) + self.eccentricity),
            0.0
        ])
        

        # Rotate around z by -argument of periapsis
        if self.argumentPeriapsis != 0.0:
            position = self.rotateZ(position, -self.argumentPeriapsis)
            velocity = self.rotateZ(velocity, -self.argumentPeriapsis)

        # Rotate around x by -inclination
        if self.inclination != 0.0:
            position = self.rotateX(position, -self.inclination)
            velocity = self.rotateX(velocity, -self.inclination)

        # Rotate around z by -longitude of ascending node
        if self.longitudeAscendingNode != 0.0:
            position = self.rotateZ(position, -self.longitudeAscendingNode)
            velocity = self.rotateZ(velocity, -self.longitudeAscendingNode)

        # Save results
        self.position = position
        self.velocity = velocity

    def calcPeriod(self, G, totalMass):
        if self.semiMajorAxis == 0:
            return 0.0
        return math.sqrt(
            4.0 * math.pi**2 * self.semiMajorAxis**3 / (G * totalMass)
        )

    def calcEccentricAnomaly(self):
        if self.eccentricity == 0.0:
            self.eccentricAnomaly = self.meanAnomaly
            return

        Eold = self.eccentricAnomaly
        tolerance = 1e30
        ncalc = 0
        while abs(tolerance) > 1e-3 and ncalc < 100:
            fE = Eold - self.eccentricity * math.sin(Eold) - self.meanAnomaly
            fdashE = 1 - self.eccentricity * math.cos(Eold)
            Enext = Eold - fE / fdashE if fdashE != 0 else Eold * 1.05
            tolerance = Enext - Eold
            Eold = Enext
            ncalc += 1
        self.eccentricAnomaly = Eold

    def calcTrueAnomaly(self, G=None, totalMass=None, time=None):
        if time is None:
            self.calcEccentricAnomaly()
            self.trueAnomaly = 2.0 * math.atan2(
                math.sqrt(1.0 + self.eccentricity) * math.sin(self.eccentricAnomaly / 2.0),
                math.sqrt(1.0 - self.eccentricity) * math.cos(self.eccentricAnomaly / 2.0)
            )
        else:
            period = self.calcPeriod(G, totalMass)
            self.meanAnomaly = (2.0 * math.pi * time / period) % (2.0 * math.pi)
            self.calcEccentricAnomaly()
            self.trueAnomaly = 2.0 * math.atan2(
                math.sqrt(1.0 + self.eccentricity) * math.sin(self.eccentricAnomaly / 2.0),
                math.sqrt(1.0 - self.eccentricity) * math.cos(self.eccentricAnomaly / 2.0)
            )

    def changeFrame(self, framepos, framevel):
        self.position = self.position - framepos
        self.velocity = self.velocity - framevel

    # N-Body dynamics methods
    def calcTimestep(self, greekEta):
        tol = 1e-20
        normA = np.linalg.norm(self.acceleration)
        normJ = np.linalg.norm(self.jerk)
        normS = np.linalg.norm(self.snap)
        normC = np.linalg.norm(self.crackle)

        if normA * normS + normJ * normJ < tol:
            self.timestep = 0.0
        elif normC * normJ + normS * normS < tol:
            self.timestep = 1.0e30
        else:
            self.timestep = math.sqrt(
                greekEta * (normA * normS + normJ**2) / (normC * normJ + normS**2)
            )

    def calcAccelJerk(self, G, bodyarray, softening_length):
        for other in bodyarray:
            rel_position = self.position - other.position
            rel_velocity = self.velocity - other.velocity

            rmag = np.linalg.norm(rel_position)
            if rmag < 1.0e-2 * softening_length:
                continue

            r2 = rmag * rmag + softening_length * softening_length
            rmag = math.sqrt(r2)
            r3 = rmag**3

            factor = -G * other.mass / r3
            accelterm = factor * rel_position
            self.acceleration += accelterm

            alpha = np.dot(rel_velocity, rel_position) / r2
            jerkterm = factor * rel_velocity - 3 * alpha * accelterm
            self.jerk += jerkterm

    def calcSnapCrackle(self, G, bodyarray, softening_length):
        for other in bodyarray:
            rel_position = self.position - other.position
            rel_velocity = self.velocity - other.velocity
            rel_acceleration = self.acceleration - other.acceleration
            rel_jerk = self.jerk - other.jerk

            rmag = np.linalg.norm(rel_position)
            if rmag < 1.0e-2 * softening_length:
                continue

            r2 = rmag * rmag + softening_length * softening_length
            rmag = math.sqrt(r2)
            r3 = rmag**3

            factor = G * other.mass / r3
            accelterm = factor * rel_position

            alpha = np.dot(rel_velocity, rel_position) / r2
            jerkterm = factor * rel_velocity + 3 * alpha * accelterm

            v2 = np.dot(rel_velocity, rel_velocity)
            beta = (v2 + np.dot(rel_position, rel_acceleration)) / r2 + alpha**2

            snapterm = factor * rel_acceleration - 6 * alpha * jerkterm - 3 * beta * accelterm
            self.snap += snapterm

            gamma = (3.0 * np.dot(rel_velocity, rel_acceleration)
                     + np.dot(rel_position, rel_jerk)) / r2
            gamma += alpha * (3.0 * beta - 4.0 * alpha**2)

            crackleterm = (factor * rel_jerk
                           - 9.0 * alpha * snapterm
                           - 9.0 * beta * jerkterm
                           - 3.0 * gamma * accelterm)
            self.crackle += crackleterm

    
    # Safer version of acos
    def safeAcos(self, x):
        return np.acos(max(-1.0, min(1.0, x)))

    # Unimplemented methods declared in the C++ version
    def setLuminosity(self, lum): pass
    def getLuminosity(self): return -1.0
    def calcMainSequenceLuminosity(self): pass
    def calculatePeakWavelength(self): return -1.0
    def getTeff(self): return -1.0

    def setEquilibriumTemperature(self, temp): pass
    def getEquilibriumTemperature(self): return -1.0
    def setReflectiveLuminosity(self, lum): pass
    def calcLuminosity(self): pass
    def setAlbedo(self, alb): pass
    def getAlbedo(self): return -1.0

    def getNLongitude(self): return -1
    def getNLatitude(self): return -1
    def getNStars(self): return -1
    def getLongPick(self): return -1
    def getLatPick(self): return -1
    def getPSpin(self): return -1.0
    def getObliquity(self): return -1.0
    def getFluxMax(self): return -1.0
    def setNLongitude(self, nlong): pass
    def setNLatitude(self, nlat): pass
    def setNStars(self, s): pass
    def setPSpin(self, spin): pass
    def setObliquity(self, obliq): pass
    def initialiseArrays(self): pass
    def initialiseOutputVariables(self, prefixString, stars): pass
    def resetFluxTotals(self): pass
    def findSurfaceLocation(self, longitude, latitude): pass
    def calcLongitudeOfNoon(self, star, istar): pass
    
    def getMass(self):
        return self.mass
    
    def getAcceleration(self):
        return self.acceleration
    
    def setAcceleration(self, acc):
        self.acceleration = acc
        
    def getJerk(self):
        return self.jerk
        
    def setJerk(self, je):
        self.jerk = je
        
    def setSnap(self, sn):
        self.snap = sn
        
    def setCrackle(self, cr):
        self.crackle = cr
        
    def setPosition(self, pos):
        self.position = pos
        
    def getPosition(self):
        return self.position
    
    def setVelocity(self, vel):
        self.velocity = vel
    
    def getVelocity(self):
        return self.velocity
    
    def getType(self):
        return self.type
    
    def getName(self):
        return self.name
    
    def getRadius(self):
        return self.radius
    
    def getSemiMajorAxis(self):
        return self.semiMajorAxis
    
    def getEccentricity(self):
        return self.eccentricity
    
    def getInclination(self):
        return self.inclination
    
    def getLongitudeAscendingNode(self):
        return self.longitudeAscendingNode
    
    def getArgumentPeriapsis(self):
        return self.argumentPeriapsis
    
    def getMeanAnomaly(self):
        return self.meanAnomaly
    