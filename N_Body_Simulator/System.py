import numpy as np
import sys
from astropy.constants import G

# N-Body System on which our simulation will run
class System:
    def __init__(self, name="System", bodyarray=None):
        # Included bodies information
        self.name = name
        self.bodies = bodyarray if bodyarray is not None else []
        self.bodyCount = len(self.bodies)
        
        # Relevant mass-energy info is calculated farther down
        self.totalMass = sum(body.mass for body in self.bodies)
        self.initialEnergy = 0.0
        self.totalEnergy = 0.0
        
        # Initial params from the C++ version
        self.timeStep = 0.0
        self.timeControl = 0.00002
        # self.G = 1.0
        self.G = G.value
        self.softeningLength = 1.0e-5
        
        # More energy + momentum factors to be computed below
        self.initialAngularMomentum = np.zeros(3)
        self.totalAngularMomentum = np.zeros(3)
        self.deltaAngularMomentum = 0.0
        self.deltaEnergy = 0.0
        
        # Position and velocity added below
        self.positionCOM = np.zeros(3)
        self.velocityCOM = np.zeros(3)
        self.accelerationCOM = np.zeros(3)
        
        self.planetaryIlluminationOn = False

    # Add a body to the system
    def addBody(self, newBody):
        self.bodies.append(newBody)
        self.bodyCount = len(self.bodies)
        self.totalMass += newBody.mass

    # Remove a body from the system, forcing recomputations
    def removeBody(self, index):
        del self.bodies[index]
        self.bodyCount = len(self.bodies)
        self.calcInitialProperties()

    # Set the "host" bodies (the parent star/s)
    def setHostBodies(self, orbitCenter):
        for i in range(self.bodyCount):
            if orbitCenter[i] > 0:
                host_index = orbitCenter[i] - 1
                self.bodies[i].hostBody = self.bodies[host_index]
                self.bodies[host_index].hostMass += self.bodies[i].mass

    # Find the center of mass and define it as the origin of the system
    def calcCOMFrame(self, participants=None):
        if participants is None:
            participants = np.ones(self.bodyCount, dtype=int)
        
        positionCOM = np.zeros(3)
        velocityCOM = np.zeros(3)
        participantMass = 0.0

        for i, p in enumerate(participants):
            if p == 1:
                m = self.bodies[i].mass
                participantMass += m
                positionCOM += self.bodies[i].position * m
                velocityCOM += self.bodies[i].velocity * m

        if participantMass > 0:
            positionCOM /= participantMass
            velocityCOM /= participantMass

        self.positionCOM = positionCOM
        self.velocityCOM = velocityCOM

    # Transform coordinates (position and velocity) to COMFrame
    def transformToCOMFrame(self, participants=None):
        self.calcCOMFrame(participants)
        if participants is None:
            participants = np.ones(self.bodyCount, dtype=int)
        for i, p in enumerate(participants):
            if p == 1:
                self.bodies[i].position -= self.positionCOM
                self.bodies[i].velocity -= self.velocityCOM

    # Safe Acosine method
    @staticmethod
    def safeAcos(x):
        return np.arccos(np.clip(x, -1.0, 1.0))
    
    # Find total angular momentum of the system and store in fields
    def calcTotalAngularMomentum(self):
        L = np.zeros(3)
        for body in self.bodies:
            L += np.cross(body.position, body.mass * body.velocity)
        self.totalAngularMomentum = L
        self.deltaAngularMomentum = np.linalg.norm(L - self.initialAngularMomentum)

    # Find the total energy of the system and store in fields
    def calcEnergy(self):
        kinetic = 0.0
        potential = 0.0
        for i, bi in enumerate(self.bodies):
            m_i = bi.mass
            v_i = np.linalg.norm(bi.velocity)
            kinetic += 0.5 * m_i * v_i**2
            for j in range(i+1, self.bodyCount):
                bj = self.bodies[j]
                r = np.linalg.norm(bi.position - bj.position)
                potential -= self.G * m_i * bj.mass / np.sqrt(r**2 + self.softeningLength**2)
        self.totalEnergy = kinetic + potential
        self.deltaEnergy = self.totalEnergy - self.initialEnergy

    # Find the size of the time step for integrating
    # NEVER USED
    def calcTimeStep(self):
        dt = np.inf
        for _, bi in enumerate(self.bodies):
            r_acc = np.linalg.norm(bi.acceleration)
            if r_acc > 0:
                dt = min(dt, np.sqrt(self.timeControl * np.linalg.norm(bi.velocity) / r_acc))
        self.timeStep = dt
        
    # Calculate timestep given bodies array
    def calcNBodyTimestep(self, bodies=None, dtmax=np.inf):
        if bodies is None:
            bodies = self.bodies

        dt = dtmax
        for bi in bodies:
            r_acc = np.linalg.norm(bi.acceleration)
            if r_acc > 0:
                dt = min(dt, np.sqrt(self.timeControl * np.linalg.norm(bi.velocity) / r_acc))
        self.timeStep = dt
        return dt

    # Set orbit of one planet/child body based on the host body
    # Will have to be extended to work for multiple host bodies
    ## NEED TO ADD a SET_HOST_BODIES method to Body.py
    ## ANd call those here
    def setOrbit(self, index, host_index, a, e, inclination=0.0, Omega=0.0, omega=0.0, f=0.0):
        body = self.bodies[index]
        host = self.bodies[host_index]

        r = a * (1 - e**2) / (1 + e * np.cos(f))
        mu = self.G * (body.mass + host.mass)

        x_orb = r * np.cos(f)
        y_orb = r * np.sin(f)

        h = np.sqrt(mu * a * (1 - e**2))
        vx_orb = -mu / h * np.sin(f)
        vy_orb = mu / h * (e + np.cos(f))

        cos_O = np.cos(Omega)
        sin_O = np.sin(Omega)
        cos_i = np.cos(inclination)
        sin_i = np.sin(inclination)
        cos_w = np.cos(omega)
        sin_w = np.sin(omega)

        R = np.array([
            [cos_O * cos_w - sin_O * sin_w * cos_i, -cos_O * sin_w - sin_O * cos_w * cos_i, sin_O * sin_i],
            [sin_O * cos_w + cos_O * sin_w * cos_i, -sin_O * sin_w + cos_O * cos_w * cos_i, -cos_O * sin_i],
            [sin_w * sin_i, cos_w * sin_i, cos_i]
        ])

        pos_vec = R @ np.array([x_orb, y_orb, 0.0])
        vel_vec = R @ np.array([vx_orb, vy_orb, 0.0])

        body.position = host.position + pos_vec
        body.velocity = host.velocity + vel_vec
        body.hostBody = host
        host.hostMass += body.mass

    # Additional utility methods
    def updateAccelerations(self):
        for body in self.bodies:
            body.acceleration[:] = 0.0
        for i, bi in enumerate(self.bodies):
            for j in range(i+1, self.bodyCount):
                bj = self.bodies[j]
                diff = bj.position - bi.position
                r = np.linalg.norm(diff)
                if r == 0:
                    continue
                a_i = self.G * bj.mass / (r**3 + self.softeningLength**3) * diff
                bi.acceleration += a_i
                bj.acceleration -= self.G * bi.mass / (r**3 + self.softeningLength**3) * diff

    # Advance the system by a time step
    def evolve(self, dt):
        self.updateAccelerations()
        for body in self.bodies:
            body.velocity += body.acceleration * dt
            body.position += body.velocity * dt

    def printBodies(self):
        for i, body in enumerate(self.bodies):
            print(f"Body {i}: Pos {body.position}, Vel {body.velocity}, Mass {body.mass}")

    def resetVelocities(self):
        for body in self.bodies:
            body.velocity[:] = 0.0

    def applyVelocityKick(self, factor):
        for body in self.bodies:
            body.velocity += body.acceleration * factor

    def applyPositionDrift(self, factor):
        for body in self.bodies:
            body.position += body.velocity * factor

    def scaleMasses(self, factor):
        for body in self.bodies:
            body.mass *= factor

    def translateSystem(self, vector):
        for body in self.bodies:
            body.position += vector
            
    def calcForces(self, bodyarray):
        zeroVector = np.zeros(3)
        for body in bodyarray:
            body.setAcceleration(zeroVector)
            body.setJerk(zeroVector)
            body.calcAccelJerk(self.G, bodyarray, self.softeningLength)

        for body in bodyarray:
            body.setSnap(zeroVector)
            body.setCrackle(zeroVector)
            body.calcSnapCrackle(self.G, bodyarray, self.softeningLength)

    # Calculate/recalculate inital system parameters
    def calcInitialProperties(self):
        self.bodyCount = len(self.bodies)
        self.totalMass = sum(body.getMass() for body in self.bodies)
        self.transformToCOMFrame()
        self.calcForces(self.bodies)
        self.initialEnergy = self.totalEnergy
        self.deltaEnergy = 0.0
        self.calcTotalAngularMomentum()
        self.initialAngularMomentum = self.totalAngularMomentum
        self.deltaAngularMomentum = 0.0

    # Deal with eclipse cases
    # May be much more common for high # bodies
    def checkForEclipses(self, bodyIndex):
        eclipsefrac = [0.0] * self.bodyCount
        for i in range(self.bodyCount):
            if i == bodyIndex:
                continue

            vector_i = self.getBody(bodyIndex).getPosition() - self.getBody(i).getPosition()
            mag_i = np.linalg.norm(vector_i)
            rad_i = self.getBody(i).getRadius()

            for j in range(self.bodyCount):
                if j == i or j == bodyIndex:
                    continue
                vector_j = self.getBody(bodyIndex).getPosition() - self.getBody(j).getPosition()
                mag_j = np.linalg.norm(vector_j)
                rad_j = self.getBody(j).getRadius()

                if mag_i > 0 and mag_j > 0 and mag_i > mag_j:
                    idotj = np.dot(vector_i, vector_j) / (mag_i * mag_j)
                    b = mag_j * np.sqrt(1.0 - idotj ** 2)

                    if b < (rad_i + rad_j) and idotj < 0:
                        rad_i2 = rad_i ** 2
                        rad_j2 = rad_j ** 2
                        angle1 = 2.0 * self.safeAcos((rad_i2 + b ** 2 - rad_j2) / (2.0 * b * rad_i))
                        angle2 = 2.0 * self.safeAcos((rad_j2 + b ** 2 - rad_i2) / (2.0 * b * rad_j))
                        area_i = 0.5 * rad_i2 * (angle1 - np.sin(angle1))
                        area_j = 0.5 * rad_j2 * (angle2 - np.sin(angle2))
                        eclipsefrac[i] = min(max((area_i + area_j) / (np.pi * rad_i2), 0.0), 1.0)

        return eclipsefrac

    # Evolve the system from t0 to t
    def evolveSystem(self, tbegin, tend=None):
        if tend is None:
            tend = tbegin
            tbegin = 0.0

        time = tbegin
        dtmax = (tend - tbegin) / 2.0
        self.calcInitialProperties()
        self.calcForces(self.bodies)
        self.calcNBodyTimestep(self.bodies, dtmax)
        self.calcTotalEnergy()
        self.calcTotalAngularMomentum()

        predicted = [body.nBodyClone() for body in self.bodies]

        while time < tend:
            # Predict positions and velocities
            for i, body in enumerate(self.bodies):
                pos, vel, acc, jerk = body.getPosition(), body.getVelocity(), body.getAcceleration(), body.getJerk()
                pos_p = pos + vel * self.timeStep + 0.5 * acc * self.timeStep**2 + (1/6) * jerk * self.timeStep**3
                vel_p = vel + acc * self.timeStep + 0.5 * jerk * self.timeStep**2
                predicted[i].setPosition(pos_p)
                predicted[i].setVelocity(vel_p)

            # Calculate predicted forces
            self.calcForces(predicted)

            # Correct positions and velocities
            for i, body in enumerate(self.bodies):
                pos_p, vel_p = predicted[i].getPosition(), predicted[i].getVelocity()
                acc_p, jerk_p = predicted[i].getAcceleration(), predicted[i].getJerk()
                pos, vel = body.getPosition(), body.getVelocity()
                acc, jerk = body.getAcceleration(), body.getJerk()

                accterm = 0.5 * self.timeStep * (acc_p + acc)
                jerkterm = (self.timeStep**2 / 12) * (jerk_p - jerk)
                vel_c = vel + accterm + jerkterm

                accterm = (self.timeStep**2 / 12) * (acc_p - acc)
                velterm = 0.5 * self.timeStep * (vel_c + vel)
                pos_c = pos + velterm + accterm

                body.setPosition(pos_c)
                body.setVelocity(vel_c)

            # Update forces
            self.calcForces(self.bodies)
            time += self.timeStep
            self.calcNBodyTimestep(self.bodies, dtmax)
            self.calcTotalEnergy()
            self.calcTotalAngularMomentum()

        # Cleanup predicted bodies
        del predicted

    # Calculate eq temperatures for planets
    def calcPlanetaryEquilibriumTemperatures(self):
        if not self.planetaryIlluminationOn:
            return [0.0] * self.bodyCount

        temperatures = [0.0] * self.bodyCount
        for i, body in enumerate(self.bodies):
            # Fraction of stellar flux blocked by eclipses
            eclipsefrac = self.checkForEclipses(i)
            # Incoming flux from all other bodies
            flux = 0.0
            for j, other in enumerate(self.bodies):
                if i == j:
                    continue
                distance = np.linalg.norm(body.getPosition() - other.getPosition())
                if distance > 0:
                    flux += (1 - eclipsefrac[j]) * other.getLuminosity() / (4.0 * np.pi * distance**2)
            # Equilibrium temperature (Stefan-Boltzmann law)
            temperatures[i] = (flux / self.sigma_SB) ** 0.25
        return temperatures

    # 2d Flux map creation for planetsurface bodies
    def calc2DFlux(self, time, dt):

        # Loop through each body
        for j, body in enumerate(self.bodies):
            if body.getType() == "PlanetSurface":
                body.resetFluxTotals()

                eclipsefrac = self.checkForEclipses(j)

                for i, source in enumerate(self.bodies):
                    if i == j:
                        continue

                    if source.getType() == "Star":
                        body.calcFlux(i, source, eclipsefrac[i], time, dt)

                    elif source.getType() == "Planet" and self.planetaryIlluminationOn:
                        body.calcFlux(i, source, eclipsefrac[i], time, dt)

                body.calcIntegratedQuantities(dt)

    # Write final system state to a file
    def outputSystemState(self, filename="system_output.txt"):
        with open(filename, "w") as f:
            f.write("# BodyID x y z vx vy vz mass radius temp\n")
            temps = self.calcPlanetaryEquilibriumTemperatures()
            for i, body in enumerate(self.bodies):
                pos = body.getPosition()
                vel = body.getVelocity()
                mass = body.getMass()
                radius = body.getRadius()
                temp = temps[i]
                f.write(f"{i} {pos[0]:.6e} {pos[1]:.6e} {pos[2]:.6e} "
                        f"{vel[0]:.6e} {vel[1]:.6e} {vel[2]:.6e} "
                        f"{mass:.6e} {radius:.6e} {temp:.2f}\n")

    # Write output energy and angular momentum to a file
    def outputEnergyAndAngularMomentum(self, filename="energy_angular.txt"):
        with open(filename, "w") as f:
            f.write("# Time TotalEnergy DeltaEnergy TotalAngularMomentum DeltaAngularMomentum\n")
            f.write(f"{self.nTime*self.timeStep:.6e} {self.totalEnergy:.6e} {self.deltaEnergy:.6e} "
                    f"{self.totalAngularMomentum:.6e} {self.deltaAngularMomentum:.6e}\n")
            
    # Not 100% sure what this is doing
    def initialise2DFluxOutput(self, prefixString):
        for body in self.bodies:
            if getattr(body, "type", None) == "PlanetSurface":
                # Ensure the PlanetSurface body has this method
                if hasattr(body, "initialiseOutputVariables"):
                    body.initialiseOutputVariables(prefixString, self.bodies)
        
        # To be implemented
        if hasattr(self, "calcLongitudesOfNoon"):
            self.calcLongitudesOfNoon()
        
    # Calculate the total system energy
    def calcTotalEnergy(self):
        gravitational_potential = 0.0
        kinetic_energy = 0.0
        
        for i, body_ref in enumerate(self.bodies):
            body_ref_pos = body_ref.getPosition()
            body_ref_mass = body_ref.getMass()
            body_ref_vel = body_ref.getVelocity()

            # Gravitational potential
            for j, body_question in enumerate(self.bodies):
                if i != j:
                    body_question_pos = body_question.getPosition()
                    body_question_mass = body_question.getMass()

                    r_vector = body_question_pos - body_ref_pos
                    r_distance = np.linalg.norm(r_vector)

                    # Small amount to deal with gravitational potential of overlapping bodies
                    # Tweak as needed
                    epsilon = 1e-9
                    gravitational_potential += -self.G * body_ref_mass * body_question_mass / np.sqrt(r_distance**2 + epsilon**2)


            # Kinetic energy
            velocity = np.linalg.norm(body_ref_vel)
            kinetic_energy += 0.5 * body_ref_mass * velocity ** 2

        self.totalEnergy = kinetic_energy + gravitational_potential
        
        # Again safety check for div0
        if self.initialEnergy == 0:
            self.deltaEnergy = 0.0
        else:
            self.deltaEnergy = (self.totalEnergy - self.initialEnergy) / self.initialEnergy
        
    # Set up all orbits given the orbit list of orbit centers
    def setupOrbits(self, orbitCenters):
        """
        Sets up orbits for all bodies using NumPy arrays.
        orbitCentre: list of ints (0=CoM, >0=host body index+1, <0=(0,0,0))
        """

        participants = np.zeros(self.bodyCount, dtype=int)
        hosts = np.zeros(self.bodyCount, dtype=int)

        # Firstly, set up objects around centre of mass
        for b in range(self.bodyCount):
            if orbitCenters[b] == 0:
                participants[b] = 1
                self.bodies[b].calcVectorFromOrbit(self.G, self.totalMass)
            elif orbitCenters[b] < 0:
                self.bodies[b].calcVectorFromOrbit(self.G, self.totalMass - self.bodies[b].mass)

        # Transform them to COM Frame
        self.transformToCOMFrame(participants)

        # Arrays for host system totals
        totalMassAroundHost = np.zeros(self.bodyCount)
        xCOMroundHost = np.zeros((self.bodyCount, 3))
        vCOMroundHost = np.zeros((self.bodyCount, 3))

        # i) Place satellites around host at centre
        for b in range(self.bodyCount):
            if orbitCenters[b] > 0:
                ihost = orbitCenters[b] - 1
                hosts[ihost] = 1
                totalMassAroundHost[ihost] += self.bodies[b].mass

        # Add host mass to totals
        for b in range(self.bodyCount):
            if hosts[b] == 1:
                totalMassAroundHost[b] += self.bodies[b].mass

        # Setup bodies in orbit around each host
        for b in range(self.bodyCount):
            if orbitCenters[b] > 0:
                ihost = orbitCenters[b] - 1
                self.bodies[b].calcVectorFromOrbit(self.G, totalMassAroundHost[ihost])

                # Transform to host frame
                framepos = -self.bodies[ihost].position
                framevel = -self.bodies[ihost].velocity
                self.bodies[b].changeFrame(framepos, framevel)

                # Compute COM of host system (minus host contribution)
                xCOMroundHost[ihost] += self.bodies[b].mass * self.bodies[b].position
                vCOMroundHost[ihost] += self.bodies[b].mass * self.bodies[b].velocity

        # Adjust hosts so the CoM of the system follows host's orbit
        for b in range(self.bodyCount):
            if hosts[b] == 1:
                newPosition = (self.bodies[b].position * totalMassAroundHost[b] - xCOMroundHost[b]) / self.bodies[b].mass
                newVelocity = (self.bodies[b].velocity * totalMassAroundHost[b] - vCOMroundHost[b]) / self.bodies[b].mass

                self.bodies[b].position = newPosition
                self.bodies[b].velocity = newVelocity
 
        
    # Write nbody information
    def outputNBodyData(self, output_file, time, orbit_centers, input_params):
        # Transform to the Centre of Mass Frame
        self.transformToCOMFrame()
        
        # Enum over the bodies
        for j, body in enumerate(self.bodies):
            # if orbit_centers[j] > 0:
            #     # Use another body as the orbit center
            #     # Don't think this line is right
            #     ref_body = self.bodies[orbit_centers[j] - 1]
            #     body.calcOrbitFromVector(self.G, ref_body)
            # else:
            #     body.calcOrbitFromVector(self.G, self.totalMass)
            
            if orbit_centers[j] > 0:
                # Orbiting another body (star, planet, etc.)
                ref_body = self.bodies[orbit_centers[j] - 1]
                body.calcOrbitFromVector(self.G, ref_body.getMass(), ref_body.getPosition(), ref_body.getVelocity())
            else:
                # No host → this is the central reference (e.g., primary star or barycenter)
                body.eccentricity = 0.0
                body.eccentricityVector = np.zeros(3)

            position = body.getPosition()
            velocity = body.getVelocity()
            
            body_name = input_params.getStringVariable('BodyName', j)

            # Output format CSV
            line = (
                f"{time:+.4E},"
                f"{self.totalEnergy:+.4E},"
                f"{body_name},"
                f"{body.getMass():+.4E},"
                f"{body.getRadius():+.4E},"
                f"{position[0]:+.4E},{position[1]:+.4E},{position[2]:+.4E},"
                f"{velocity[0]:+.4E},{velocity[1]:+.4E},{velocity[2]:+.4E},"
                f"{body.getSemiMajorAxis():+.4E},"
                f"{body.getEccentricity():+.4E},"
                f"{body.getInclination():+.4E},"
                f"{body.getLongitudeAscendingNode():+.4E},"
                f"{body.getArgumentPeriapsis():+.4E},"
                f"{body.getMeanAnomaly():+.4E}\n"
            )

            output_file.write(line)

        output_file.flush()  
        
    # Write planetsurface fluxes to files
    def output2DFluxData(self, snapshotNumber, tSnap, prefixString, input_params):
        for i, body in enumerate(self.bodies):
            if body.getType() == "PlanetSurface":
                
                if self.fullOutput:  
                    body.writeFluxFile(snapshotNumber, self.nTime, tSnap, prefixString)
                
                body.writeToLocationFiles(tSnap, self.bodies, body_name=input_params.getStringVariable("BodyName", i))
    
    # Write integrated flux info to file
    def outputIntegratedFluxData(self):
        for body in self.bodies:
            if body.getType() == "PlanetSurface":
                body.writeIntegratedFile()
           
    # Create the output .info file     
    def outputInfoFile(self, n_snaps, input_params):
        file_name = f"{self.getName()}.info"
        globalFluxMax = 0.0
        nStars = self.countStars()

        with open(file_name, "w") as infoFile:
            # Write number of snapshots
            infoFile.write(f"# Number of snapshots\n{n_snaps}\n")
            # Write number of stars
            infoFile.write(f"# Number of stars\n{nStars}\n")

            for i, body in enumerate(self.bodies):
                if body.getType() == "Star":
                    name_of_body = input_params.getStringVariable("BodyName", i)
                    infoFile.write(f"# Star name\n{name_of_body}\n")
                    infoFile.write(
                        f"# Radius (m)  Teff (K)  Peak wavelength (m)\n"
                        f"{body.getRadius():+.4E} {body.getTeff():+.4E} {body.calculatePeakWavelength():+.4E}\n"
                    )
                elif body.getType() == "PlanetSurface":
                    if body.getFluxMax() > globalFluxMax:
                        globalFluxMax = body.getFluxMax()

            # Write maximum flux
            infoFile.write(f"# Maximum flux across all PlanetSurface objects (W/m^2)\n{globalFluxMax:+.4E}\n")


            
    # Count the number of stars in the system
    def countStars(self):
        return sum(1 for body in self.bodies if body.getType() == "Star")

    
    # Additional getter setters
    def setName(self, namestring):
        self.name = namestring

    def setBodyCount(self, count):
        self.bodyCount = count

    def setTotalMass(self, mtot):
        self.totalMass = mtot

    def setTimestep(self, dt):
        self.timeStep = dt

    def setNTime(self, t):
        self.nTime = t

    def setIllumination(self, flag):
        self.planetaryIlluminationOn = bool(flag)
        
    def setFluxOutput(self, full):
        self.fullOutput = bool(full)
        
    def getTimestep(self):
        return self.timeStep
    
    def getName(self):
        return self.name
    
    def getBody(self, index):
        return self.bodies[index]