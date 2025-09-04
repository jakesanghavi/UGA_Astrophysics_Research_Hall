import numpy as np
from N_Body_Simulator.Body import Body
from constants import twopi, pi, fluxsol

# Class for a planetary surface body type
class PlanetSurface(Body):
    # Consants as defined in the C++ version
    nStarMax = 10
    nLatMax = 500
    nLongMax = 500
    
    def __init__(self, name="Planet", mass=1.0, radius=1.0, pos=None, vel=None,
                 nStars=1, nLatitude=100, nLongitude=100, Pspin=1.0, obliquity=0.5123):
        super().__init__(name, mass, radius, pos, vel)
        
        self.type = "PlanetSurface"
        self.nStars = nStars
        self.nLatitude = nLatitude
        self.nLongitude = nLongitude
        self.Pspin = Pspin
        self.obliquity = obliquity
        self.fluxmax = 0.0
        
        self.noon = np.zeros(self.nStarMax)
        self.longitude = np.linspace(0, 2*np.pi, nLongitude, endpoint=False)
        self.latitude = np.linspace(0, np.pi, nLatitude, endpoint=False)
        
        self.flux = np.zeros((self.nStarMax, nLongitude, nLatitude))
        self.altitude = np.zeros_like(self.flux)
        self.azimuth = np.zeros_like(self.flux)
        self.hourAngle = np.zeros((self.nStarMax, nLongitude))
        
        self.fluxtot = np.zeros((nLongitude, nLatitude))
        self.integratedflux = np.zeros((nLongitude, nLatitude))
        self.darkness = np.zeros((nLongitude, nLatitude))
        self.fluxsol = fluxsol
        
        # Pick default surface location
        self.iLongPick = nLongitude // 2
        self.iLatPick = nLatitude // 2

    def reset_flux_totals(self):
        self.fluxtot.fill(0)
        self.flux.fill(0)

    def find_surface_location(self, longitude=np.pi, latitude=np.pi/2):
        self.iLongPick = (np.abs(self.longitude - longitude)).argmin()
        self.iLatPick = (np.abs(self.latitude - latitude)).argmin()
        
    import numpy as np

    # Calculate flux from star over planet's surface
    def calcFlux(self, istar, star, eclipseFraction, time, dt):

        # Planet and star positions
        planetpos = self.getPosition()
        starpos = star.getPosition()
        pos = starpos - planetpos
        magpos = np.linalg.norm(pos)
        unitpos = pos / magpos
        lstar = star.getLuminosity()

        # Declination vector
        decVector = np.copy(unitpos)
        if self.obliquity != 0.0:
            # Rotation around x-axis
            c, s = np.cos(self.obliquity), np.sin(self.obliquity)
            rotX = np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
            decVector = rotX @ decVector

        rdotn = np.dot(unitpos, decVector)
        declination = np.arccos(np.clip(rdotn, -1.0, 1.0))

        zvector = np.array([0.0, 0.0, 1.0])

        # Loop over longitude
        for j in range(self.nLongitude):
            long_apparent = (self.longitude[j] - self.noon[istar] + twopi * time / self.Pspin) % twopi
            longSurface = np.array([np.cos(long_apparent), np.sin(long_apparent), 0.0])

            rdotn = np.dot(unitpos, longSurface)
            self.hourAngle[istar][j] = np.arccos(np.clip(rdotn, -1.0, 1.0))

            if np.dot(np.cross(unitpos, longSurface), zvector) > 0.0:
                self.hourAngle[istar][j] *= -1.0

            # Loop over latitude
            for k in range(self.nLatitude):
                surface = np.array([
                    np.sin(self.latitude[k]) * np.cos(long_apparent),
                    np.sin(self.latitude[k]) * np.sin(long_apparent),
                    np.cos(self.latitude[k])
                ])
                surface /= np.linalg.norm(surface)

                # Rotate around X axis
                if self.obliquity != 0.0:
                    surface = rotX @ surface

                rdotn = np.dot(unitpos, surface)
                fluxtemp = lstar * rdotn / (4.0 * pi * magpos**2) if rdotn > 0.0 else 0.0

                self.flux[istar][j][k] = fluxtemp * (1.0 - eclipseFraction) * self.fluxsol
                self.fluxtot[j][k] += self.flux[istar][j][k]

                if self.fluxtot[j][k] > self.fluxmax:
                    self.fluxmax = self.fluxtot[j][k]

                # Altitude calculation
                alt = -np.cos(declination) * np.cos(self.hourAngle[istar][j]) * np.sin(self.latitude[k]) \
                    + np.sin(declination) * np.cos(self.latitude[k])
                self.altitude[istar][j][k] = np.arcsin(np.clip(alt, -1.0, 1.0))

                # Azimuth calculation
                denom = np.cos(self.altitude[istar][j][k]) * np.sin(self.latitude[k])
                if denom != 0.0:
                    az = (np.sin(self.altitude[istar][j][k]) * np.sin(self.latitude[k]) - np.sin(declination)) / denom
                    self.azimuth[istar][j][k] = np.arccos(np.clip(az, -1.0, 1.0))
                else:
                    self.azimuth[istar][j][k] = 0.0

                # Adjust azimuth for afternoon
                if self.hourAngle[istar][j] > 0.0:
                    self.azimuth[istar][j][k] = twopi - self.azimuth[istar][j][k]
                 
    # Update darkness array based on total flux at a certain time   
    def calcIntegratedQuantities(self, dt):
        # Update integrated flux
        self.integratedflux += self.fluxtot * dt

        # Update darkness where flux is effectively zero
        mask = self.fluxtot < 1.0e-6
        self.darkness[mask] += dt

    # Write location data for a certain timestep
    def writeToLocationFiles(self, time, bodies, body_name):
        for istar in range(self.nStars):
            body = bodies[istar]
            if body.getType() == "Star":
                # Generate filename from star name
                filename = f"Star_{body_name}_locations.txt"

                # Star position relative to planet
                starpos = body.getPosition() - self.getPosition()  # NumPy array

                # Extract picked longitude and latitude
                lon = self.longitude[self.iLongPick]
                lat = self.latitude[self.iLatPick]

                # Flux, altitude, azimuth, hour angle
                flux_val = self.flux[istar][self.iLongPick, self.iLatPick]
                alt_val = self.altitude[istar][self.iLongPick, self.iLatPick]
                az_val = self.azimuth[istar][self.iLongPick, self.iLatPick]
                hour_angle_val = self.hourAngle[istar][self.iLongPick]

                # Open the file in append mode and write the data
                with open(filename, 'a') as f:
                    f.write(
                        f"{time:+.4E} {starpos[0]:+.4E} {starpos[1]:+.4E} {starpos[2]:+.4E} "
                        f"{lon:+.4E} {lat:+.4E} {flux_val:+.4E} {alt_val:+.4E} "
                        f"{az_val:+.4E} {hour_angle_val:+.4E}\n"
                    )

    # Write to a file
    def writeIntegratedFile(self):
        filename = "integrated_flux.txt"

        with open(filename, 'w') as f:
            # Write the number of latitude and longitude points
            f.write(f"{self.nLatitude} {self.nLongitude}\n")

            # Loop over longitude and latitude
            for j in range(self.nLongitude):
                for k in range(self.nLatitude):
                    f.write(
                        f"{self.longitude[j]:+.4E} {self.latitude[k]:+.4E} "
                        f"{self.integratedflux[j, k]:+.4E} {self.darkness[j, k]:+.4E}\n"
                    )