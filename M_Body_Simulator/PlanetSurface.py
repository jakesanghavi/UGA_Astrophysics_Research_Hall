import numpy as np
from M_Body_Simulator.Body import Body

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
        
        # Pick default surface location
        self.iLongPick = nLongitude // 2
        self.iLatPick = nLatitude // 2

    def reset_flux_totals(self):
        self.fluxtot.fill(0)
        self.flux.fill(0)

    def find_surface_location(self, longitude=np.pi, latitude=np.pi/2):
        self.iLongPick = (np.abs(self.longitude - longitude)).argmin()
        self.iLatPick = (np.abs(self.latitude - latitude)).argmin()