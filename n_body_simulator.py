import time
from N_Body_Simulator.ParFile import ParFile
from N_Body_Simulator.System import System
from N_Body_Simulator.Star import Star
from N_Body_Simulator.Planet import Planet
from N_Body_Simulator.PlanetSurface import PlanetSurface
from constants import Gsi as G, twopi
from utils import safe_get
import sys

# Run the simulation
def main(argv):
    snapshot_number = 0

    # Record start time
    start_time = time.time()

    # Read in parameters file
    input_file = argv[1] if len(argv) == 2 else None
    input_params = ParFile()
    if input_file:
        print(f"\tReading file {input_file}")
        input_params.read_file(input_file)
    else:
        raise FileNotFoundError("Input file required to initialize ParFile.")
    
    # Check and display parameters to the console
    input_params.checkParameters()
    input_params.displayParameters()

    # Retrieve parameters
    t_max = input_params.getDoubleVariable("MaximumTime")
    t_snap = input_params.getDoubleVariable("SnapshotTime")
    restart = input_params.getBoolVariable("Restart")
    full_output = input_params.getBoolVariable("FullOutput")
    system_name = input_params.getStringVariable("SystemName")

    n_time = int(t_max / t_snap) + 1
    
    # Parameter from the C++ version - not sure yet how to use this
    if restart:
        print("Restart - Using vector data from nbody output")

    # Create bodies
    body_array = []
    n_bodies = input_params.getIntVariable("Number_Bodies")
    print("Creating bodies")
    for i in range(n_bodies):
        # Get relevant parameters
        body_name = safe_get(input_params.getStringVariable, "BodyName", i)
        body_type = safe_get(input_params.getStringVariable, "BodyType", i)
        mass = safe_get(input_params.getDoubleVariable, "Mass", i)
        radius = safe_get(input_params.getDoubleVariable, "Radius", i)
        position = safe_get(input_params.getDoubleVariable, "Position", i)
        velocity = safe_get(input_params.getDoubleVariable, "Velocity", i)
        semimaj = safe_get(input_params.getDoubleVariable, "SemiMajorAxis", i)
        ecc = safe_get(input_params.getDoubleVariable, "Eccentricity", i)
        inc = safe_get(input_params.getDoubleVariable, "Inclination", i)
        longascend = safe_get(input_params.getDoubleVariable, "LongAscend", i)
        argper = safe_get(input_params.getDoubleVariable, "Periapsis", i)
        meananom = safe_get(input_params.getDoubleVariable, "MeanAnomaly", i)
        
        # No need for safe_get, this is pre-defined
        totalMass = input_params.getDoubleVariable("TotalMass")
        
        print(body_type)

        
        if body_type == "Star":
            body_array.append(Star(name=body_name, mass=mass, radius=radius, 
                                   position=position, velocity=velocity, semimaj=semimaj, 
                                   ecc=ecc, inc=inc, longascend=longascend, argper=argper, 
                                   meananom=meananom, G=G, totalMass=totalMass))
        elif body_type == "Planet":
            body_array.append(Planet(name=body_name, mass=mass, radius=radius, 
                                   position=position, velocity=velocity, semimaj=semimaj, 
                                   ecc=ecc, inc=inc, longascend=longascend, argper=argper, 
                                   meananom=meananom, G=G, totalMass=totalMass))
        elif body_type == "PlanetSurface":
            body_array.append(PlanetSurface(name=body_name, mass=mass, radius=radius, 
                                   pos=position, vel=velocity))
            # Handle restart for planetsurface body type
            # if restart:
            #     print(f"Reading Temperature data for World {body_array[-1].getName()}")
            #     snapshot_number = body_array[-1].getRestartParameters()

    # Set up the system
    print(f"Setting up system {system_name}")
    nbody_system = System(system_name, body_array)

    # Set orbit centers
    orbit_centers = [input_params.getIntVariable("OrbitCenter", i) for i in range(n_bodies)]
    nbody_system.setHostBodies(orbit_centers)

    # Setup orbits if required
    # Have not made this compatible for orbital parameters
    if input_params.getStringVariable("ParType") == "Orbital" and not restart:
        nbody_system.setupOrbits(orbit_centers)

    # Calculate initial system properties
    nbody_system.calcInitialProperties()
    
    # Right now illumination is just always set to false
    # Not 100% sure how to implement this
    nbody_system.setIllumination(input_params.getBoolVariable("PlanetaryIllumination"))

    # Set up output file
    output_file_name = input_params.getStringVariable("NBodyOutput")
    mode = "a" if restart and snapshot_number != 0 else "w"
    output_file = open(output_file_name, mode)
    if mode == "w":
        output_file.write(f"Number of Bodies, {n_bodies}\n")

    nbody_system.initialise2DFluxOutput(system_name)
    nbody_system.setFluxOutput(full_output)

    if full_output:
        print("Run will produce full output")

    # Convert times to code units
    t_max *= twopi
    t_snap *= twopi
    dt_max = 0.1 * t_snap

    print(f"Body Array: {body_array}")

    nbody_system.calcNBodyTimestep(bodies=body_array, dtmax=dt_max)
    dt_unit = nbody_system.getTimestep()
    time_unit = 0.0
    dtyr = dt_unit / twopi

    print("System set up: Running Simulation")
    while time_unit < t_max:
        t_stop = time_unit + t_snap
        dt_flux = 0.0

        while time_unit < t_stop:
            nbody_system.evolveSystem(dt_unit)
            time_unit += dt_unit
            dt_flux += dtyr

            nbody_system.calcNBodyTimestep(bodies=body_array, dtmax=dt_max)
            dt_unit = nbody_system.getTimestep()
            time_yr = time_unit / twopi
            dtyr = dt_unit / twopi

        print(f"Time: {time_yr:.4E} yr, N-Body Timestep: {dtyr:.4E} years, {dt_unit:.4E} units")

        nbody_system.calc2DFlux(time_yr, dt_flux)
        snapshot_number += 1

        # Output data
        nbody_system.outputNBodyData(output_file, time_yr, orbit_centers, input_params)
        nbody_system.output2DFluxData(snapshot_number, time_yr, system_name, input_params)

    # Close output file
    output_file.close()

    # Write integrated data
    nbody_system.outputIntegratedFluxData()
    nbody_system.outputInfoFile(snapshot_number, input_params)

    # End time
    elapsed_time = time.time() - start_time
    print(f"Run {nbody_system.getName()} complete")
    print(f"Wall Clock Runtime: {elapsed_time:.2f} s")
    
main(["ph", "orbital_input.txt"])