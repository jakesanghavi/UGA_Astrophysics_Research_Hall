import numpy as np
import os
from constants import msolToMEarth, solradToREarth

# Define variable groups as seen in the C++
string_var = ["Output_Dir", "Integrator", "ParType", "NBodyOutput"]
bool_var = ["Restart", "FullOutput", "PlanetaryIllumination"]
int_var = ["Number_Bodies", "NLatitude", "NLongitude", "NLambda"]
double_var = ["TimeStep", "SomeDouble", "SnapshotTime", "MaximumTime", "TotalMass"]
vector_string_var = ["BodyName", "BodyType"]
vector_int_var = ["OrbitCenter"]
vector_double_var = ["Position", "Velocity", "Mass", "Radius", "SemiMajorAxis", "Eccentricity", \
                    "Inclination", "LongAscend", "Periapsis", "MeanAnomaly", "Luminosity", \
                    "EffectiveTemperature", "Obliquity"]

class ParFile:
    def __init__(self, filename=None):
        # A few constants that aren't found in astropy
        self.screenBar = "-" * 60
        self.msolToMEarth = msolToMEarth
        self.solradToREarth = solradToREarth
        
        # Dictionaries for variable storage
        self.string_variables = {}
        self.int_variables = {}
        # Set this as default 0
        self.double_variables = {"TotalMass": 0}
        self.vector_string_variables = {}
        self.vector_int_variables = {}
        self.vector_double_variables = {}
        self.bool_variables = {}
        self.variable_locations = {}

        self.set_variable_locations()

        # Read in the .txt input file
        if filename:
            self.read_file(filename)

    # Cateogirze each variable
    def set_variable_locations(self):
        for var in string_var: self.variable_locations[var] = "string"
        for var in bool_var: self.variable_locations[var] = "bool"
        for var in int_var: self.variable_locations[var] = "int"
        for var in double_var: self.variable_locations[var] = "double"
        for var in vector_string_var: self.variable_locations[var] = "vector_string"
        for var in vector_int_var: self.variable_locations[var] = "vector_int"
        for var in vector_double_var: self.variable_locations[var] = "vector_double"

    # Read in the file
    def read_file(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        body_index = -1
        
        # Read line by line and set the variable dicts
        for line in lines:
            line = line.strip()
            if not line or line.startswith('--'):
                continue
            tokens = line.split()
            par = tokens[0]
            if par not in self.variable_locations:
                continue
            body_index = self.read_variable(par, tokens[1:], body_index)

        self.convert_to_radians(self.int_variables.get("Number_Bodies", 0))
        self.initialize_all_booleans()
        self.double_variables["SystemTime"] = 0.0

        if self.bool_variables.get("Restart", False):
            self.setup_restart_positions()
    
    # Read in and set each variable according to its type
    def read_variable(self, par, values, body_index):
        vtype = self.variable_locations[par]

        if vtype == "string":
            self.string_variables[par] = values[0]
        elif vtype == "int":
            self.int_variables[par] = int(values[0])
            if par == "Number_Bodies":
                self.initialize_vectors(self.int_variables[par])
        elif vtype == "double":
            self.double_variables[par] = float(values[0])
        elif vtype == "bool":
            self.bool_variables[par] = values[0].lower() in ['t', 'y', 'true', '1']
        elif vtype == "vector_string":
            # Only increment at the start of a new body
            if par == "BodyName":
                body_index += 1
            self.vector_string_variables.setdefault(par, [])[body_index:body_index+1] = [values[0]]
        elif vtype == "vector_int":
            self.vector_int_variables.setdefault(par, [])[body_index:body_index+1] = [int(values[0])]
        elif vtype == "vector_double":
            if par in ["Position", "Velocity"]:
                x, y, z = map(float, values[:3])
                self.vector_double_variables.setdefault(f"X{par}", [])[body_index:body_index+1] = [x]
                self.vector_double_variables.setdefault(f"Y{par}", [])[body_index:body_index+1] = [y]
                self.vector_double_variables.setdefault(f"Z{par}", [])[body_index:body_index+1] = [z]
            elif par == "Mass":
                self.double_variables["TotalMass"] += float(values[0])
                self.vector_double_variables.setdefault(par, [])[body_index:body_index+1] = [float(values[0])]
            else:
                self.vector_double_variables.setdefault(par, [])[body_index:body_index+1] = [float(values[0])]
        return body_index

    # Initialize "vector" variables
    # This includes things like velocity which will have more than 1 dimension
    # For some reason all the body details (even mass, for example)
    # are classified as "vectors". I've kept that in our version but can be
    # changed if needed
    def initialize_vectors(self, n_bodies):
        self.vector_int_variables = {var: [0] * n_bodies for var in vector_int_var}

        for var in vector_double_var:
            if var in ["Position", "Velocity"]:
                for axis in "XYZ":
                    self.vector_double_variables[f"{axis}{var}"] = [0.0] * n_bodies
            else:
                self.vector_double_variables[var] = [0.0] * n_bodies

        # Initialize string vectors only if n_bodies > 0
        self.vector_string_variables = {var: [] for var in vector_string_var} if n_bodies > 0 else {}

    # Init booleans to false if nonexistent
    def initialize_all_booleans(self):
        for par in bool_var:
            self.bool_variables[par] = self.bool_variables.get(par, False)

    # Not yet implemented
    def convert_to_radians(self, n_bodies):
        pass

    # Return the position of a body in 3-space
    def get_body_position(self, index):
        return np.array([
            self.vector_double_variables["XPosition"][index],
            self.vector_double_variables["YPosition"][index],
            self.vector_double_variables["ZPosition"][index]
        ])

    # Return the velocity of a body in 3-space
    def get_body_velocity(self, index):
        return np.array([
            self.vector_double_variables["XVelocity"][index],
            self.vector_double_variables["YVelocity"][index],
            self.vector_double_variables["ZVelocity"][index]
        ])

    # Restart logic. Have to test this more
    def setup_restart_positions(self):
        restart_dir = self.string_variables.get("Output_Dir", ".")
        restart_file = os.path.join(restart_dir, "restart.txt")

        if not os.path.exists(restart_file):
            print(f"Restart file {restart_file} not found. Starting fresh.")
            return

        print(f"Reading restart file: {restart_file}")
        with open(restart_file, 'r') as f:
            lines = f.readlines()

        n_bodies = self.int_variables.get("Number_Bodies", 0)
        # Determine which vectors we have (assume each line has all vectors in order)
        vector_keys = list(self.vector_double_variables.keys())

        for i, line in enumerate(lines):
            if i >= n_bodies:
                break
            tokens = line.strip().split()
            if len(tokens) < len(vector_keys):
                print(f"Skipping incomplete line {i}: {line}")
                continue
            for j, key in enumerate(vector_keys):
                self.vector_double_variables[key][i] = float(tokens[j])

        print("Restart data loaded successfully for all vectors.")
    
    # Checks params are defined properly
    def checkParameters(self):
        # Set a default name for system if not specified
        if self.string_variables.get("SystemName", "") == "":
            self.string_variables["SystemName"] = "System"

        # Not sure why the below values are the ones checked
        # That is how it was in the C++ but we can change this as needed
        
        # Ensure certain integer values are defined
        for key in ["NLatitude", "NLongitude", "NLambda"]:
            self.checkIntValueDefined(key)

        # Ensure certain double values are defined
        for key in ["SnapshotTime", "MaximumTime"]:
            self.checkDoubleValueDefined(key)

    # Print out the params from the file
    def displayParameters(self):
        """
        Writes all inputted parameters to the screen
        """
        print("Global Parameters:")
        print(self.screenBar)
        print(f"System Name: {self.string_variables.get('SystemName', '')}")
        print(f"Initial Time: {self.double_variables.get('SystemTime', 0.0):.1E}")
        print(f"Number of Bodies: {self.int_variables.get('Number_Bodies', 0)}")
        print(f"N Body Output File: {self.string_variables.get('NBodyOutput', '')}")
        print(f"Maximum Time: {self.double_variables.get('MaximumTime', 0.0):.1E} years")
        print(f"Snapshot Time: {self.double_variables.get('SnapshotTime', 0.0):.1E} years")
        print(self.screenBar)

        if self.bool_variables.get("Restart", False):
            print("This is a restart - using vector data from pre-existing nbody output file")

        if self.bool_variables.get("PlanetaryIllumination", False):
            print("Planetary Illumination is ON")
        else:
            print("Planetary Illumination is OFF")

        print(self.screenBar)
        print("Individual Body Parameters")
        print(self.screenBar)

        n_bodies = self.int_variables.get("Number_Bodies", 0)
        for i in range(n_bodies):
            name = self.vector_string_variables["BodyName"][i]
            body_type = self.vector_string_variables["BodyType"][i]
            print(f"Body {i}: Name {name}, Type {body_type}")

            if body_type == "Star":
                mass = self.vector_double_variables["Mass"][i]
                radius = self.vector_double_variables["Radius"][i]
                print(f"Mass: {mass:.2f} solar masses")
                print(f"Radius: {radius} solar radii")
            else:
                # Convert from solar units to Earth units
                mass = self.vector_double_variables["Mass"][i] * self.msolToMEarth
                radius = self.vector_double_variables["Radius"][i] * self.solradToREarth
                print(f"Mass: {mass:.2f} Earth masses")
                print(f"Radius: {radius} Earth radii")

            par_type = self.string_variables.get("ParType", "Positional")
            if par_type == "Positional":
                print("Position:")
                print(self.get_body_position(i))
                print("Velocity:")
                print(self.get_body_velocity(i))
            elif par_type == "Orbital":
                print("Orbit: a e i LongAscend Periapsis MeanAnomaly")
                a = self.vector_double_variables["SemiMajorAxis"][i]
                e = self.vector_double_variables["Eccentricity"][i]
                inc = self.vector_double_variables["Inclination"][i]
                long_asc = self.vector_double_variables["LongAscend"][i]
                peri = self.vector_double_variables["Periapsis"][i]
                mean_anom = self.vector_double_variables["MeanAnomaly"][i]
                print(f"{a} {e} {inc} {long_asc} {peri} {mean_anom}")

            print(self.screenBar)

    # Checking methods
    def checkIntValueDefined(self, key):
        if key not in self.int_variables:
            raise ValueError(f"Integer variable '{key}' is not defined.")

    def checkDoubleValueDefined(self, key):
        if key not in self.double_variables:
            raise ValueError(f"Double variable '{key}' is not defined.")
        
        
    # Various getter methods
    def getDoubleVariable(self, name, default=None):
        if name in self.double_variables:
            return self.double_variables[name]
        elif default is not None:
            return default
        else:
            raise ValueError(f"Double variable '{name}' is not defined.")

    def getIntVariable(self, name, default=None):
        if name in self.int_variables:
            return self.int_variables[name]
        elif default is not None:
            return default
        else:
            raise ValueError(f"Integer variable '{name}' is not defined.")

    def getStringVariable(self, name, ix=None):
        # Use ix to differentiate between vec and non vec
        if name in self.string_variables:
            return self.string_variables[name]
        elif ix is not None:
            return self.vector_string_variables[name][ix]
        else:
            raise ValueError(f"String variable '{name}' is not defined.")
        
    def getDoubleVariable(self, name, ix=None):
        # Use ix to differentiate between vec and non vec
        if name in self.double_variables:
            return self.double_variables[name]
        elif ix is not None:
            return self.vector_double_variables[name][ix]
        else:
            raise ValueError(f"Double variable '{name}' is not defined.")

    def getIntVariable(self, name, ix=None):
        # Use ix to differentiate between vec and non vec
        if name in self.int_variables:
            return self.int_variables[name]
        elif ix is not None:
            return self.vector_int_variables[name][ix]
        else:
            raise ValueError(f"Integer variable '{name}' is not defined.")

    def getBoolVariable(self, name):
        if name in self.bool_variables:
            return self.bool_variables[name]
        else:
            raise ValueError(f"Bool variable '{name}' is not defined.")