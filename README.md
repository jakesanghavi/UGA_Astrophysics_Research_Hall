# UGA_Astrophysics_Research_Hall
GitHub repository for astrophysics/astrobiology research with Dr. Cassandra Hall at the University of Georgia.

# Files Guide

## Helper Files
`constants.py`: contains constants that are helpful to have for astronomical/chemical calculations
`utils.py`: contains functions mainly used to perform calculations for photosynthesis rate maps

## Photosynthesis Rate Map
`photosynthesis_rate_plotter.py`: brings in functions from `utils.py` and creates the photosythesis rate plot grid shown in the meeting

## M-Body Simulator
`M_Body_Simulator`: folder containing all files which lay the foundation for the simulation
`m_body_simulator.py`: file which calls to the above folder and runs the simulation
`orbital_input.txt` and `positional_input.txt`: two different ways to give the input to the above Python file