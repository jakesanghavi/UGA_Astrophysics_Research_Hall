from model_helpers import model_fun
from time import time, strftime, gmtime

MASS_RATIOS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]

for m in MASS_RATIOS:
    print(f"Starting model for mass ratio: {m}")
    
    start_time = time()
    
    model_fun(m, resolution="T42")
    
    end_time = time()
    elapsed_seconds = end_time - start_time
    
    formatted_time = strftime("%H:%M:%S", gmtime(elapsed_seconds))
    
    print(f"Finished mass ratio {m}. Execution time: {formatted_time}")
    print("-" * 30)