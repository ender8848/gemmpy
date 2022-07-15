# writing custom N-dimensional array containers that are 
# compatible with the numpy API and provide custom 
# implementations of numpy functionality.

import numpy as np

class IntervalArray:
    def __init__(self, array):
        """
        args:
            array: any real-valued array-like structure
        """
        
