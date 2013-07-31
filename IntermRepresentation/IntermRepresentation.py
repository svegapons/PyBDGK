"""
"""
import numpy as np
import os
import pdb

class IntermRep(object):
    """Intermediate representation for all kinds of brain data.

    Notes
    -----
    notes here...
    """    
    
    def __init__(self, arr_voxels, arr_xyz, subj_name = None, cls = None):
        """
        The intermediate representation is based on two arrays. 
        arr_voxels: containing a list of voxels
        arr_xyz: containing the x,y,z coordinates for each voxel.
        """
        self.arr_voxels = arr_voxels
        self.arr_xyz = arr_xyz
        self.subj_name = subj_name
        self.cls = cls
        
        
    @classmethod
    def load_from_file(cls, file_voxels, file_xyz):
        """
        Creates an IntermRep object from a file:
        @file_voxels: A file with the time series. Rows are voxels, and 
        columns are time points. 
        @file_xyz: files are the corresponding x, y, z coordinates of each 
        voxel.
        
        """   
        arr_voxels = np.genfromtxt(file_voxels, dtype = np.float, 
                                       delimiter = ' ')
        arr_xyz = np.genfromtxt(file_xyz, dtype = np.float, 
                                    delimiter = ' ')
        return cls(arr_voxels, arr_xyz)
                                    
        
