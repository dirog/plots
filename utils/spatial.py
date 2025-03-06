import numpy as np
from scipy.spatial import Delaunay, ConvexHull


def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def hull_points(points):
    try:
        hull = ConvexHull(np.array(points))
    except:
        print('Error in ConvexHull')
        return np.zeros((1,1))
    return hull.points[hull.vertices,:]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def get_convex_hull(mask, R_range, Ds_range, Du_range, mode='full', index=None):
    if mode == 'full':
        R_indices, Ds_indices, Du_indices = np.where(mask)
        tmp1 = [(R_range[i], Ds_range[j], Du_range[k]) for i, j, k in zip(R_indices, Ds_indices, Du_indices)]
        full = hull_points(tmp1)
        return np.array(full), np.array(tmp1)
    
    if mode == 'fixed R':
        Ds_indices, Du_indices = np.where(mask[index,:,:])
        tmp2 = [(Ds_range[j], Du_range[k]) for j, k in zip(Ds_indices, Du_indices)]
        fixed_R = hull_points(tmp2)
        return np.array(fixed_R), np.array(tmp2)
    
    if mode == 'fixed Ds':
        R_indices, Du_indices = np.where(mask[:,index,:])
        tmp3 = [(R_range[i], Du_range[k]) for i, k in zip(R_indices, Du_indices)]
        fixed_Ds = hull_points(tmp3)
        return np.array(fixed_Ds), np.array(tmp3)
    
    if mode == 'fixed Du':
        R_indices, Ds_indices = np.where(mask[:,:,index])
        tmp4 = [(R_range[i], Ds_range[j]) for i, j in zip(R_indices, Ds_indices)]
        fixed_Du = hull_points(tmp4)
        return np.array(fixed_Du), np.array(tmp4)
    
    else:
        raise ValueError()