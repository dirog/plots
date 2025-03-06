import numpy as np
from utils.spatial import hull_points



def get_inner_convex_hull(mask, R_range, Ds_range, Du_range, index : int, type : str):
    if type == 'full':
        R_indices, Ds_indices, Du_indices = np.where(mask)
        tmp1 = [(R_range[i], Ds_range[j], Du_range[k])
                for i, j, k in zip(R_indices, Ds_indices, Du_indices)]
        return hull_points(tmp1)
    if type == 'fixed-rate':
        Ds_indices, Du_indices = np.where(mask[index,:,:])
        tmp2 = [(Ds_range[j], Du_range[k]) for j, k in zip(Ds_indices, Du_indices)]
        return hull_points(tmp2)
    if type == 'fixed-sem-dist':
        R_indices, Du_indices = np.where(mask[:,index,:])
        tmp3 = [(R_range[i], Du_range[k]) for i, k in zip(R_indices, Du_indices)]
        return hull_points(tmp3)
    if type == 'fixed-src-dist':
        R_indices, Ds_indices = np.where(mask[:,:,index])
        tmp4 = [(R_range[i], Ds_range[j]) for i, j in zip(R_indices, Ds_indices)]
        return hull_points(tmp4)
    else:
        raise ValueError('Unknown Type!')
    #return full, fixed_R, fixed_Ds, fixed_Du, (np.array(tmp2),np.array(tmp3),np.array(tmp4))