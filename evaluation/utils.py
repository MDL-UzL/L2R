import numpy as np
import scipy.ndimage
import nibabel as nib
from evalutils.exceptions import ValidationError
from scipy.ndimage import map_coordinates
from surface_distance import *

def warp(fix_seg,mov_seg,disp):
    D,H,W = fix_seg.shape
    identity = np.meshgrid(np.arange(D),
                           np.arange(H),
                           np.arange(W), indexing='ij')
    
    mov_seg_warped = map_coordinates(mov_seg, identity + disp.transpose(3,0,1,2), order=0)
    return mov_seg_warped

##### metrics #####
def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)

    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet

def compute_tre(fix_lms, mov_lms, disp, spacing_fix, spacing_mov):
    
    fix_lms_disp_x = map_coordinates(disp[:, :, :, 0], fix_lms.transpose())
    fix_lms_disp_y = map_coordinates(disp[:, :, :, 1], fix_lms.transpose())
    fix_lms_disp_z = map_coordinates(disp[:, :, :, 2], fix_lms.transpose())
    fix_lms_disp = np.array((fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)).transpose()

    fix_lms_warped = fix_lms * spacing_fix + fix_lms_disp
    
    return np.linalg.norm(fix_lms_warped - mov_lms * spacing_mov, axis=1)


def compute_dice(fixed,moving,moving_warped,labels):
    dice = 0
    count = 0
    for i in labels:
        if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
            continue
        dice += compute_dice_coefficient((fixed==i), (moving_warped==i))
        count += 1
    dice /= count
    return dice
    
def compute_hd95(fixed,moving,moving_warped,labels):
    hd95 = 0
    count = 0
    for i in labels:
        if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
            continue
        hd95 += compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i), np.ones(3)), 95.)
        count += 1
    hd95 /= count
    return hd95

##### validation errors #####
def raise_missing_file_error(fname):
    message = (
        f"The displacement field {fname} is missing. "
        f"Please provide all required displacement fields."
    )
    raise ValidationError(message)
    
def raise_dtype_error(fname, dtype):
    message = (
        f"The displacement field {fname} has a wrong dtype ('{dtype}'). "
        f"All displacement fields should have dtype 'float16'."
    )
    raise ValidationError(message)
    
def raise_shape_error(fname, shape, expected_shape):
    message = (
        f"The displacement field {fname} has a wrong shape ('{shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}'). "
        f"The expected shape of displacement fields for this task is {expected_shape[0]}x{expected_shape[1]}x{expected_shape[2]}x{expected_shape[3]}."
    )
    raise ValidationError(message)
