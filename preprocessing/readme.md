## PREPROCESSING for NLST (Learn2Reg 2022 Task 1)

1) **Identify** suitable patients with lung nodules and follow-up scans from https://wiki.cancerimagingarchive.net/display/NLST/National+Lung+Screening+Trial
2) **Download** a large file list (100-500 scans each), this can be automatised when creating a custom manifest file
3) **Convert** them from Dicom into Nifti using c3d 
4) **Segment** the region of interest (both lungs) using nnUNet in both scans
5) **Resample** the fixed (baseline) scan to 1.5mm isotropic resolution and fixed dimensions: 224x192x224 voxel
6) **Register** the moving image using linearBCV_quick and apply final transformation with single interpolation
7) **Establish** automatic correspondences using corrField 
