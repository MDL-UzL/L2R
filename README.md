<img src="https://user-images.githubusercontent.com/57064392/167838907-719665bf-f513-43bd-8bad-c46dc11cc56b.png" alt="drawing" width="200"/>

# Learn2Reg Repository
Repository for additional L2R files

Including:
- evaluation files
- utilites (soon)


## What is Learn2Reg?
Learn2Reg is a comprehensive multi-task medical image registration challenge, hosted on https://learn2reg.grand-challenge.org/. Have a look!

Are you looking to benchmark your registration algorithm? Please visit https://learn2reg-test.grand-challenge.org/, where you can evaluate your alorithm on data from former Learn2Reg-challenges.

######  Motivation: Standardised benchmark for the best conventional and learning based medical registration methods:

* Analyse accuracy, robustness and speed on complementary tasks for clinical impact. 
* Remove entry barriers for new teams with expertise in deep learning but not necessarily registration.

######  Learn2Reg removes pitfalls for learning and applying transformations by providing:

* python evaluation code for voxel displacement fields and open-source code all evaluation metrics
* anatomical segmentation labels, manual landmarks, masks and keypoint correspondences for deep learning

######  Learn2Reg addresses four of the imminent challenges of medical image registration:

* learning from relatively small datasets
* estimating large deformations
* dealing with multi-modal scans
* learning from noisy annotations

######  Evaluation: Comprehensive and fair evaluation criteria that include:

* Dice / surface distance and TRE toe measure accuracy and robustness of transferring anatomical annotations 
* standard deviation and extreme values of Jacobian determinant to promote plausible deformations,
* low computation time for easier clinical translation evaluated using docker containers on GPUs provided by organisers.

Any questions? Head to our forum at https://learn2reg.grand-challenge.org/ or mail us directly via learn2reg@gmail.com.
