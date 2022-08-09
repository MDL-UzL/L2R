
# L2R/examples/docker

Repository for (to be) dockerized sample methods:

* **zerofield**: A sample method which produces zero-fields as minimal working example
* **corrfield**: Correspondence fields for large motion image registration.

```text
[1] Heinrich, Mattias P., Heinz Handels, and Ivor JA Simpson. "Estimating large lung motion in COPD patients by symmetric regularised correspondence fields." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2015.

[2] Hansen, Lasse, and Mattias P. Heinrich. "GraphRegNet: Deep Graph Regularisation Networks on Sparse Keypoints for Dense Registration of 3D Lung CTs." IEEE Transactions on Medical Imaging (2021).
```

* **vxm++**: Voxelmorph++

```text
Heinrich, Mattias P., and Lasse Hansen. "Voxelmorph++ Going beyond the cranial vault with keypoint supervision and multi-channel instance optimisation" arxiv https://doi.org/10.48550/arXiv.2203.00046
```

## HowTo Docker

We recommend using [repo2docker](https://github.com/jupyterhub/repo2docker) to containerize these algorithms:

`jupyter-repo2docker --no-run --user-name l2r --image vxmplus example_submissions/vxm++/`

To start training simply call

```bash
docker run --entrypoint /home/l2r/train.sh --gpus all
-v /PATH/TO/Learn2Reg_Dataset_release_v1.1/AdbomenMRCT/:/home/l2r/data
-v /PATH/TO/OUTPUT/:/home/l2r/output
-v /PATH/TO/MODEL_FOLDER/:/home/l2r/models
vxmplus /PATH/TO/AbdomenMRCT_dataset.json 15000

```

For inference:

```bash
docker run --entrypoint /home/l2r/test.sh --gpus all
-v /PATH/TO/Learn2Reg_Dataset_release_v1.1/AdbomenMRCT/:/home/l2r/data
-v /PATH/TO/OUTPUT/:/home/l2r/output
-v /PATH/TO/MODEL_FOLDER/:/home/l2r/models
vxmplus /PATH/TO/AbdomenMRCT_dataset.json

```
