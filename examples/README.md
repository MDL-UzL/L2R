
# L2R/examples

Repository for L2R examples

So far, we provide the following examples:

* **AbdomenCTCT**: We first train a segmentation network as features extractor, followed by a two-step training registration network
* **NLST**: A ResNet/UNet-Combination. You can find its performance at https://learn2reg.grand-challenge.org/evaluation/task1-extended-labels/leaderboard/ (NLST Example)

We also provide easy-to-docker-examples located in `docker`:

* **zerofield**: A sample method which produces zero-fields as minimal working example
* **corrfield**: Correspondence fields for large motion image registration
* **Voxelmorph++**: Voxelmorph with keypoint supervision and multi-channel instance optimisation
