
# L2R/examples

Repository for L2R examples

So far, we provide the following examples:

* task_specific:
  - **AbdomenCTCT**: We first train a segmentation network as features extractor, followed by a two-step training registration network
  - **NLST**: A ResNet/UNet-Combination. You can find its performance at https://learn2reg.grand-challenge.org/evaluation/task1-extended-labels/leaderboard/ (NLST Example)

* basic_docker_examples:
  - **zerofield**: A sample method which produces zero-fields as minimal working example
  - **corrfield**: Correspondence fields for large motion image registration
  - **Voxelmorph++**: Voxelmorph with keypoint supervision and multi-channel instance optimisation

* Submission Examples:
  - **submission_example**: How to submit to L2R 2022 Task1 (NLST) for inference time testing

* Self-Configuring Examples:
  - **l2r_SCB**: A **S**elf**C**onfiguring**B**aseline for L2R 2002 Task3 
