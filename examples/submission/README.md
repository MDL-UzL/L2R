
# L2R/examples/submission/zerofield

Examplare docker submission for inference time measurement

## HowTo Docker

We recommend using [repo2docker](https://github.com/jupyterhub/repo2docker) to containerize the algorithm:

`jupyter-repo2docker --no-run --user-name l2r --image vxmplus example_submissions/vxm++/`

Build the docker:

```bash
jupyter-repo2docker --no-run --user-name l2r --image zerofield zerofield/
```

Inference:

```bash
docker run --entrypoint /home/l2r/test.sh --gpus all 
-v /PATH_TO_DATA/NLST/:/home/l2r/data 
-v /PATH_TO_DATA/OUTPUT:/home/l2r/output 
zerofield /home/l2r/data/NLST_dataset.json
```

Of course, you may use docker without `jupyter-repo2docker`as well:

Build the docker:

```bash
docker build -t nlst_submission zerofield/
```

Inference:

```bash
docker run --rm  --mount src="/PATH/TO/DATA/NLST/",target=/l2r/data,type=bind 
--mount src="/PATH/TO/OUTPUT/",target=/l2r/output,type=bind nlst_submission -c /l2r/data/NLST_dataset.json
```
