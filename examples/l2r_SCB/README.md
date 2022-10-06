# L2R-SelfConfiguringBaseline
Build the docker:

```bash
docker build  -t l2r_scb l2r_SCB/
```

Examplary Training:

```bash
docker run --rm --gpus all \
-v "$(pwd)"/models:/l2r/models \
-v "$(pwd)"/output/:/l2r/output/ \
-v "$(pwd)"/../HIDDENDATA/AbdomenMRMR:/l2r/data:ro \
--entrypoint ./train.sh \
l2r_scb data/AbdomenMRMR_dataset.json 5000 0
```
Examplary Inference:

```bash
docker run --rm --gpus all \
-v "$(pwd)"/models:/l2r/models \
-v "$(pwd)"/output/:/l2r/output/ \
-v "$(pwd)"/../HIDDENDATA/AbdomenMRMR:/l2r/data:ro \
--entrypoint ./test.sh \
l2r_scb data/AbdomenMRMR_dataset.json 0
```
