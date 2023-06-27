# Hyperspherical Embedding for Point Cloud Completion

### [[Project Page]](https://haomengz.github.io/hyperpc/index.html) [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Hyperspherical_Embedding_for_Point_Cloud_Completion_CVPR_2023_paper.pdf)
This repository contains source code for Hyperspherical Embedding for Point Cloud Completion (CVPR 2023).


## Prerequisites
1. Download the datasets for point cloud completion: [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip), [Completion3D](http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip), [MVP](https://mvp-dataset.github.io/index.html), [ShapeNet](https://shapenet.org/). Fill in the corresponding data path in `run.sh`.

2. Check the docker files in `docker/` and build docker image:
```bash
cd docker/
./build.sh
```

3. Create docker container using one GPU
```bash
./run.sh 0
```

4. Check the config file `cfgs/config.yaml` each time you run the experiment.

## Training
```bash
./train.sh
```
+ For multi-task learning: Set the `task` in `cfgs/config.yaml` to be a list of the desired tasks. For example, to train on both classification and comletion tasks, set `task` to be `['classification','completion']`.

## Evaluation
Keep all the config parameters as training, and set `eval` to `True`, and then run:

```bash
./train.sh
```

## TensorBoard Visualization
Set the `--logdir` in `tensorboard.sh` to be the desired log directory and run:
```bash
./tensorboard.sh
```

## Citations
If you find this work useful for your research, please cite HyperPC in your publications.

```
@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Junming and Zhang, Haomeng and Vasudevan, Ram and 
                Johnson-Roberson, Matthew},
    title     = {Hyperspherical Embedding for Point Cloud Completion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and 
                Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {5323-5332}
}
```

