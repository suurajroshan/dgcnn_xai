# pGS-CAM implementation for DGCNN
Implementation of the paper *Interpretable LiDAR point cloud semantic segmentation via gradient based localization* for semantic point cloud segmentation. Allows to visualise the explainability study of the paper for DGCNN models.

## Pretrained Model (S3DIS dataset)
To run a pretrain of DGCNN (Dynamic Graph Convolution Neural Networks) S3DIS dataset:
```
python main_semseg_s3dis.py --exp_name=semseg_s3dis_6 --test_area=6 
```

## Training on Custom Dataset
To run a training on the custom dataset:
```

```

## Visualization
To run a visualisation run:
```

```


[TODO] FLAGS to implement:
- `load_pretrain: bool`
- `custom_datapath: str`
- `S3DIS_datapath: str`
