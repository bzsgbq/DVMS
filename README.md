## Datasets

**Wu_MMSys_17**: 即vr-dataset; https://dl.acm.org/doi/10.1145/3083187.3083210

**Nasrabadi_MMSys_19**: https://dl.acm.org/doi/abs/10.1145/3304109.3325812

**David_MMSys_18**: https://dl.acm.org/doi/abs/10.1145/3204949.3208139

> *NOTE*: 
- [A-large-dataset](https://github.com/360VidStr/A-large-dataset-of-360-video-user-behaviour/tree/main) 的6个数据集中, 有4个数据集未在论文中明确 "视点位置坐标系" 与 "ERP格式的视频画面" 在位置上的对应关系, 因而无法进行对齐. 这里选择剩余两个 ([4] (Wu_MMSys_17) 和 [6] (Nasrabadi_MMSys_19) )进行处理. 
- David_MMSys_18数据集没有包含在A-large-dataset中, 是我之前做视点预测一直在用的主数据集, 因为它体积小, 训练测试块, 而且视点运动幅度大, HT+VC的VP模型更容易取得比较好的性能.


## Directory structure

```
DVMS
├── Wu_MMSys_17
│   ├── dataset
│   ├── sampled_dataset_thetaphi
│   │   ├── video1
│   │   │   ├── user1
│   │   │   ├── user2
│   │   │   └── ...
│   │   ├── video2
│   │   └── ...
│   └── saliency_src
│       ├── video1.npy
│       ├── video2.npy
│       └── ...
├── Nasrabadi_MMSys_19
└── David_MMSys_18
```

- `dataset` : 原始数据集
- `sampled_dataset_thetaphi` : 师兄最终需要的视点数据; 
    - 共有3列: ['playback_time(s)', 'theta', 'phi' ], 其中: theta ranges from 0 to 2*pi, and phi ranges from 0 to pi, origin of equirectangular in the top-left corner;
    - 采样率: 5Hz; 即0.2s采样一次;
    - 每个数据文件的读取方式为: `data = pd.read_csv(path, header=None)`;
- `saliency_src` : 
    - shape = (采样点数量, 224, 448); 即原始的saliency map的形状为224x448;
    - 可通过运行项目根目录下的 `resize_salmaps.py` 脚本将 `saliency_src` resize 成任意尺寸;


## Pipeline (using dataset Wu_MMSys_17 as an example)

```bash
cd path/to/DVMS
conda activate gbq_pytorch 
```

### Head movement traces

```bash
python Wu_MMSys_17/Read_Dataset.py --creat_orig_dat  # dataset --> original_dataset_xyz (统一目录结构, 以及视点位置的表示格式 (x,y,z) )
python Wu_MMSys_17/Read_Dataset.py --creat_samp_dat  # original_dataset_xyz --> sampled_dataset (统一采样率为5Hz, 即0.2s一个数据点)
python Wu_MMSys_17/Read_Dataset.py --creat_thph_dat  # sampled_dataset --> sampled_dataset_thetaphi (将视点位置表示格式从(x,y,z)转成(theta, phi) ; theta ranges from 0 to 2*pi, and phi ranges from 0 to pi, origin of equirectangular in the top-left corner)
```

trivial: 压缩以方便传输:
```bash
cd Wu_MMSys_17
zip -r dataset.zip ./dataset/
zip -r sampled.zip ./sampled_dataset_thetaphi/
```

### Saliency maps

#### 生成:

https://github.com/bzsgbq/PAVER-main

#### resize:

- 首先打开项目根目录下的 `resize_salmaps.py`, 配置dataset_name和SALMAP_SHAPE;
- 然后直接在项目根目录下运行脚本: `python resize_salmaps.py`


## Main References

https://gitlab.com/miguelfromeror/head-motion-prediction

https://gitlab.com/DVMS_/DVMS

https://github.com/HS-YN/PAVER