# benchmark

The directory is used to save benchmark dataset.

1. Download [mlm4dnn_benchmark.tar.gz](https://mega.nz/file/o2NXhKaT#YQFkmbWnYyeTyOt6m9qKc0s_ZPxwSqiwsxedgANxEYs)
2. Extract and move it here: `tar xzvf mlm4dnn_benchmark.tar.gz`

The tree of this folder is as follows:
```
.
├── README.md
└── mlm4dnn_benchmark
    ├── benchmark.csv  # samples with metainfo
    ├── samples
    │   ├── bug        # bug samples
    │   │   └── ...
    │   └── fixed      # fixed samples
    │       └── ...
    └── train_work_dir # working dir for model training
        └── data       # dataset for samples in benchmark
            └── ...
```
