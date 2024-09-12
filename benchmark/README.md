# benchmark

The directory is used to save benchmark dataset.

1. Download [mlm4dnn_benchmark.tar.gz](https://mega.nz/file/4nkVwBCJ#06DpFjv9k7xLDiN_Zx0o9R3jfy1_ODYuqyhoEXJfmdo)
2. Extract and move it here: `tar xzvf mlm4dnn_benchmark.tar.gz`

The tree of this folder is as follows:
```
.
├── README.md
└── mlm4dnn_benchmark
    ├── samples
    │   ├── bug          # bug samples
    │   │   └── ...
    │   └── fixed        # fixed samples
    │       └── ...
    ├── train_result_dir # train result
    │   └── ...
    └── train_work_dir   # working dir for model training
        └── data         # dataset for samples in benchmark
            └── ...
```
