# MLM4DNN (Element-based Automated DNN Repair with Fine-tuned Masked Language Model)

## Main Components
1. Fine-tuning
2. Patch Generation
3. Patch Filtering
4. Patch Validation

## Preparation
1. Clone repository
    ```shell
    git clone https://github.com/AnonymousRepository2024/MLM4DNN.git
    ```

2. Benchmark preparation
    * Refer to [benchmark/README.md](https://github.com/AnonymousRepository2024/MLM4DNN/blob/main/benchmark/README.md) to get our benchmark (i.e., $Benchmark_{51}$ and $Benchmark_{38}$)

3. Dataset preparation
    * Refer to [dataset/README.md](https://github.com/AnonymousRepository2024/MLM4DNN/blob/main/dataset/README.md) to get our dataset (i.e., $Dataset_{MLM}$)

4. Model preparation
    * Refer to [models/README.md](https://github.com/AnonymousRepository2024/MLM4DNN/blob/main/models/README.md) to get our fine-tuned model

5. Main [conda](https://www.anaconda.com/) env preparation (for component 1,2,3)
    * Create: ```conda create --name mlm4dnn_main python=3.11.0```
    * Install libraries for this env
        * pytorch 2.1.0
        * transformers 4.36.2
        * requests 2.31.0
        * tqdm 4.66.1
    * See [mlm4dnn_main_env.yml](https://github.com/AnonymousRepository2024/MLM4DNN/blob/main/mlm4dnn_main_env.yml) for details

6. PV [conda](https://www.anaconda.com/) env preparation (for component 4)
    * Create: ```conda create --name mlm4dnn_pv python=3.6.13```
    * Install libraries for this env
        * keras 2.3.1
        * tensorflow 2.1.0
        * numpy 1.18.5
        * pandas 1.0.5
        * scikit-learn 0.23.1
        * astunparse 1.6.3
    * See [mlm4dnn_pv_env.yml](https://github.com/AnonymousRepository2024/MLM4DNN/blob/main/mlm4dnn_pv_env.yml) for details

## Repro on $Benchmark_{51}$ and $Benchmark_{38}$

```shell
conda activate mlm4dnn_main
python mlm4dnn.py repro --at-env-name mlm4dnn_pv --output-dir /path/to/output
```

## Repro on Your Dataset

1. Prepare your dataset
    * Put your bug DNNs in a directory (`--bug-files-dir`)
    * Make a working directory in which these DNNs can be trained (`--train-work-dir`)

2. Launch MLM4DNN
    ```shell
    conda activate mlm4dnn_main
    python mlm4dnn.py repair \
        --at-env-name mlm4dnn_pv \
        --output-dir /path/to/output \
        --bug-files-dir /path/to/bug_dnns \
        --train-work-dir /path/to/working_dir_for_training
    ```

## Re-Finetune Model & Perform MLM4DNN on New Model

1. Model Fine-tuning
    * ```conda activate mlm4dnn_main```
    * ```python mlm4dnn.py train```
    * The log and checkpoints will be saved in models/

2. MLM4DNN Performing
    * Create a config file (refer to [configs/infill_api_config.json](https://github.com/AnonymousRepository2024/MLM4DNN/blob/main/configs/infill_api_config.json))
        * Change `output_dir` to new model's
    * To `Repro`: `python mlm4dnn.py repro ... --infill-api-config-file /path/to/your_config`
    * To `Repair`: `python mlm4dnn.py repair ... --infill-api-config-file /path/to/your_config`

## More infomation

* Complete [Table-3](https://github.com/AnonymousRepository2024/MLM4DNN/blob/main/appendix/table3.xlsx) that shows results of RQ1 in our paper
* The [outputs](https://github.com/AnonymousRepository2024/MLM4DNN/blob/main/appendix/deepdiagnosis_outputs.xlsx) of [DeepDiagnosis](https://github.com/DeepDiagnosis/ICSE2022) on our two benchmarks
