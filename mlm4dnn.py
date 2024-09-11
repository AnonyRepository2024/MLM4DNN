import os
import sys
import json
import argparse
import subprocess
import scripts._rs_utils as pgrsu


def get_root_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def get_path(relpath: str, check=True) -> str:
    path = os.path.join(get_root_dir(), relpath)
    if check and not os.path.exists(path):
        raise RuntimeError(f"Not found {path}")
    return path


def system(cmd: str) -> int:
    return subprocess.run(
        [cmd], cwd=get_root_dir(), stdout=None, stderr=None, shell=True
    ).returncode


def repair_and_validate(
    bug_files_dir,
    fixed_files_dir,
    bugfixed_train_result_dir,
    train_work_dir,
    train_env_name,
    output_dir,
    infill_api_config_file,
):
    repo2model_file = get_path("scripts/repo2model.py")
    repo2model_s_file = get_path("scripts/repo2model_s.py")
    infill_api_name = pgrsu._load_json(infill_api_config_file)["_infill_api_name"]

    # Preprocess bug_files and fixed_files to formatted versions
    fmt_bug_files_dir = os.path.join(output_dir, "fmt_bug_files")
    fmt_fixed_files_dir = os.path.join(output_dir, "fmt_fixed_files")
    os.makedirs(fmt_bug_files_dir, exist_ok=True)
    os.makedirs(fmt_fixed_files_dir, exist_ok=True)

    launch_format_cmd = """\
python scripts/format_models.py \
    "{files_dir}" \
    "{fmt_files_dir}" \
    {repo2model_s_file}"""

    pgrsu._ilog("===================== Preprocess... =====================")

    if sorted(os.listdir(bug_files_dir)) == sorted(os.listdir(fmt_bug_files_dir)):
        pgrsu._wlog(">>> Bug files have been processed, SKIP")
    else:
        pgrsu._ilog(">>> Process bug files...")
        system(
            launch_format_cmd.format(
                files_dir=bug_files_dir,
                fmt_files_dir=fmt_bug_files_dir,
                repo2model_s_file=repo2model_s_file,
            )
        )

    if os.path.exists(fixed_files_dir):
        if sorted(os.listdir(fixed_files_dir)) == sorted(
            os.listdir(fmt_fixed_files_dir)
        ):
            pgrsu._wlog(">>> Fixed files have been processed, SKIP")
        else:
            pgrsu._ilog(">>> Process fixed files...")
            system(
                launch_format_cmd.format(
                    files_dir=fixed_files_dir,
                    fmt_files_dir=fmt_fixed_files_dir,
                    repo2model_s_file=repo2model_s_file,
                )
            )
    else:
        pgrsu._wlog(">>> Not found fixed files, IGNORE")

    # Launch Reapir Procedure
    print("======================= Repairing... =======================")

    launch_repair_cmd = """\
python scripts/model_repair.py \
    --buggy-models-dir  {fmt_bug_files_dir} \
    --correct-models-dir {fmt_fixed_files_dir} \
    --train-work-dir {train_work_dir} \
    --infill-api-name {infill_api_name} \
    --infill-api-config-file {infill_api_config_file} \
    --repo2model-path {repo2model_file} \
    --model-train-env-name {train_env_name} \
    --out-dir {output_dir} \
    --ops {ops}"""

    system(
        launch_repair_cmd.format(
            fmt_bug_files_dir=fmt_bug_files_dir,
            fmt_fixed_files_dir=fmt_fixed_files_dir,
            train_work_dir=train_work_dir,
            infill_api_name=infill_api_name,
            infill_api_config_file=infill_api_config_file,
            repo2model_file=repo2model_file,
            train_env_name=train_env_name,
            output_dir=output_dir,
            ops="1 2 3",  # 1: Gen DNNm; 2: Infill DNNm; 3: Filter Patch
        )
    )

    # Launch Validation Procedure
    print("======================= Validating... =======================")

    # bug_fixed_train_result_dir=FK2_benchmark_SgEle_benchmarkNew_train_stat
    # patches_root_dir=model_repair_without_loc/FK2_finetuned_unixcoder_repair_results
    # output_dir=model_repair_without_loc/FK2_finetuned_unixcoder_repair_results/validate_results

    system(
        launch_repair_cmd.format(
            fmt_bug_files_dir=fmt_bug_files_dir,
            fmt_fixed_files_dir=fmt_fixed_files_dir,
            train_work_dir=train_work_dir,
            infill_api_name=infill_api_name,
            infill_api_config_file=infill_api_config_file,
            repo2model_file=repo2model_file,
            train_env_name=train_env_name,
            output_dir=output_dir,
            ops="4",  # 4: Prepare files for validation
        )
    )

    launch_validation_cmd = """\
python scripts/validate_patches.py \
    --bug_fixed_train_result_dir {bug_fixed_train_result_dir} \
    --patches_root_dir {patches_root_dir} \
    --output_dir {output_dir}"""

    system(
        launch_validation_cmd.format(
            bug_fixed_train_result_dir=bugfixed_train_result_dir,
            patches_root_dir=output_dir,
            output_dir=f"{output_dir}/validate_results",
        )
    )

    # Collect results
    print("======================= Collecting Results... =======================")
    launch_collect_result_cmd = """\
python scripts/collect_result.py \
    --correct_models_dir {fmt_fixed_files_dir} \
    --patch_source_filename filtered_possible_repaired_models.json \
    --repair_result_dir {output_dir} \
    --verbose"""

    system(
        launch_collect_result_cmd.format(
            fmt_fixed_files_dir=fmt_fixed_files_dir,
            output_dir=output_dir,
        )
    )


def train():
    finetune_config_jf = get_path("configs/finetune_config.json")
    with open(finetune_config_jf, "r", encoding="UTF-8") as fp:
        finetune_config = json.load(fp)

    launch_cmd = """\
IM4DNN_REMOVE_ROOT=1 bash scripts/finetune_unixcoder.sh \
    {model_name_or_path} \
    {dataset_path} \
    {output_dir} \
    {tag_prefix}"""

    system(launch_cmd.format_map(finetune_config))


def repro():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infill-api-config-file", type=str, default=None)
    parser.add_argument("--dnn-train-env-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    args = parser.parse_args()

    infill_api_config_file = args.infill_api_config_file
    dnn_train_env_name = args.dnn_train_env_name
    output_dir = args.output_dir

    if not infill_api_config_file:
        infill_api_config_file = get_path("configs/infill_api_config.json")

    bug_files_dir = get_path("benchmark/mlm4dnn_benchmark/samples/bug")
    fixed_files_dir = get_path("benchmark/mlm4dnn_benchmark/samples/fixed")
    train_work_dir = get_path("benchmark/mlm4dnn_benchmark/train_work_dir")
    bugfixed_train_result_dir = get_path("benchmark/mlm4dnn_benchmark/train_result_dir")

    repair_and_validate(
        bug_files_dir=bug_files_dir,
        fixed_files_dir=fixed_files_dir,
        bugfixed_train_result_dir=bugfixed_train_result_dir,
        train_work_dir=train_work_dir,
        train_env_name=dnn_train_env_name,
        output_dir=output_dir,
        infill_api_config_file=infill_api_config_file,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mlm4dnn.py <subcmd> ...")
        exit(-1)

    available_subcmds = ["train", "repro"]
    subcmd = sys.argv.pop(1)
    if subcmd not in available_subcmds:
        print(f"Available subcmds: {available_subcmds}")
        exit(-1)

    eval(f"{subcmd}")()
