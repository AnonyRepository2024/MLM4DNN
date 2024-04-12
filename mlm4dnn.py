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


def repair_impl(
    bug_files_dir,
    fixed_files_dir,
    train_work_dir,
    at_env_name,
    output_dir,
    infill_api_config_file,
):
    repo2model_file = get_path("scripts/repo2model.py")
    repo2model_s_file = get_path("scripts/repo2model_s.py")
    at_lib_path = get_path("scripts/AUTOTRAINER/AutoTrainer/utils/utils.py")

    # Preprocess bug_files and fixed_files to formatted versions
    fmt_bug_files_dir = os.path.join(output_dir, "fmt_bug_files")
    fmt_fixed_files_dir = os.path.join(output_dir, "fmt_fixed_files")
    os.makedirs(fmt_bug_files_dir, exist_ok=True)
    os.makedirs(fmt_fixed_files_dir, exist_ok=True)

    launch_cmd = """\
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
            launch_cmd.format(
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
                launch_cmd.format(
                    files_dir=fixed_files_dir,
                    fmt_files_dir=fmt_fixed_files_dir,
                    repo2model_s_file=repo2model_s_file,
                )
            )
    else:
        pgrsu._wlog(">>> Not found fixed files, IGNORE")

    # Launch Reapir Procedure
    launch_cmd = """\
python scripts/model_repair.py \
    --buggy-models-dir "{fmt_bug_files_dir}" \
    --buggy-models-with-perfect-fl-dir DUMMY \
    --correct-models-dir "{fmt_fixed_files_dir}" \
    --train-work-dir "{train_work_dir}" \
    --infill-api-name FinetunedUniXcoder \
    --infill-api-config-file "{infill_api_config_file}" \
    --repo2model-path "{repo2model_file}" \
    --model-train-env-name "{at_env_name}" \
    --autotrainer-lib-path "{at_lib_path}" \
    --autotrainer-env-name "{at_env_name}" \
    --out-dir "{output_dir}" \
    --ops 1 2 3 4 5 6"""

    print("======================= Repair... =======================")
    system(
        launch_cmd.format(
            fmt_bug_files_dir=fmt_bug_files_dir,
            fmt_fixed_files_dir=fmt_fixed_files_dir,
            train_work_dir=train_work_dir,
            infill_api_config_file=infill_api_config_file,
            repo2model_file=repo2model_file,
            at_env_name=at_env_name,
            at_lib_path=at_lib_path,
            output_dir=output_dir,
        )
    )


def train():
    finetune_config_jf = get_path("configs/finetune_config.json")
    with open(finetune_config_jf, "r", encoding="UTF-8") as fp:
        finetune_config = json.load(fp)

    launch_cmd = """\
cd {root_dir} && bash scripts/finetune_unixcoder.sh \
    {model_name_or_path} \
    {dataset_path} \
    {output_dir} \
    {tag_prefix}"""

    finetune_config["root_dir"] = get_root_dir()
    system(launch_cmd.format_map(finetune_config))


def repro():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at-env-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--infill-api-config-file", type=str, default=None)

    args = parser.parse_args()

    at_env_name = args.at_env_name
    output_dir = args.output_dir
    infill_api_config_file = args.infill_api_config_file

    if not infill_api_config_file:
        infill_api_config_file = get_path("configs/infill_api_config.json")

    bug_files_dir = get_path("benchmark/mlm4dnn_benchmark/samples/bug")
    fixed_files_dir = get_path("benchmark/mlm4dnn_benchmark/samples/fixed")
    train_work_dir = get_path("benchmark/mlm4dnn_benchmark/train_work_dir")

    repair_impl(
        bug_files_dir=bug_files_dir,
        fixed_files_dir=fixed_files_dir,
        train_work_dir=train_work_dir,
        at_env_name=at_env_name,
        output_dir=output_dir,
        infill_api_config_file=infill_api_config_file,
    )


def repair():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at-env-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--bug-files-dir", type=str, required=True)
    parser.add_argument("--fixed-files-dir", type=str, default="DUMMY")
    parser.add_argument("--train-work-dir", type=str, required=True)
    parser.add_argument("--infill-api-config-file", type=str, default=None)

    args = parser.parse_args()

    at_env_name = args.at_env_name
    output_dir = args.output_dir
    bug_files_dir = args.bug_files_dir
    fixed_files_dir = args.fixed_files_dir
    train_work_dir = args.train_work_dir
    infill_api_config_file = args.infill_api_config_file

    if not infill_api_config_file:
        infill_api_config_file = get_path("configs/infill_api_config.json")

    repair_impl(
        bug_files_dir=bug_files_dir,
        fixed_files_dir=fixed_files_dir,
        train_work_dir=train_work_dir,
        at_env_name=at_env_name,
        output_dir=output_dir,
        infill_api_config_file=infill_api_config_file,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mlm4dnn.py <subcmd> ...")
        exit(-1)

    available_subcmds = ["train", "repro", "repair"]
    subcmd = sys.argv.pop(1)
    if subcmd not in available_subcmds:
        print(f"Available subcmds: {available_subcmds}")
        exit(-1)

    eval(f"{subcmd}")()
