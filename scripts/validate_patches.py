import os
import sys
import csv
import tqdm
import glob
import pickle
import argparse
import pandas as pd
import numpy as np
import statsmodels
import statsmodels.api as sm
from patsy import dmatrices

import _rs_utils as pgrsu


def _sp_run(command, **kwargs):
    import subprocess as sp

    try:
        result = sp.run(
            command, shell=True, check=True, stdout=sp.PIPE, stderr=sp.PIPE, **kwargs
        )

        return_code = result.returncode
        stdout = result.stdout.decode("utf-8")
        stderr = result.stderr.decode("utf-8")

        if return_code != 0:
            print(f"[W] Run '{command}' failed: out='{stdout}', err='{stderr}'")

        return return_code, stdout, stderr
    except KeyboardInterrupt:
        if "y" == input("exit? (y/n)").lower():
            exit(0)
        return -1, "", ""
    except sp.CalledProcessError as e:
        stdout = e.output.decode("utf-8") if e.output else ""
        stderr = e.stderr.decode("utf-8") if e.stderr else ""
        return e.returncode, stdout, stderr


argparser = argparse.ArgumentParser()
argparser.add_argument("--bug_fixed_train_result_dir", type=str, required=True)
argparser.add_argument("--patches_root_dir", type=str, required=True)
argparser.add_argument("--output_dir", type=str, required=True)
argparser.add_argument("--n", type=int, default=20)
argparser.add_argument("--alpha", type=int, default=0.05)
argparser.add_argument("--beta", type=int, default=0.2)
argparser.add_argument(
    "--patch-subdir",
    type=str,
    default="filtered_possible_repaired_models-trainable_sfmodels",
)

args = argparser.parse_args()
bug_fixed_train_result_dir = args.bug_fixed_train_result_dir
patches_root_dir = args.patches_root_dir
output_dir = args.output_dir
N = args.n
alpha = args.alpha
beta = args.beta
patch_subdir = args.patch_subdir
assert os.path.exists(bug_fixed_train_result_dir)
assert os.path.exists(patches_root_dir)
# assert os.path.exists(output_dir)
os.makedirs(output_dir, exist_ok=True)


def train_sfmodel(sfmodel_dir, sfmodel_output_dir, log_path):
    sfmodel_dir = os.path.abspath(sfmodel_dir)
    model_path = os.path.join(sfmodel_dir, "__MODEL__.h5")
    compile_hyp_path = os.path.join(sfmodel_dir, "__COMPILE_HYP__.pkl")
    fit_hyp_path = os.path.join(sfmodel_dir, "__FIT_HYP__.pkl")
    assert os.path.isfile(model_path)
    assert os.path.isfile(compile_hyp_path)
    assert os.path.isfile(fit_hyp_path)

    ret, out, err = _sp_run(
        f"python -u scripts/train_sfmodel.py --model_dir {sfmodel_dir} --out_dir {sfmodel_output_dir}"
    )
    if ret != 0:
        print(f"[W] train_sfmodel failed: our=`{out}`, err=`{err}`")
        pgrsu._sp_run(f"touch {sfmodel_output_dir}/__TRAIN_FAILED__.mark")

    with open(log_path, "w") as fp:
        fp.write(out)
        fp.write("\n=======================================\n")
        if ret != 0:
            fp.write("stderr:`\n")
            fp.write(err)
            fp.write("`\n")

    return ret


# Modified from https://github.com/dlfaults/deepcrime/blob/main/stats.py
def _is_diff_sts(orig_accuracy_list, accuracy_list, alpha, beta):
    def cohen_d(orig_accuracy_list, accuracy_list):
        nx = len(orig_accuracy_list)
        ny = len(accuracy_list)
        dof = nx + ny - 2
        pooled_std = np.sqrt(
            (
                (nx - 1) * np.std(orig_accuracy_list, ddof=1) ** 2
                + (ny - 1) * np.std(accuracy_list, ddof=1) ** 2
            )
            / dof
        )
        result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
        return float(result)

    def p_value_glm(orig_accuracy_list, accuracy_list):
        list_length = len(orig_accuracy_list)

        zeros_list = [0] * list_length
        ones_list = [1] * list_length
        mod_lists = zeros_list + ones_list
        acc_lists = orig_accuracy_list + accuracy_list

        data = {"Acc": acc_lists, "Mod": mod_lists}
        df = pd.DataFrame(data)

        response, predictors = dmatrices("Acc ~ Mod", df, return_type="dataframe")
        glm = sm.GLM(response, predictors)
        glm_results = glm.fit()
        glm_sum = glm_results.summary()
        pv = str(glm_sum.tables[1][2][4])
        p_value_g = float(pv)

        return p_value_g

    p_value: float = p_value_glm(orig_accuracy_list, accuracy_list)
    assert p_value >= 0 and p_value <= 1
    effect_size: float = cohen_d(orig_accuracy_list, accuracy_list)
    is_sts = (p_value < alpha) and abs(effect_size) >= beta

    return is_sts, p_value, effect_size


def _is_better_sts(bug_acc_list, fixed_acc_list, alpha, beta, bigger_better):
    """
    Return if fixed is statistically significantly better than bug
    """

    print("bug_acc_list:", bug_acc_list)
    print("fixed_acc_list:", fixed_acc_list)

    # if (bug_acc_list == fixed_acc_list):
    # print('=======================================')

    fixed_acc_mean = float(np.mean(fixed_acc_list))
    bug_acc_mean = float(np.mean(bug_acc_list))
    mean_is_better = (
        fixed_acc_mean > bug_acc_mean
        if bigger_better
        else fixed_acc_mean < bug_acc_mean
    )

    if not mean_is_better:
        return {
            "bigger_better": bigger_better,
            "bug_mean": bug_acc_mean,
            "fixed_mean": fixed_acc_mean,
            "p_value": None,
            "effect_size": None,
            "is_diff_sts": None,
            "is_better_sts": False,
        }

    try:
        is_diff_sts, p_value, effect_size = _is_diff_sts(
            bug_acc_list, fixed_acc_list, alpha, beta
        )
        is_better = mean_is_better and is_diff_sts

        return {
            "bigger_better": bigger_better,
            "bug_mean": bug_acc_mean,
            "fixed_mean": fixed_acc_mean,
            "p_value": p_value,
            "effect_size": effect_size,
            "is_diff_sts": is_diff_sts,
            "is_better_sts": is_better,
        }
    except statsmodels.tools.sm_exceptions.PerfectSeparationError:
        return {
            "bigger_better": bigger_better,
            "bug_mean": bug_acc_mean,
            "fixed_mean": fixed_acc_mean,
            "p_value": None,
            "effect_size": None,
            "is_diff_sts": "statsmodels.tools.sm_exceptions.PerfectSeparationError",
            "is_better_sts": "statsmodels.tools.sm_exceptions.PerfectSeparationError",
        }


def _proc_perTrain(output_dir):
    import keras

    compile_hyp_pkl = f"{output_dir}/__COMPILE_HYP__.pkl"
    eval_result_pkl = f"{output_dir}/__EVALUATE_RESULT__.pkl"
    print("loading", f"{output_dir}/__COMPILE_HYP__.pkl")
    with open(compile_hyp_pkl, "rb") as fp:
        compile_hyp = pickle.load(fp)
    print("loaded", f"{output_dir}/__COMPILE_HYP__.pkl")
    print("loading", f"{output_dir}/__EVALUATE_RESULT__.pkl")
    with open(eval_result_pkl, "rb") as fp:
        eval_result = pickle.load(fp)
    print("loaded", f"{output_dir}/__EVALUATE_RESULT__.pkl")

    metrics = compile_hyp["metrics"]
    if not metrics:  # None or []
        # print(eval_result)
        assert not isinstance(eval_result, list)
        eval_result = [eval_result, eval_result]
        eval_metrics = ["loss", "loss"]
    else:
        assert isinstance(eval_result, list)
        eval_result = eval_result
        eval_metrics = ["loss"] + metrics

    eval_score = eval_result[1]
    eval_metric = eval_metrics[1]

    # metrics=['accuracy']
    # metrics=['MSE']
    # metrics=['acc']
    # metrics=['categorical_accuracy']
    # metrics=['mse', 'mae', 'mape', 'accuracy']
    # metrics=['mse']
    # metrics=[]
    # metrics=[keras.metrics.Accuracy()]
    # metrics=[metrics.binary_accuracy]

    if not (  # Only for our benchmark models
        eval_metric
        in [
            "loss",
            "accuracy",
            "MSE",
            "acc",
            "categorical_accuracy",
            "mse",
            "mae",
            "mape",
            "binary_accuracy",
            "mean_squared_error",
            "mean_absolute_error",
            "binary_crossentropy",
        ]
        or eval_metric.__class__ == keras.metrics.Accuracy
        or eval_metric == keras.metrics.binary_accuracy
    ):
        print(eval_metric)
        raise ValueError(
            "eval_metric must be one of the following: "
            "'loss', 'accuracy', 'MSE', 'acc', 'categorical_accuracy', "
            "'mse', 'mae', 'mape', 'keras.metrics.Accuracy()', "
            "'binary_accuracy', 'keras.metrics.binary_accuracy', "
            "'mean_squared_error', 'mean_absolute_error', "
            "'binary_crossentropy'"
        )

    def _metric_is_bigger_better(metric):
        if metric in ["accuracy", "acc", "categorical_accuracy"]:
            return True
        elif metric.__class__ == keras.metrics.Accuracy:
            return True
        elif metric == keras.metrics.binary_accuracy or metric == "binary_accuracy":
            return True
        else:
            return False

    bigger_better = _metric_is_bigger_better(eval_metric)

    return float(eval_score), bigger_better


# bug_sfmodels = sorted(glob.glob(f"{bug_dir}/*.py"))
# fixed_sfmodels = sorted(glob.glob(f"{fixed_dir}/*.py"))
per_model_pacthes = sorted(glob.glob(f"{patches_root_dir}/*.py"))
pgrsu._ilog(f"Found {len(per_model_pacthes)} models")

correct_patches = []
for d in per_model_pacthes:
    model_name = os.path.basename(d)
    d = f"{d}/{patch_subdir}"
    assert os.path.isdir(d)

    weak_correct_patch_valid_result_jf = (  # not weak correct patch in paper
        f"{output_dir}/{model_name}/weak_correct_patch_valid_result.json"
    )
    strong_correct_patch_valid_result_jf = (  # not strong correct patch in paper
        f"{output_dir}/{model_name}/strong_correct_patch_valid_result.json"
    )
    if os.path.isfile(weak_correct_patch_valid_result_jf) and os.path.isfile(
        strong_correct_patch_valid_result_jf
    ):
        pgrsu._wlog(f"Found valid result, skip {model_name}")
        correct_patches.append(
            {
                "model_name": model_name,
                "weak_correct_patch": pgrsu._load_json(
                    weak_correct_patch_valid_result_jf
                ),
                "strong_correct_patch": pgrsu._load_json(
                    strong_correct_patch_valid_result_jf
                ),
            }
        )
        continue

    found_weak_correct_patch = False  # patch_is_better_sts == True
    found_strong_correct_patch = (
        False  # patch_is_better_sts and patch_is_better_than_fixed == True
    )
    weak_correct_patch = None
    strong_correct_patch = None

    model_lock = f"{output_dir}/{model_name}.lock"
    with pgrsu.FileLock(model_lock, retry=0) as flock:  # Allow multiple processes
        if flock.locked:
            # Train and stat all patches
            os.makedirs(f"{output_dir}/{model_name}", exist_ok=True)
            for patch in sorted(
                glob.glob(f"{d}/*.py.*"), key=lambda x: int(x.split(".")[-1])
            ):
                train_failed = False
                patch_name = os.path.basename(patch)
                assert model_name in patch_name
                print(f"[PATCH] Train {patch_name} ...")
                for i in tqdm.tqdm(range(1, N + 1)):
                    patch_train_output_dir = (
                        f"{output_dir}/{model_name}/{patch_name}/train{i}"
                    )
                    patch_train_log_path = f"{patch_train_output_dir}/train.log"
                    os.makedirs(patch_train_output_dir, exist_ok=True)
                    if os.path.exists(
                        f"{patch_train_output_dir}/__TRAIN_FAILED__.mark"
                    ):
                        print(f"[W] Fd __TRAIN_FAILED__.mark, skip {patch_name}@{i}")
                        train_failed = True
                        break
                    if os.path.exists(patch_train_log_path):
                        print(f"[W] Fd {patch_train_log_path}, skip {patch_name}@{i}")
                        continue
                    if 0 != train_sfmodel(
                        sfmodel_dir=patch,
                        sfmodel_output_dir=patch_train_output_dir,
                        log_path=patch_train_log_path,
                    ):
                        train_failed = True
                        break

                if train_failed:
                    print(f"[W] Train failed, skip this patch: {patch_name}")
                    continue

                del i
                del patch_train_output_dir
                del patch_train_log_path

                print(f"[PATCH] Stat {patch_name} ...")

                valid_result = {}
                bug_scores = []
                fixed_scores = []
                patch_scores = []
                bigger_better = None
                metric_changed = False

                for i in tqdm.tqdm(range(1, N + 1)):
                    # bug
                    bug_output_dir = (
                        f"{bug_fixed_train_result_dir}/{model_name}/bug/train{i}"
                    )
                    bug_score, _bigger_better = _proc_perTrain(bug_output_dir)

                    if bigger_better is None:
                        bigger_better = _bigger_better
                    else:
                        assert bigger_better == _bigger_better
                    del _bigger_better

                    bug_scores.append(bug_score)

                    # fixed
                    fixed_output_dir = (
                        f"{bug_fixed_train_result_dir}/{model_name}/fixed/train{i}"
                    )
                    fixed_score, _bigger_better = _proc_perTrain(fixed_output_dir)
                    assert bigger_better == _bigger_better
                    del _bigger_better

                    fixed_scores.append(fixed_score)

                    # pacth
                    patch_output_dir = (
                        f"{output_dir}/{model_name}/{patch_name}/train{i}"
                    )
                    patch_score, _bigger_better = _proc_perTrain(patch_output_dir)
                    if bigger_better != _bigger_better:
                        metric_changed = True
                        break
                    del _bigger_better

                    patch_scores.append(patch_score)

                if metric_changed:
                    continue

                try:
                    assert len(bug_scores) == len(patch_scores)
                    assert len(fixed_scores) == len(patch_scores)
                    assert len(patch_scores) == N

                    ibs_ret = _is_better_sts(
                        bug_acc_list=bug_scores,
                        fixed_acc_list=patch_scores,
                        alpha=alpha,
                        beta=beta,
                        bigger_better=bigger_better,
                    )

                    bug_mean_score = float(np.mean(bug_scores))
                    fixed_mean_score = float(np.mean(fixed_scores))
                    patch_mean_score = float(np.mean(patch_scores))

                    valid_result["patch_name"] = patch_name
                    valid_result["bug"] = bug_scores
                    valid_result["fixed"] = fixed_scores
                    valid_result["patch"] = patch_scores
                    valid_result["bug_mean"] = bug_mean_score
                    valid_result["fixed_mean"] = fixed_mean_score
                    valid_result["patch_mean"] = patch_mean_score
                    valid_result["bug_patch_stat"] = {
                        "bigger_better": ibs_ret["bigger_better"],
                        "p_value": ibs_ret["p_value"],
                        "effect_size": ibs_ret["effect_size"],
                        "is_diff_sts": ibs_ret["is_diff_sts"],
                        "patch_is_better_sts": ibs_ret["is_better_sts"],
                        "patch_is_better_than_fixed": (
                            patch_mean_score >= fixed_mean_score
                            if ibs_ret["bigger_better"]
                            else patch_mean_score <= fixed_mean_score
                        ),
                    }

                    # Save valid_result.json
                    valid_result_jf = (
                        f"{output_dir}/{model_name}/{patch_name}/valid_result.json"
                    )
                    pgrsu._save_as_json(valid_result, valid_result_jf)

                    # Early stop
                    if valid_result["bug_patch_stat"]["patch_is_better_sts"]:
                        pgrsu._ilog(f"Found weak correct patch: {patch_name}")
                        found_weak_correct_patch = True
                        weak_correct_patch = valid_result

                    if (
                        valid_result["bug_patch_stat"]["patch_is_better_sts"]
                        and valid_result["bug_patch_stat"]["patch_is_better_than_fixed"]
                    ):
                        pgrsu._ilog(f"Found strong correct patch: {patch_name}")
                        found_strong_correct_patch = True
                        strong_correct_patch = valid_result

                    if found_weak_correct_patch and found_strong_correct_patch:
                        break

                except Exception as e:
                    pgrsu._wlog(f"Failed to process {patch_name}: {e}")
                    raise e

            # Save weak&strong correct patches
            pgrsu._save_as_json(
                weak_correct_patch,
                filename=weak_correct_patch_valid_result_jf,
            )
            pgrsu._save_as_json(
                strong_correct_patch,
                filename=strong_correct_patch_valid_result_jf,
            )
            correct_patches.append(
                {
                    "model_name": model_name,
                    "weak_correct_patch": weak_correct_patch,
                    "strong_correct_patch": strong_correct_patch,
                }
            )

            del model_name
            del weak_correct_patch
            del strong_correct_patch


def _is_fixed_patch(patch_scores, fixed_scores, bigger_better):
    patch_mean = float(np.mean(patch_scores))
    fixed_mean = float(np.mean(fixed_scores))
    patch_mean_is_better = (
        patch_mean >= fixed_mean if bigger_better else patch_mean <= fixed_mean
    )

    if patch_mean_is_better:
        # pgrsu._wlog(f"Mean is better, {patch_mean} >= {fixed_mean}")
        return True

    try:
        is_diff_sts, p_value, effect_size = _is_diff_sts(
            fixed_scores, patch_scores, alpha, beta
        )
        # if not is_diff_sts:
        #     pgrsu._wlog(f"PATCH,FIXED {patch_mean} ~= {fixed_mean}")
        return not is_diff_sts
    except AssertionError:
        return False
    except statsmodels.tools.sm_exceptions.PerfectSeparationError:
        return False


count_weak1st_correct_patch = 0
for d in tqdm.tqdm(per_model_pacthes, desc="Finding Weak Correct Patches"):
    if ".0." in d:
        continue
    model_name = os.path.basename(d)
    found_weak1st_correct_patch = False
    weak1st_correct_patch = None
    weak1st_correct_patch_valid_result_jf = (
        f"{output_dir}/{model_name}/__weak1st_correct_patch_valid_result.json"
    )
    for patch in sorted(
        glob.glob(f"{d}/{patch_subdir}/*.py.*"), key=lambda x: int(x.split(".")[-1])
    ):
        patch_name = os.path.basename(patch)
        assert model_name in patch_name
        patch_output_dir = f"{output_dir}/{model_name}/{patch_name}"
        valid_result_jf = f"{patch_output_dir}/valid_result.json"

        if not os.path.isfile(valid_result_jf):
            # pgrsu._wlog(f"Not found valid result file: {valid_result_jf}")
            continue  # continue, not break

        valid_result = pgrsu._load_json(valid_result_jf)
        if valid_result["bug_patch_stat"]["patch_is_better_sts"]:
            found_weak1st_correct_patch = True
            weak1st_correct_patch = valid_result
            count_weak1st_correct_patch += 1
            break
    pgrsu._save_as_json(
        weak1st_correct_patch,
        filename=weak1st_correct_patch_valid_result_jf,
    )


count_strong2_correct_patch = 0
for d in tqdm.tqdm(per_model_pacthes, desc="Finding Strong Correct Patches"):
    if ".0." in d:
        continue
    model_name = os.path.basename(d)
    found_strong2_correct_patch = False
    strong2_correct_patch = None
    strong2_correct_patch_valid_result_jf = (
        f"{output_dir}/{model_name}/__strong2_correct_patch_valid_result.json"
    )
    # strong_correct_patch_valid_result_jf = (
    #     f"{output_dir}/{model_name}/strong_correct_patch_valid_result.json"
    # )
    # if os.path.exists(strong_correct_patch_valid_result_jf):
    #     j = pgrsu._load_json(strong_correct_patch_valid_result_jf)
    for patch in sorted(
        glob.glob(f"{d}/{patch_subdir}/*.py.*"), key=lambda x: int(x.split(".")[-1])
    ):
        patch_name = os.path.basename(patch)
        assert model_name in patch_name
        patch_output_dir = f"{output_dir}/{model_name}/{patch_name}"
        valid_result_jf = f"{patch_output_dir}/valid_result.json"

        if not os.path.isfile(valid_result_jf):
            # pgrsu._wlog(f"Not found valid result file: {valid_result_jf}")
            continue  # continue, not break

        valid_result = pgrsu._load_json(valid_result_jf)
        fixed_scores = valid_result["fixed"]
        patch_scores = valid_result["patch"]
        bigger_better = valid_result["bug_patch_stat"]["bigger_better"]

        is_strong2 = _is_fixed_patch(
            patch_scores=patch_scores,
            fixed_scores=fixed_scores,
            bigger_better=bigger_better,
        )

        if is_strong2:
            found_strong2_correct_patch = True
            strong2_correct_patch = valid_result
            count_strong2_correct_patch += 1
            # pgrsu._ilog(f"Found strong2 correct patch: {patch_name}")
            break
    # if j is not None and not strong2_correct_patch:
    #     pgrsu._wlog(f"assert failed: {model_name}")
    #     input("Any key to continue ---")
    pgrsu._save_as_json(
        strong2_correct_patch,
        filename=strong2_correct_patch_valid_result_jf,
    )
pgrsu._ilog(f"Found {count_weak1st_correct_patch} Weak Correct Patches")
pgrsu._ilog(f"Found {count_strong2_correct_patch} Strong Correct Patches")
