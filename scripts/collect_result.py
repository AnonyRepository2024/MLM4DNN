import os
import ast
import glob
import argparse
import _rs_utils as pgrsu

from repo2model_s import _std_keras_usage_style


def _semantic_equivalent(a: str, b: str) -> bool:
    """
    _std_keras_usage_style(ast.parse(a)) == _std_keras_usage_style(ast.parse(b))
    """

    a = a.replace("__root__.", "")
    b = b.replace("__root__.", "")

    if a.strip() == b.strip():
        return True

    try:
        a_ast = ast.parse(a)
        b_ast = ast.parse(b)
    except SyntaxError:
        return False

    a_ast.body = [
        stmt
        for stmt in a_ast.body
        if not isinstance(stmt, (ast.Import, ast.ImportFrom))
    ]
    b_ast.body = [
        stmt
        for stmt in b_ast.body
        if not isinstance(stmt, (ast.Import, ast.ImportFrom))
    ]

    std_a_ast = _std_keras_usage_style(a_ast)
    std_b_ast = _std_keras_usage_style(b_ast)
    if ast.unparse(std_a_ast) == ast.unparse(std_b_ast):
        return True

    # ...
    return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair_result_dir", type=str, required=True)
    parser.add_argument("--correct_models_dir", type=str, required=True)
    parser.add_argument("--patch_source_filename", type=str, required=True)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def main(args):
    repair_result_dir = os.path.abspath(args.repair_result_dir)
    correct_models_dir = os.path.abspath(args.correct_models_dir)
    patch_source_jfilename = args.patch_source_filename
    verbose = args.verbose
    assert os.path.isdir(repair_result_dir)
    assert os.path.isdir(correct_models_dir)
    assert patch_source_jfilename.endswith(".json")

    validate_result_dir = f"{repair_result_dir}/validate_results"
    models = sorted(glob.glob(f"{repair_result_dir}/*.py"))
    v_models = sorted(glob.glob(f"{validate_result_dir}/*.py"))

    if True:
        v_models = [v for v in v_models if ".0." not in os.path.basename(v)]

    if len(models) > len(v_models):
        pgrsu._wlog(f"Found {len(models)} but {len(v_models)} validated, ignoring some")
        models = [
            m
            for m in models
            if any(os.path.basename(m) == os.path.basename(v) for v in v_models)
        ]
    assert len(models) == len(v_models)

    pgrsu._ilog(f"Found {len(models)} models")

    weak1st_patch_jfs = sorted(
        f"{v}/__weak1st_correct_patch_valid_result.json" for v in v_models
    )
    weak1st_patch_jfs = list(filter(os.path.isfile, weak1st_patch_jfs))
    strong2_patch_jfs = sorted(
        f"{v}/__strong2_correct_patch_valid_result.json" for v in v_models
    )
    strong2_patch_jfs = list(filter(os.path.isfile, strong2_patch_jfs))
    assert len(weak1st_patch_jfs) == len(strong2_patch_jfs) == len(models)
    weak1st_patch_jsons = [pgrsu._load_json(j) for j in weak1st_patch_jfs]
    strong2_patch_jsons = [pgrsu._load_json(j) for j in strong2_patch_jfs]

    res_WRC = sum(1 if j is not None else 0 for j in weak1st_patch_jsons)
    res_SRC = sum(1 if j is not None else 0 for j in strong2_patch_jsons)

    ranks_WRC = [
        int(j["patch_name"].split(".")[-1]) + 1 if j else -1
        for j in weak1st_patch_jsons
    ]
    ranks_SRC = [
        int(j["patch_name"].split(".")[-1]) + 1 if j else -1
        for j in strong2_patch_jsons
    ]

    res_SMC = 0
    ranks_SMC = []
    emr_models = []
    for m in models:
        patch_jf = f"{m}/{patch_source_jfilename}"
        assert os.path.isfile(patch_jf)

        patch_sources = pgrsu._load_json(patch_jf)
        correct_source = pgrsu._load_txt(f"{correct_models_dir}/{os.path.basename(m)}")

        this_rank = -1
        for i, patch_source in enumerate(patch_sources, start=1):
            if isinstance(patch_source, list):
                assert len(patch_source) == 2
                patch_source = patch_source[0].replace("__mask_0__", patch_source[1])
            if _semantic_equivalent(patch_source, correct_source):
                this_rank = i
                res_SMC += 1
                break
        ranks_SMC.append(this_rank)
        if this_rank != -1:
            emr_models.append(os.path.basename(m))
    assert len(ranks_WRC) == len(ranks_SRC) == len(ranks_SMC) == len(models)

    if verbose:
        wcr_models = []
        for j in weak1st_patch_jsons:
            if j is not None:
                wcr_models.append(".".join(j["patch_name"].split(".")[:-1]))
        pgrsu._ilog(f"Weak Correct Repaired: {wcr_models}")

        scr_models = []
        for j in strong2_patch_jsons:
            if j is not None:
                scr_models.append(".".join(j["patch_name"].split(".")[:-1]))
        pgrsu._ilog(f"Strong Correct Repaired: {scr_models}")

        pgrsu._ilog(f"Exact Match Repaired: {emr_models}")

    ranks_WRC = [r for r in ranks_WRC if r != -1]
    ranks_SRC = [r for r in ranks_SRC if r != -1]
    ranks_SMC = [r for r in ranks_SMC if r != -1]
    pgrsu._ilog(f"Ranks of W : {ranks_WRC}")
    pgrsu._ilog(f"Ranks of S : {ranks_SRC}")
    pgrsu._ilog(f"Ranks of EM: {ranks_SMC}")
    assert len(ranks_WRC) == res_WRC
    assert len(ranks_SRC) == res_SRC
    assert len(ranks_SMC) == res_SMC

    mean_rank_WRC = sum(ranks_WRC) / len(ranks_WRC) if len(ranks_WRC) > 0 else "-"
    mean_rank_SRC = sum(ranks_SRC) / len(ranks_SRC) if len(ranks_SRC) > 0 else "-"
    mean_rank_SMC = sum(ranks_SMC) / len(ranks_SMC) if len(ranks_SMC) > 0 else "-"
    res_Rank = f"{mean_rank_WRC}/{mean_rank_SRC}/{mean_rank_SMC}"

    pgrsu._ilog(f"WRC\tSRC\tSMC\tRank")
    pgrsu._ilog(f"{res_WRC}\t{res_SRC}\t{res_SMC}\t{res_Rank}")


if __name__ == "__main__":
    main(get_args())
