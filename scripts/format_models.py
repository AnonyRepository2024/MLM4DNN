import os
import sys
import glob
import tqdm
import tempfile
import _rs_utils as pgrsu


if __name__ == "__main__":
    if not (4 == len(sys.argv)):
        print("format_models <in_dir> <out_dir> <repo2model_s_path>")
        sys.exit(-1)

    def _remove_func_def_with_mask(code: str):
        import ast

        toremove = []
        codeast = ast.parse(code)
        for stmt in codeast.body:
            if isinstance(stmt, ast.FunctionDef):
                if "__mask_0__" in ast.unparse(stmt):
                    toremove.append(stmt)
        for stmt in toremove:
            codeast.body.remove(stmt)
        codeast = ast.fix_missing_locations(codeast)
        return ast.unparse(codeast)

    in_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)
    repo2model_s_path = os.path.abspath(sys.argv[3])

    files = sorted(glob.glob(f"{in_dir}/*.py"))
    for f in tqdm.tqdm(files, desc="model -> formatted model"):
        basename = os.path.basename(f)
        out_f = f"{out_dir}/{basename}"
        with tempfile.NamedTemporaryFile("w", encoding="UTF-8") as tmp:
            filename = tmp.name
            with open(f, "r", encoding="UTF-8") as fp:
                code = fp.read().replace(
                    "__mask_0__ __mhint_", "__mask_0__S__E__P__mhint_"
                )
                tmp.write(code)
                tmp.flush()
            if 0 != pgrsu._sp_system(
                "R2MS_ASTT_KERAS_PROGRAM_1F_RETAIN_FIT_ALL_ARGS=1 R2MS_ASTT_KERAS_PROGRAM_1F_RETAIN_FITG_ALL_ARGS=1 "
                + f'python -u "{repo2model_s_path}" --no-output -p "{filename}" -o "{out_f}" --v6 --std-keras-usage --CONFIG_cared_ast_transformer keras_program_1f_main',
                logging=False,
            ):
                pgrsu._wlog(f"Failed to process {f}")
                continue
        with open(out_f, "r", encoding="UTF-8") as fp:
            code = fp.read().replace("__mask_0__S__E__P__mhint_", "__mask_0__ __mhint_")
            code = _remove_func_def_with_mask(code)
        with open(out_f, "w", encoding="UTF-8") as fp:
            fp.write(code)
