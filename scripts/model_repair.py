import os
import sys
import ast
import uuid
import tqdm
import time
import multiprocessing
import subprocess as sp
import data_utils as du
import element_mask_actions as actions
import _rs_utils as pgrsu
import infill_api as infill
import model_valid_api as mvalid
from typing import Tuple
from repo2model_s import _std_keras_usage_style, _set_parent


def _fix_sfmodel_code(code: str):
    import ast

    def _is_name_node(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return True
        elif isinstance(node, ast.Attribute):
            return _is_name_node(node.value)
        return False

    def _wrap_try(imports: list) -> list:
        return [f"try: {i}\nexcept ImportError: pass" for i in imports]

    codeast = ast.parse(code)
    _root = "__root__"
    imports = set()  # ['import ...', ...]
    for node in filter(_is_name_node, ast.walk(codeast)):
        if isinstance(node, ast.Attribute):
            parts = ast.unparse(node).strip().split(".")
            if not parts:
                continue
            if parts[0] != _root:
                continue
            if len(parts) == 1:
                continue
            parts = parts[1:-1]
            for i in range(len(parts)):
                imports.add(f"import {'.'.join(parts[:i + 1])}")

    code = code.strip()
    code = code.replace(f"{_root}.", "")
    code = "\n".join(_wrap_try(sorted(imports))) + "\n" + code
    return code


def _semantic_equivalent(a: str, b: str) -> bool:
    """
    _std_keras_usage_style(ast.parse(a)) == _std_keras_usage_style(ast.parse(b))
    """

    if a.strip() == b.strip():
        return True

    try:
        a_ast = ast.parse(a)
        b_ast = ast.parse(b)
    except SyntaxError:
        return False

    std_a_ast = _std_keras_usage_style(a_ast)
    std_b_ast = _std_keras_usage_style(b_ast)
    if ast.unparse(std_a_ast) == ast.unparse(std_b_ast):
        return True

    # ...
    return False


def _try_get_min_bad_change_check_context(
    model_with_mask: str, mask_pred: str
) -> ast.expr | ast.stmt:
    assert model_with_mask.count("__mask_0__") == 1
    ori_mask_line = next(
        filter(lambda x: "__mask_0__" in x, model_with_mask.splitlines())
    )
    ori_mask_line_ast = _set_parent(
        ast.fix_missing_locations(ast.parse(ori_mask_line.strip()).body[0])
    )

    def _is_cared_context(node: ast.expr | ast.stmt) -> bool:
        if not node:
            return False
        node_code = ast.unparse(node)
        if node_code.startswith("__root__.keras."):
            return True
        if ".compile(" in ast.unparse(node):
            return True
        if ".fit(" in ast.unparse(node):
            return True
        return False

    def impl(node) -> ast.expr | ast.stmt:
        if not node:
            return ori_mask_line_ast
        node_code = ast.unparse(node)
        if "__root__.keras." in node_code:
            return node
        if ".compile(" in node_code:
            return node
        if ".fit(" in node_code:
            return node
        return impl(node.parent)

    mask_0_node = next(
        filter(
            lambda x: isinstance(x, ast.Name) and x.id == "__mask_0__",
            ast.walk(ori_mask_line_ast),
        )
    )
    if _is_cared_context(ast.parse(mask_pred.strip())):
        ctx = ast.parse(mask_pred.strip())
        if isinstance(ctx, ast.Module):
            ctx = ctx.body[0]
        if isinstance(ctx, ast.Expr):
            ctx = ctx.value
        return ctx
    ctx = ast.fix_missing_locations(impl(mask_0_node))
    ctx = ast.parse(ast.unparse(ctx).replace("__mask_0__", mask_pred))
    if isinstance(ctx, ast.Module):
        ctx = ctx.body[0]
    if isinstance(ctx, ast.Expr):
        ctx = ctx.value
    return ctx


def _is_keras_layer(context: ast.expr | ast.stmt) -> bool:
    if not context:
        return False
    std_ctx = _std_keras_usage_style(context)
    return isinstance(std_ctx, ast.Call) and ast.unparse(std_ctx).startswith(
        "__root__.keras.layers."
    )


def _fix_list_to_tuple(node: ast.AST) -> ast.AST:
    class _FixListToTuple(ast.NodeTransformer):
        def visit_List(self, node: ast.List) -> ast.Tuple:
            return ast.Tuple(elts=node.elts, ctx=node.ctx)

    return _FixListToTuple().visit(node)


def _get_input_shape(layer: ast.Call) -> str | None:
    assert isinstance(layer, ast.Call)

    def _get_karg(kwargs, key) -> ast.AST | None:
        arg = kwargs.get(key, None)
        if arg is None:
            return None
        return _fix_list_to_tuple(arg)

    def _layer(kwargs) -> str | None:
        input_dim = _get_karg(kwargs, "input_dim")
        input_shape = _get_karg(kwargs, "input_shape")

        if input_dim is not None:
            return ast.unparse(ast.Tuple(elts=[input_dim], ctx=ast.Load()))
        elif input_shape is not None:
            return ast.unparse(input_shape)
        else:
            return None

    def _Input_layer(kwargs) -> str | None:
        shape = _get_karg(kwargs, "shape")
        if shape is not None:
            return ast.unparse(shape)
        else:
            return None

    kwargs = {kw.arg: kw.value for kw in layer.keywords}
    layer_code = ast.unparse(layer.func).strip()

    if "__root__.keras.layers.Input" in layer_code:
        return _Input_layer(kwargs)

    if "__root__.keras.layers." in layer_code:
        return _layer(kwargs)

    raise ValueError(f"Unknown layer: {layer_code}")


def _get_output_shape(
    layer: ast.Call, layer_idx, layer_count, layer_lines
) -> str | None:
    def _get_arg(args, idx) -> ast.AST | None:
        if len(args) > idx:
            return args[idx]
        return None

    def _get_karg(kwargs, key) -> ast.AST | None:
        arg = kwargs.get(key, None)
        if arg is None:
            return None
        return _fix_list_to_tuple(arg)

    kwargs = {kw.arg: kw.value for kw in layer.keywords}
    args = layer.args
    layer_code = ast.unparse(layer.func).strip()

    if "__root__.keras.layers.Dense" in layer_code:
        units = _get_arg(args, 0) or _get_karg(kwargs, "units")
        return ast.unparse(units) if units is not None else None
    if "__root__.keras.layers.Activation" in layer_code and layer_idx > 0:
        prev_layer = ast.parse(layer_lines[layer_idx - 1].strip())
        prev_layer = list(filter(_is_keras_layer, ast.walk(prev_layer)))
        if len(prev_layer) == 1:
            prev_layer = prev_layer[0]
            return _get_output_shape(
                prev_layer,
                layer_idx - 1,
                layer_count,
                layer_lines,
            )

    raise ValueError(f"Unprocessed layer: {layer_code}")


def _input_shape_change_is_bad(
    src_context: ast.Call, trg_context: ast.Call, layer_idx: int, layer_count: int
) -> bool:
    CANNOT_CONFIRM = False

    # print(">>>src_context:", ast.unparse(src_context))
    # print(">>>trg_context:", ast.unparse(trg_context))

    try:
        src_input_shape: tuple[int] | None = _get_input_shape(src_context)
        trg_input_shape: tuple[int] | None = _get_input_shape(trg_context)
    except ValueError:
        return CANNOT_CONFIRM

    # print(">>>src_input_shape:", src_input_shape)
    # print(">>>trg_input_shape:", trg_input_shape)

    if layer_idx == 0:
        if src_input_shape is None:
            return True  # None -> * is bad
        if trg_input_shape is None:
            return True  # * -> None is bad
        return (  # bad change if the input shape changes
            src_input_shape != trg_input_shape
        )
    else:
        if src_input_shape is None and trg_input_shape is None:
            return False  # None -> None is ok
        elif src_input_shape is None and trg_input_shape is not None:
            return CANNOT_CONFIRM  # None -> Not None is not confirmed
        elif src_input_shape is not None and trg_input_shape is None:
            return False  # Not None -> None is ok
        else:
            return (  # bad change if the input shape changes
                src_input_shape != trg_input_shape
            )
    raise NotImplementedError("Should not reach here")


def _output_shape_change_is_bad(
    src_context: ast.Call,
    trg_context: ast.Call,
    layer_idx: int,
    layer_count: int,
    layer_lines,
) -> bool:
    CANNOT_CONFIRM = False
    # print(">>>Call _output_shape_change_is_bad")

    def _is_last_layer() -> bool:
        if layer_idx == layer_count - 1:
            return True
        if (
            layer_idx == layer_count - 2
            and "__root__.keras.layers.Activation" in layer_lines[-1]
        ):
            return True
        return False

    if _is_last_layer():
        try:
            src_output_shape: tuple[int] | None = _get_output_shape(
                src_context, layer_idx, layer_count, layer_lines
            )
            trg_output_shape: tuple[int] | None = _get_output_shape(
                trg_context, layer_idx, layer_count, layer_lines
            )
        except ValueError as ex:
            return CANNOT_CONFIRM
        # print(">>>src_output_shape:", src_output_shape)
        # print(">>>trg_output_shape:", trg_output_shape)
        if src_output_shape is not None and trg_output_shape is not None:
            return (  # bad change if the output shape changes
                src_output_shape != trg_output_shape
            )

    return False


def _get_masked_layer_idx(model_with_mask: str) -> tuple[int, int]:
    # 1. Sequential style
    def _try_sequential_style(model_with_mask):
        lines = map(lambda x: x.strip(), model_with_mask.splitlines())
        sequential_name = None
        layer_idx = None
        layer_count = 0
        layer_lines = []
        for i, line in enumerate(lines):
            if "__root__.keras.models.Sequential(" in line:
                if sequential_name:
                    raise ValueError("Multiple Sequential")
                sequential_name = line.split("=")[0].strip()
            if sequential_name is not None and line.startswith(
                f"{sequential_name}.add("
            ):
                layer_count += 1
                layer_lines.append(line)
                if "__mask_0__" in line:
                    layer_idx = layer_count - 1
        if layer_idx is not None and layer_count > 0:
            return layer_idx, layer_count, layer_lines
        raise ValueError("Not found")

    # 2. Functional style
    def _try_functional_style(model_with_mask):
        import re

        lines = map(lambda x: x.strip(), model_with_mask.splitlines())
        input_layer_name = None
        ouput_layer_name = None
        layer_idx = None
        layer_count = 0
        layer_lines = []
        for i, line in enumerate(lines):
            if "__root__.keras.layers.Input(" in line:
                if input_layer_name or layer_count != 0:
                    raise ValueError("Multiple Input")
                layer_count = 1
                layer_lines.append(line)
                input_layer_name = line.split("=")[0].strip()
            # {ouput_layer_name} = __root__.keras.layers.{layer}({args})({input_layer_name})
            if input_layer_name is not None and (
                match := re.match(
                    r"(.*) = __root__.keras.layers\.(.+)\((.*)\)\((.*)\)DUMMY",
                    line.strip() + "DUMMY",
                )
            ):
                l_output_layer_name = match.group(1)
                l_layer = match.group(2)
                l_args = match.group(3)
                l_input_layer_name = match.group(4)
                if l_input_layer_name == input_layer_name:
                    if layer_count != 1:
                        raise ValueError("Multiple Input")
                    layer_count = 2
                    layer_lines.append(line)
                    if "__mask_0__" in line:
                        layer_idx = 1
                    ouput_layer_name = l_output_layer_name
            # {ouput_layer_name} = __root__.keras.layers.{layer}({args})({output_layer_name})
            if ouput_layer_name is not None and (
                match := re.match(
                    r"(.*) = __root__.keras.layers\.(.+)\((.*)\)\((.*)\)DUMMY",
                    line.strip() + "DUMMY",
                )
            ):
                l_output_layer_name = match.group(1)
                l_layer = match.group(2)
                l_args = match.group(3)
                l_input_layer_name = match.group(4)
                if l_input_layer_name == ouput_layer_name:
                    layer_count += 1
                    layer_lines.append(line)
                    if "__mask_0__" in line:
                        layer_idx = layer_count - 1
                    ouput_layer_name = l_output_layer_name
        if layer_idx is not None and layer_count > 0:
            return layer_idx, layer_count, layer_lines
        raise ValueError("Not found")

    model_with_mask = ast.unparse(ast.parse(model_with_mask))  # format

    try:
        return _try_sequential_style(model_with_mask)
    except ValueError:
        pass

    try:
        return _try_functional_style(model_with_mask)
    except ValueError:
        pass

    return None, None, None


def _is_bad_change_impl(model_with_mask: str, mask_src: str, mask_trg: str) -> bool:
    assert "__mask_0__" in model_with_mask

    # print("----------------------------------------------")

    model_with_mask = ast.unparse(_std_keras_usage_style(ast.parse(model_with_mask)))
    mask_src = ast.unparse(_std_keras_usage_style(ast.parse(mask_src)))
    mask_trg = ast.unparse(_std_keras_usage_style(ast.parse(mask_trg)))
    src_context = _try_get_min_bad_change_check_context(model_with_mask, mask_src)
    trg_context = _try_get_min_bad_change_check_context(model_with_mask, mask_trg)

    if _is_keras_layer(src_context) and _is_keras_layer(trg_context):
        layer_idx, layer_count, layer_lines = _get_masked_layer_idx(model_with_mask)
        # print(">>>layer_idx:", layer_idx)
        # print(">>>layer_count:", layer_count)
        if layer_idx is None or layer_count is None:
            return False  # Not confirmed
        # 1. Check the change of the input shape of the layer
        if _input_shape_change_is_bad(src_context, trg_context, layer_idx, layer_count):
            # print(">>>input_shape_change_is_bad")
            return True
        # 2. Check the change of the output shape of the layer if the layer is the last layer
        if _output_shape_change_is_bad(
            src_context, trg_context, layer_idx, layer_count, layer_lines
        ):
            # print(">>>output_shape_change_is_bad")
            return True

    return False  # Not confirmed


def _is_bad_change(model_with_mask: str, mask_src: str, mask_trg: str) -> bool:
    try:
        return _is_bad_change_impl(model_with_mask, mask_src, mask_trg)
    except Exception as e:
        pgrsu._wlog(f"Call _is_bad_change failed: {e}")
        return False


def train_sfmodels(
    sfmodel_dirs: list[str | None],
    train_sfmodel_path: str,
    env_name: str,
    out_root_dir: str,
) -> list[str]:  # (sfmodel_dir, ...)
    # Tool-Usage: python train_sfmodel.py --model_dir <model_dir> --out-dir <out_dir>
    out_root_dir = os.path.abspath(out_root_dir)
    os.makedirs(out_root_dir, exist_ok=True)
    trained_sfmodel_dirs = []
    for sfmodel_dir in sfmodel_dirs:
        if sfmodel_dir is None:
            continue
        assert os.path.isdir(sfmodel_dir)
        sfmodel_dir = os.path.abspath(sfmodel_dir)
        out_dir = os.path.join(out_root_dir, os.path.basename(sfmodel_dir))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "train_sfmodel.log"), "w") as log_store:
            proc = sp.Popen(
                " ".join(
                    [
                        "conda",
                        "run",
                        "-n",
                        env_name,
                        "python",
                        "-u",
                        "-W",
                        "ignore",
                        train_sfmodel_path,
                        "--model_dir",
                        sfmodel_dir,
                        "--out_dir",
                        out_dir,
                    ]
                ),
                shell=True,
                stdout=log_store,
                stderr=sp.PIPE,
            )
            ret = proc.wait()
        if ret != 0:
            trained_sfmodel_dirs.append(None)
            pgrsu._wlog(
                f"Train sfmodel failed: {sfmodel_dir}\n",
                "stderr:\n",
                proc.stderr.read().decode("UTF-8"),
            )
            continue
        assert os.path.exists(out_dir)
        assert os.path.exists(os.path.join(out_dir, "__MODEL__.h5"))
        assert os.path.exists(os.path.join(out_dir, "__COMPILE_HYP__.pkl"))
        assert os.path.exists(os.path.join(out_dir, "__FIT_HYP__.pkl"))
        assert os.path.exists(os.path.join(out_dir, "__SF_MODEL__.py"))
        trained_sfmodel_dirs.append(out_dir)

    return trained_sfmodel_dirs


# Generate masked buggy models by actions             |=> `masked buggy models`
def generate_masked_buggy_models(
    model_src: str, actions: list[actions.MaskAction]
) -> list[str, str]:  # (code with mask, mask ori)
    masked_buggy_models = []
    for a in actions:
        masked_buggy_models.extend(a.generate_masked_models(model_src))
    return masked_buggy_models


# Fill in the <mask> in the `masked buggy models`         |=> `possible repaired models`
def infill_masked_buggy_models(
    masked_buggy_models: list[str, str], infill_api: infill.InfillAPI, top_k: int
) -> list[str, str]:
    if IM4DNN_ENABLE_ONLY_MCTX:

        def proc_masked_buggy_model_1(args):
            c, m = args
            r2ms = __import__("repo2model_s")
            c_ast = ast.parse(c)
            mctx, _ = r2ms._split_model_data_context(c_ast)
            mctx = r2ms._remove_fn_head(mctx)
            return ast.unparse(mctx), m

    else:

        def proc_masked_buggy_model_1(args):
            return args

    if IM4DNN_ENABLE_MASKLINE:

        def proc_masked_buggy_model_2(args):
            c, m = args
            return du._trans_to_mask_line(c, m)

    else:

        def proc_masked_buggy_model_2(args):
            return args

    original_masked_buggy_models = masked_buggy_models.copy()
    masked_buggy_models = list(map(proc_masked_buggy_model_1, masked_buggy_models))
    masked_buggy_models = list(map(proc_masked_buggy_model_2, masked_buggy_models))
    masked_buggy_models = list(
        map(
            lambda x: (*x[0], x[1][0]),
            zip(masked_buggy_models, original_masked_buggy_models),
        )
    )

    if IM4DNN_ENABLE_MASKLINE:
        masked_buggy_models = list(
            map(
                lambda x: (x[0], x[1], du._trans_to_mask_line(x[2], "<|DUMMY|>")[0]),
                masked_buggy_models,
            )
        )

    possible_repaired_models = []
    for masked_buggy_model, _, _ in masked_buggy_models:
        infill_api.infill(masked_buggy_model, top_k)
    for masked_buggy_model, mask_pred_topk in zip(
        masked_buggy_models, infill_api.commit(), strict=True
    ):
        for mask_pred in mask_pred_topk:
            # NOTE: Add masked_buggy_model[2] instead of masked_buggy_model[0]
            possible_repaired_models.append((masked_buggy_model[2], mask_pred))
    return possible_repaired_models


# Filter the `possible repaired models` by static check |=> `filtered possible repaired models`
def filter_possible_repaired_models(
    possible_repaired_models: list[str, str],
    masked_buggy_models: list[str, str],
    original_buggy_model: str,
    enable_syntax_error_filter=True,
    enable_eq_filter=True,
    enable_api_usage_filter=True,
    enable_bad_change_filter=True,
) -> list[str]:
    formated_original_buggy_model = ast.unparse(ast.parse(original_buggy_model)).strip()
    filtered_possible_repaired_models = []
    assert len(possible_repaired_models) == len(masked_buggy_models)
    for i, (model_with_mask, mask_pred) in enumerate(possible_repaired_models):
        model = model_with_mask.replace("__mask_0__", mask_pred)
        # Skip the model with syntax error
        if enable_syntax_error_filter:
            try:
                model_ast = ast.parse(model)
            except SyntaxError:
                continue
        # Skip the original buggy model
        if enable_eq_filter:
            formated_model = ast.unparse(model_ast).strip()
            if formated_original_buggy_model == formated_model:
                continue
        # Skip the model that is semantically equivalent to the original buggy model
        if enable_api_usage_filter:
            mask_ori = masked_buggy_models[i][1]
            assert mask_ori != mask_pred
            if _semantic_equivalent(mask_ori, mask_pred):
                continue
        # Skip the model that can't be trained (some of
        if enable_bad_change_filter:
            mask_ori = masked_buggy_models[i][1]
            assert mask_ori != mask_pred
            if _is_bad_change(model_with_mask, mask_ori, mask_pred):
                continue

        filtered_possible_repaired_models.append(model)
    return filtered_possible_repaired_models


# Construct trainable sfmodel for each `filtered possible repaired models` |=> `trainable sfmodels`
## Reused by `buggy-models-to-sfmodel`
def construct_trainable_sfmodel(
    name: str,
    filtered_possible_repaired_models: list[str],
    train_work_dir: str,
    repo2model_path: str,
    out_dir: str,
    env_name: str,
) -> list[str]:
    train_work_dir = os.path.abspath(train_work_dir)
    out_dir = os.path.abspath(out_dir)
    assert os.path.exists(train_work_dir)
    assert os.path.exists(repo2model_path)
    # assert not os.path.exists(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    def make_sfmodel_dirname(i: int) -> str:
        return name + "." + str(i)

    trainable_sfmodel_dirs = []
    for i, repaired_model in enumerate(filtered_possible_repaired_models):
        try:
            # Temp file name for the trainable model, generated by mkstemp()
            _tmp_trainable_model_filename = str(uuid.uuid4()) + ".py"
            tmp_trainable_model_file = os.path.join(
                train_work_dir, _tmp_trainable_model_filename
            )
            tmp_sfmodel_dir = os.path.join(out_dir, _tmp_trainable_model_filename)
            del _tmp_trainable_model_filename  # Avoid misuse
            assert not os.path.exists(tmp_trainable_model_file)
            assert not os.path.exists(tmp_sfmodel_dir)
            repaired_model = _fix_sfmodel_code(repaired_model)
            with open(tmp_trainable_model_file, "w", encoding="UTF-8") as f:
                f.write(repaired_model)
            # Use repo2model to generate the sfmodel
            ret = pgrsu._sp_system(
                " ".join(
                    [
                        "R2M_EARLY_STOP=1",
                        "conda",
                        "run",
                        "-n",
                        env_name,
                        "python",
                        "-u",
                        "-W",
                        "ignore",
                        repo2model_path,
                        "--repo-path",
                        tmp_trainable_model_file,
                        "--t-repo-path",
                        os.path.dirname(tmp_trainable_model_file),
                        "--out-dir",
                        out_dir,
                        "--disable-ignore-exception",
                    ]
                ),
                logging=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            if ret != 0:
                trainable_sfmodel_dirs.append(None)
                continue
            assert os.path.exists(tmp_trainable_model_file)
            assert os.path.exists(os.path.join(tmp_sfmodel_dir, "__MODEL__.h5"))
            assert os.path.exists(os.path.join(tmp_sfmodel_dir, "__COMPILE_HYP__.pkl"))
            assert os.path.exists(os.path.join(tmp_sfmodel_dir, "__FIT_HYP__.pkl"))
            assert os.path.exists(os.path.join(tmp_sfmodel_dir, "__SF_MODEL__.py"))
            os.rename(  # Save the trainable model
                tmp_trainable_model_file,
                os.path.join(tmp_sfmodel_dir, "__TRAINABLE_MODEL__.py"),
            )
            target_sfmodel_dir = os.path.join(out_dir, make_sfmodel_dirname(i))
            if os.path.exists(target_sfmodel_dir):
                pgrsu._sp_system(f"rm -rf {target_sfmodel_dir}", logging=False)
                pgrsu._wlog(f"{target_sfmodel_dir} already exists, remove it")
            os.rename(tmp_sfmodel_dir, target_sfmodel_dir)
            trainable_sfmodel_dirs.append(target_sfmodel_dir)
        except Exception as e:
            if os.path.exists(tmp_trainable_model_file):
                os.remove(tmp_trainable_model_file)
            pgrsu._wlog(f"Construct trainable sfmodel failed ([{i}]): {e}\n{e}")
            trainable_sfmodel_dirs.append(None)
    return trainable_sfmodel_dirs


# Validate the `filtered possible repaired models`      |=> `plausible repaired models`  # Plausible patches
def validate_filtered_possible_repaired_models(
    filtered_possible_repaired_models: list[str],
    trainable_sfmodel_dirs: list[str],
    model_valid_api: mvalid.ModelValidAPI,
    out_root_dir: str,
    enable_early_stop: bool,
) -> Tuple[list[str], list]:
    assert len(filtered_possible_repaired_models) == len(trainable_sfmodel_dirs)
    plausible_repaired_models = []
    model_valid_stat = []  # [{sfmodel_dir, passed, time_cost}, ...]
    for r_model, sfmodel_dir in tqdm.tqdm(
        zip(filtered_possible_repaired_models, trainable_sfmodel_dirs, strict=True),
        total=len(filtered_possible_repaired_models),
    ):
        assert sfmodel_dir is not None
        out_dir = os.path.join(out_root_dir, os.path.basename(sfmodel_dir))
        passed, time_cost = model_valid_api.valid(sfmodel_dir, out_dir)
        if passed:
            plausible_repaired_models.append(r_model)
        # model_valid_stat.append((os.path.basename(sfmodel_dir), passed, time_cost))
        model_valid_stat.append(
            {
                "file": os.path.basename(sfmodel_dir),
                "passed": passed,
                "time_cost": time_cost,
            }
        )
        if enable_early_stop and passed:
            break
    return plausible_repaired_models, model_valid_stat


# Check the `plausible repaired models` by developer    |=> `correct repaired models`    # Correct patches
# ...


def print_stat_info(title, stat: list[tuple[str, int]], pre=None, suf=None):
    print(f"+======== {title} ========+")
    if pre is not None:
        if not isinstance(pre, (list, tuple)):
            pre = [pre]
        for p in pre:
            print(f"+>>> {p}")
    print(f"+>>> SUM: {sum([x[1] for x in stat])}")
    print(f"+>>> AVG: {sum([x[1] for x in stat]) / len(stat)}")
    print(f"+>>> MAX: {max([x[1] for x in stat])}")
    print(f"+>>> MID: {sorted([x[1] for x in stat])[len(stat) // 2]}")
    print(f"+>>> MIN: {min([x[1] for x in stat])}")
    if suf is not None:
        if not isinstance(suf, (list, tuple)):
            suf = [suf]
        for s in suf:
            print(f"+>>> {s}")
    print(f"+========={'=' * len(title)}=========+")


if __name__ == "__main__":
    import csv
    import glob
    import json
    import argparse

    doc = """\
Ops: 1, 2, 3, 4, 5, 6
    1: Generate masked buggy models by element masking actions (EMT)
    2: Fill in the <mask> in the masked buggy models (Element Restoration Model)
    3: Filter the possible repaired models by static check (PFT)
    4: Construct trainable sfmodel for each filtered possible repaired models (Prepare for PVT)
    5: Validate the filtered possible repaired models (PVT)
    6: Generate the stats of the results of the model repair
Cmd Args:
    1. `buggy-models-dir`: required by ALL
    2. `buggy-models-with-perfect-fl-dir`: required by wpfl
    3. `correct-models-dir`: required by stat-rank-*, 6
    4. `train-work-dir`: required by 4, 5
    5. `infill-api-name`: required by wpfl, 2
    6. `infill-api-config-file`: required by wpfl, 2
    7. `infill-top-k`: always 1
    8. `train-sfmodel-path`: required by 4, 5
    9. `repo2model-path`: required by 4
    10. `model-train-env-name`: required by 4
    11. `autotrainer-lib-path`: required by 5
    12. `autotrainer-env-name`: required by 5
    13. `out-dir`: required by ALL
    14. `ops`: required by ALL
    15. `overwrite`: optional, default False
    16. `disable-syntax-error-filter`: optional, default False
    17. `disable-eq-filter`: optional, default False
    18. `disable-api-usage-filter`: optional, default False
    19. `disable-bad-change-filter`: optional, default False
    20. `disable-early-stop`: optional, default False
    21. `model-valid-num-workers`: optional, default 1"""

    # fmt: off
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--buggy-models-dir", type=str, required=True)
    parser.add_argument("--buggy-models-with-perfect-fl-dir", type=str, required=True)
    parser.add_argument("--correct-models-dir", type=str, required=True)
    parser.add_argument("--train-work-dir", type=str, required=True)
    parser.add_argument("--infill-api-name", type=str, required=True)
    parser.add_argument("--infill-api-config-file", type=str, required=True)
    parser.add_argument("--infill-top-k", type=int, default=1)
    parser.add_argument("--repo2model-path", type=str, required=True)
    parser.add_argument("--model-train-env-name", type=str, required=True)
    parser.add_argument("--autotrainer-lib-path", type=str, required=True)
    parser.add_argument("--autotrainer-env-name", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--ops", type=str, nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)

    parser.add_argument("--disable-syntax-error-filter", action="store_true", default=False)
    parser.add_argument("--disable-eq-filter", action="store_true", default=False)
    parser.add_argument("--disable-api-usage-filter", action="store_true", default=False)
    parser.add_argument("--disable-bad-change-filter", action="store_true", default=False)

    parser.add_argument("--disable-early-stop", action="store_true", default=False)
    parser.add_argument("--model-valid-num-workers", type=int, default=1)

    args = parser.parse_args()
    pgrsu._plog("Cmd Args", vars(args))

    buggy_models_dir = os.path.abspath(args.buggy_models_dir)
    buggy_models_with_perfect_fl_dir = os.path.abspath(args.buggy_models_with_perfect_fl_dir)
    correct_models_dir = os.path.abspath(args.correct_models_dir)
    train_work_dir = os.path.abspath(args.train_work_dir)
    infill_api_name = args.infill_api_name
    infill_api_config_file = os.path.abspath(args.infill_api_config_file)
    infill_top_k = args.infill_top_k
    repo2model_path = os.path.abspath(args.repo2model_path)
    model_train_env_name = args.model_train_env_name
    autotrainer_lib_path = os.path.abspath(args.autotrainer_lib_path)
    autotrainer_env_name = args.autotrainer_env_name
    out_dir = os.path.abspath(args.out_dir)
    ops = args.ops
    overwrite = args.overwrite
    enable_syntax_error_filter = not args.disable_syntax_error_filter
    enable_eq_filter = not args.disable_eq_filter
    enable_api_usage_filter = not args.disable_api_usage_filter
    enable_bad_change_filter = not args.disable_bad_change_filter
    enable_early_stop = not args.disable_early_stop
    model_valid_num_workers = args.model_valid_num_workers
    assert os.path.isdir(buggy_models_dir)
    # assert os.path.isdir(buggy_models_with_perfect_fl_dir)
    # assert os.path.isdir(correct_models_dir)
    assert os.path.isdir(train_work_dir)
    assert os.path.exists(infill_api_config_file)
    assert infill_top_k == 1  # Always top 1
    assert os.path.exists(repo2model_path)
    assert os.path.exists(autotrainer_lib_path)
    # assert not os.path.exists(out_dir)  # DON'T CHECK
    # fmt: on

    # Temp flags for RQ3
    IM4DNN_ENABLE_MASKLINE = bool(eval(os.getenv("IM4DNN_ENABLE_MASKLINE", "False")))
    IM4DNN_ENABLE_ONLY_MCTX = bool(eval(os.getenv("IM4DNN_ENABLE_ONLY_MCTX", "False")))
    if IM4DNN_ENABLE_MASKLINE:
        pgrsu._wlog("IM4DNN_ENABLE_MASKLINE is enabled")
    if IM4DNN_ENABLE_ONLY_MCTX:
        pgrsu._wlog("IM4DNN_ENABLE_ONLY_MCTX is enabled")

    if overwrite:
        pgrsu._wlog(f"Type 'overwrite' to confirm (to remove {out_dir}): ", end="")
        if input().strip() != "overwrite":
            raise ValueError("Not confirmed to overwrite")
        pgrsu._wlog(f"Removing {out_dir}...")
        pgrsu._sp_system(f"rm -rf {out_dir}", logging=False)

    os.makedirs(out_dir, exist_ok=True)

    available_ops = [
        "wpfl",
        "1",
        "stat-1",
        "2",
        "stat-2",
        "stat-rank-2",
        "3",
        "stat-3",
        "stat-rank-3",
        "4",
        "stat-trainable-4",
        "5",
        "stat-5",
        "stat-rank-5",
        "6",
    ]

    if "stat-all" in ops:
        ops.remove("stat-all")
        ops.extend(filter(lambda x: x.startswith("stat-"), available_ops))

    if "end2end" in ops:
        ops.remove("end2end")
        ops.extend(
            [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
            ]
        )

    if not set(ops).issubset(set(available_ops)):
        raise ValueError(f"Invalid ops: {ops}")

    if not ops:
        raise ValueError(f"Empty ops")

    model_files = sorted(glob.glob(f"{buggy_models_dir}/*.py"))

    # Make out dir for each buggy model
    model_out_dirs = []
    for model_file in model_files:
        model_dirname = os.path.basename(model_file)
        model_dir = os.path.join(out_dir, model_dirname)
        os.makedirs(model_dir, exist_ok=True)
        model_out_dirs.append(model_dir)

    # wpfl. Model repair with perfect fault localization
    if "wpfl" in ops:
        print("Model repair with perfect fault localization...")
        with open(infill_api_config_file, "r", encoding="UTF-8") as f:
            infill_api_config = json.load(f)
        infill_api = infill.make_infill_api(infill_api_name, infill_api_config)
        infill_api.start()
        try:
            for model_file, m_out_dir in pgrsu._tcfor(
                tqdm.tqdm(zip(model_files, model_out_dirs), total=len(model_files)),
                tag_maker=lambda x: ("wpfl", x[0]),
            ):
                repaired_model_for_wpfl_jf = os.path.join(
                    m_out_dir, "repaired_model_for_wpfl.json"
                )
                if os.path.exists(repaired_model_for_wpfl_jf):
                    pgrsu._wlog(
                        f"Already repaired (with perfect fl), skip: {model_file}"
                    )
                    continue
                model_with_pfl_file = os.path.join(
                    buggy_models_with_perfect_fl_dir, os.path.basename(model_file)
                )
                with open(model_with_pfl_file, "r", encoding="UTF-8") as f:
                    model_src = f.read()
                repaired_model_for_wpfl = infill_masked_buggy_models(
                    [(model_src, "")], infill_api, top_k=1
                )
                assert len(repaired_model_for_wpfl) == 1
                with open(repaired_model_for_wpfl_jf, "w", encoding="UTF-8") as fp:
                    json.dump(repaired_model_for_wpfl, fp)
        finally:
            infill_api.stop()

    # 1. Generate masked buggy models by actions
    if "1" in ops:
        print("Generating masked buggy models by actions...")
        mask_actions = actions.get_all_actions()
        for model_file, m_out_dir in pgrsu._tcfor(
            tqdm.tqdm(zip(model_files, model_out_dirs), total=len(model_files)),
            tag_maker=lambda x: ("1", x[0]),
        ):
            masked_buggy_models_jf = os.path.join(m_out_dir, "masked_buggy_models.json")
            _1_time_cost_jf = os.path.join(m_out_dir, "_1_time_cost.json")
            if os.path.exists(masked_buggy_models_jf):
                pgrsu._wlog(f"Already gen masked, skip: {model_file}")
                continue
            with open(model_file, "r", encoding="UTF-8") as f:
                model_src = f.read()
            start_time = time.time()
            masked_buggy_models = generate_masked_buggy_models(model_src, mask_actions)
            time_cost = time.time() - start_time
            # assert len(masked_buggy_models) > 0
            with open(masked_buggy_models_jf, "w", encoding="UTF-8") as fp:
                json.dump(masked_buggy_models, fp)
            with open(_1_time_cost_jf, "w", encoding="UTF-8") as fp:
                json.dump({"time_cost": time_cost}, fp)

    # stat-1. Stat the num of `masked buggy models`
    if "stat-1" in ops or "1" in ops:
        print("Stat the num of `masked buggy models`...")
        masked_buggy_models_stat = []
        for model_file, m_out_dir in tqdm.tqdm(
            zip(model_files, model_out_dirs), total=len(model_files)
        ):
            with open(
                os.path.join(m_out_dir, "masked_buggy_models.json"),
                "r",
                encoding="UTF-8",
            ) as f:
                masked_buggy_models: list = json.load(f)
            masked_buggy_models_stat.append((model_file, len(masked_buggy_models)))
        masked_buggy_models_stat_jf = os.path.join(
            out_dir, "masked_buggy_models_stat.json"
        )
        with open(masked_buggy_models_stat_jf, "w", encoding="UTF-8") as fp:
            json.dump(masked_buggy_models_stat, fp)
        print_stat_info("Masked buggy models stat", masked_buggy_models_stat)

    # 2. Fill in the <mask> in the `masked buggy models` (fitmbm)
    if "2" in ops:
        print("Filling in the <mask> in the `masked buggy models`...")
        with open(infill_api_config_file, "r", encoding="UTF-8") as f:
            infill_api_config = json.load(f)
        infill_api = infill.make_infill_api(infill_api_name, infill_api_config)
        infill_api.start()
        try:
            for model_file, m_out_dir in pgrsu._tcfor(
                tqdm.tqdm(zip(model_files, model_out_dirs), total=len(model_files)),
                tag_maker=lambda x: ("2", x[0]),
            ):
                possible_repaired_models_jf = os.path.join(
                    m_out_dir, "possible_repaired_models.json"
                )
                _2_time_cost_jf = os.path.join(m_out_dir, "_2_time_cost.json")
                if os.path.exists(possible_repaired_models_jf):
                    pgrsu._wlog(f"Already infill, skip: {model_file}")
                    continue
                masked_buggy_models_jf = os.path.join(
                    m_out_dir, "masked_buggy_models.json"
                )
                with open(masked_buggy_models_jf, "r", encoding="UTF-8") as f:
                    masked_buggy_models = json.load(f)
                start_time = time.time()
                possible_repaired_models = infill_masked_buggy_models(
                    masked_buggy_models, infill_api, top_k=1
                )
                time_cost = time.time() - start_time
                assert len(possible_repaired_models) > 0
                with open(possible_repaired_models_jf, "w", encoding="UTF-8") as fp:
                    json.dump(possible_repaired_models, fp)
                with open(_2_time_cost_jf, "w", encoding="UTF-8") as fp:
                    json.dump({"time_cost": time_cost}, fp)
        finally:
            infill_api.stop()

    # stat-2. Stat the num of `possible repaired models`
    if "stat-2" in ops or "2" in ops:
        print("Stat the num of `possible repaired models`...")
        possible_repaired_models_stat = []
        for model_file, m_out_dir in tqdm.tqdm(
            zip(model_files, model_out_dirs), total=len(model_files)
        ):
            with open(
                os.path.join(m_out_dir, "possible_repaired_models.json"),
                "r",
                encoding="UTF-8",
            ) as f:
                possible_repaired_models: list = json.load(f)
            possible_repaired_models_stat.append(
                (model_file, len(possible_repaired_models))
            )
        possible_repaired_models_stat_jf = os.path.join(
            out_dir, "possible_repaired_models_stat.json"
        )
        with open(possible_repaired_models_stat_jf, "w", encoding="UTF-8") as fp:
            json.dump(possible_repaired_models_stat, fp)
        print_stat_info("Possible repaired models stat", possible_repaired_models_stat)

    # stat-rank-2. Stat the rank of the correct model in the `possible repaired models`
    if "stat-rank-2" in ops:
        print("Stat the rank of the correct model in the `possible repaired models`...")
        rank_stat = []
        for model_file, m_out_dir in tqdm.tqdm(
            zip(model_files, model_out_dirs), total=len(model_files)
        ):
            with open(
                os.path.join(m_out_dir, "possible_repaired_models.json"),
                "r",
                encoding="UTF-8",
            ) as f:
                possible_repaired_models: list = json.load(f)
            with open(
                os.path.join(correct_models_dir, os.path.basename(model_file)),
                "r",
                encoding="UTF-8",
            ) as f:
                correct_model = f.read()

            for i, (possible_repaired_model_with_mask, mask_pred) in enumerate(
                possible_repaired_models, start=1
            ):
                possible_repaired_model = possible_repaired_model_with_mask.replace(
                    "__mask_0__", mask_pred
                )
                if _semantic_equivalent(possible_repaired_model, correct_model):
                    rank_stat.append((model_file, i))
                    break
            else:
                rank_stat.append((model_file, -1))
        rank_stat_jf = os.path.join(out_dir, "possible_repaired_models_rank_stat.json")
        with open(rank_stat_jf, "w", encoding="UTF-8") as fp:
            json.dump(rank_stat, fp)
        no_crroect = sum([x[1] == -1 for x in rank_stat])
        rank_stat_inc_correct = [x for x in rank_stat if x[1] != -1]
        print_stat_info(
            "Rank stat (exclude no correct)",
            rank_stat_inc_correct,
            pre=[
                f"NO CORRECT: {no_crroect} / {len(rank_stat)} = {no_crroect / len(rank_stat):.3f}",
                f"INC CORRECT: {len(rank_stat_inc_correct)} / {len(rank_stat)} = {len(rank_stat_inc_correct) / len(rank_stat):.3f}",
            ],
        )

    # 3. Filter the `possible repaired models` by static check
    if "3" in ops:
        print("Filtering the `possible repaired models` by static check...")
        for model_file, m_out_dir in pgrsu._tcfor(
            tqdm.tqdm(zip(model_files, model_out_dirs), total=len(model_files)),
            tag_maker=lambda x: ("3", x[0]),
        ):
            filtered_possible_repaired_models_jf = os.path.join(
                m_out_dir, "filtered_possible_repaired_models.json"
            )
            _3_time_cost_jf = os.path.join(m_out_dir, "_3_time_cost.json")
            if os.path.exists(filtered_possible_repaired_models_jf):
                pgrsu._wlog(f"Already filtered, skip: {model_file}")
                continue
            with open(model_file, "r", encoding="UTF-8") as f:
                model_src = f.read()
            with open(
                os.path.join(m_out_dir, "masked_buggy_models.json"),
                "r",
                encoding="UTF-8",
            ) as f:
                masked_buggy_models = json.load(f)
            with open(
                os.path.join(m_out_dir, "possible_repaired_models.json"),
                "r",
                encoding="UTF-8",
            ) as f:
                possible_repaired_models = json.load(f)
            start_time = time.time()
            filtered_possible_repaired_models = filter_possible_repaired_models(
                possible_repaired_models,
                masked_buggy_models,
                model_src,
                enable_syntax_error_filter=enable_syntax_error_filter,
                enable_eq_filter=enable_eq_filter,
                enable_api_usage_filter=enable_api_usage_filter,
                enable_bad_change_filter=enable_bad_change_filter,
            )
            time_cost = time.time() - start_time
            assert len(filtered_possible_repaired_models) > 0
            with open(
                filtered_possible_repaired_models_jf, "w", encoding="UTF-8"
            ) as fp:
                json.dump(filtered_possible_repaired_models, fp)
            with open(_3_time_cost_jf, "w", encoding="UTF-8") as fp:
                json.dump({"time_cost": time_cost}, fp)

    # stat-3. Stat the num of `filtered possible repaired models`
    if "stat-3" in ops or "3" in ops:
        print("Stat the num of `filtered possible repaired models`...")
        filtered_possible_repaired_models_stat = []
        for model_file, m_out_dir in tqdm.tqdm(
            zip(model_files, model_out_dirs), total=len(model_files)
        ):
            with open(
                os.path.join(m_out_dir, "filtered_possible_repaired_models.json"),
                "r",
                encoding="UTF-8",
            ) as f:
                filtered_possible_repaired_models: list = json.load(f)
            filtered_possible_repaired_models_stat.append(
                (model_file, len(filtered_possible_repaired_models))
            )
        filtered_possible_repaired_models_stat_jf = os.path.join(
            out_dir, "filtered_possible_repaired_models_stat.json"
        )
        with open(
            filtered_possible_repaired_models_stat_jf, "w", encoding="UTF-8"
        ) as fp:
            json.dump(filtered_possible_repaired_models_stat, fp)
        print_stat_info(
            "Filtered possible repaired models stat",
            filtered_possible_repaired_models_stat,
        )

    # stat-rank-3. Stat the rank of the correct model in the `filtered possible repaired models`
    if "stat-rank-3" in ops:
        print(
            "Stat the rank of the correct model in the `filtered possible repaired models`..."
        )
        rank_stat = []  # [(model_file, rank)]
        for model_file, m_out_dir in tqdm.tqdm(
            zip(model_files, model_out_dirs), total=len(model_files)
        ):
            with open(
                os.path.join(m_out_dir, "filtered_possible_repaired_models.json"),
                "r",
                encoding="UTF-8",
            ) as f:
                filtered_possible_repaired_models: list = json.load(f)
            with open(
                os.path.join(correct_models_dir, os.path.basename(model_file)),
                "r",
                encoding="UTF-8",
            ) as f:
                correct_model = f.read()

            for i, filtered_possible_repaired_model in enumerate(
                filtered_possible_repaired_models,
                start=1,
            ):
                if _semantic_equivalent(
                    filtered_possible_repaired_model, correct_model
                ):
                    rank_stat.append((model_file, i))
                    break
            else:
                rank_stat.append((model_file, -1))
        rank_stat_jf = os.path.join(
            out_dir, "filtered_possible_repaired_models_rank_stat.json"
        )
        with open(rank_stat_jf, "w", encoding="UTF-8") as fp:
            json.dump(rank_stat, fp)
        no_crroect = sum([x[1] == -1 for x in rank_stat])
        rank_stat_inc_correct = [x for x in rank_stat if x[1] != -1]
        print_stat_info(
            "Rank stat (exclude no correct)",
            rank_stat_inc_correct,
            pre=[
                f"NO CORRECT: {no_crroect} / {len(rank_stat)} = {no_crroect / len(rank_stat):.3f}",
                f"INC CORRECT: {len(rank_stat_inc_correct)} / {len(rank_stat)} = {len(rank_stat_inc_correct) / len(rank_stat):.3f}",
            ],
        )

    # 4. Construct trainable sfmodel for each `filtered possible repaired models`
    if "4" in ops:
        print(
            "Constructing trainable sfmodel for each `filtered possible repaired models`..."
        )
        for model_file, m_out_dir in pgrsu._tcfor(
            tqdm.tqdm(zip(model_files, model_out_dirs), total=len(model_files)),
            tag_maker=lambda x: ("4", x[0]),
        ):
            trainable_sfmodel_dirs_jf = os.path.join(
                m_out_dir,
                "filtered_possible_repaired_models-trainable_sfmodel_dirs.json",
            )
            if os.path.exists(trainable_sfmodel_dirs_jf):
                pgrsu._wlog(f"Already constructed, skip: {model_file}")
                continue
            filtered_possible_repaired_models_jf = os.path.join(
                m_out_dir, "filtered_possible_repaired_models.json"
            )
            with open(filtered_possible_repaired_models_jf, "r", encoding="UTF-8") as f:
                filtered_possible_repaired_models = json.load(f)
            trainable_sfmodel_dirs = construct_trainable_sfmodel(
                os.path.basename(model_file),
                filtered_possible_repaired_models,
                train_work_dir,
                repo2model_path,
                os.path.join(
                    m_out_dir, "filtered_possible_repaired_models-trainable_sfmodels"
                ),
                model_train_env_name,
            )
            trainable_sfmodel_dirs = [
                os.path.relpath(sfmodel_dir, m_out_dir) if sfmodel_dir else None
                for sfmodel_dir in trainable_sfmodel_dirs
            ]
            assert len(trainable_sfmodel_dirs) > 0
            with open(trainable_sfmodel_dirs_jf, "w", encoding="UTF-8") as fp:
                json.dump(trainable_sfmodel_dirs, fp)

    # stat-trainable-4. Stat the num of `trainable sfmodel` for `filtered possible repaired models`
    if "stat-trainable-4" in ops:
        print(
            "Stat the num of `trainable sfmodel` for `filtered possible repaired models`..."
        )
        trainable_sfmodel_stat = []  # [(model_file, len(...), len(...))]
        for model_file, m_out_dir in tqdm.tqdm(
            zip(model_files, model_out_dirs), total=len(model_files)
        ):
            with open(
                os.path.join(m_out_dir, "filtered_possible_repaired_models.json"),
                "r",
                encoding="UTF-8",
            ) as f:
                filtered_possible_repaired_models: list = json.load(f)
            with open(
                os.path.join(
                    m_out_dir,
                    "filtered_possible_repaired_models-trainable_sfmodel_dirs.json",
                ),
                "r",
                encoding="UTF-8",
            ) as f:
                trainable_sfmodel_dirs: list = json.load(f)
            assert len(filtered_possible_repaired_models) == len(trainable_sfmodel_dirs)
            trainable_sfmodel_stat.append(
                (
                    model_file,
                    len(list(filter(lambda x: x is not None, trainable_sfmodel_dirs))),
                    len(trainable_sfmodel_dirs),
                )
            )
        trainable_sfmodel_stat_jf = os.path.join(
            out_dir, "filtered_possible_repaired_models-trainable_sfmodel_stat.json"
        )
        with open(trainable_sfmodel_stat_jf, "w", encoding="UTF-8") as fp:
            json.dump(trainable_sfmodel_stat, fp)
        print_stat_info(
            "Trainable sfmodel stat",
            trainable_sfmodel_stat,
            pre=[
                f"Trainable ratio: {sum([x[1] for x in trainable_sfmodel_stat])} / {sum([x[2] for x in trainable_sfmodel_stat])} = {sum([x[1] for x in trainable_sfmodel_stat]) / sum([x[2] for x in trainable_sfmodel_stat]):.3f}"
            ],
        )

    # 5. Validate the `filtered possible repaired models`
    if "5" in ops:
        print("Validating the `filtered possible repaired models`...")

        def task(args):
            workder_id, _model_files, _model_out_dirs, log_file = args
            with pgrsu.RedirectStdOutErrToFile(file_path=log_file, open_mode="a"):
                pgrsu._ilog(f"Worker {workder_id} start")
                pgrsu._ilog(f"==>> model_files: {_model_files}")
                pgrsu._ilog(f"==>> model_out_dirs: {_model_out_dirs}")
                model_valid_api = mvalid.AutoTrainerValidAPI(
                    conda_env_name=autotrainer_env_name,
                    autotrainer_lib_path=autotrainer_lib_path,
                )
                model_valid_api.start()
                try:
                    for model_file, m_out_dir in zip(
                        _model_files, _model_out_dirs
                    ):  # TODO: Add time stat
                        plausible_repaired_models_jf = os.path.join(
                            m_out_dir, "plausible_repaired_models.json"
                        )
                        model_valid_stat_jf = os.path.join(
                            m_out_dir, "at_model_valid_stat.json"
                        )
                        if os.path.exists(plausible_repaired_models_jf):
                            pgrsu._wlog(f"Already validated, skip: {model_file}")
                            continue
                        with pgrsu.FileLock(  # Allow multiple processes
                            os.path.join(m_out_dir, "5.lock"), retry=0
                        ) as flock:
                            if flock.locked:
                                pgrsu._ilog("Validating the model: " + model_file)
                                filtered_possible_repaired_models_jf = os.path.join(
                                    m_out_dir, "filtered_possible_repaired_models.json"
                                )
                                with open(
                                    filtered_possible_repaired_models_jf,
                                    "r",
                                    encoding="UTF-8",
                                ) as f:
                                    filtered_possible_repaired_models = json.load(f)
                                trainable_sfmodel_dirs_jf = os.path.join(
                                    m_out_dir,
                                    "filtered_possible_repaired_models-trainable_sfmodel_dirs.json",
                                )
                                with open(
                                    trainable_sfmodel_dirs_jf, "r", encoding="UTF-8"
                                ) as f:
                                    trainable_sfmodel_dirs = json.load(f)
                                trainable_sfmodel_dirs = [
                                    os.path.join(m_out_dir, sfmodel_dir)
                                    for sfmodel_dir in trainable_sfmodel_dirs
                                    if sfmodel_dir
                                ]
                                trainable_sfmodel_indexes = [
                                    int(sfmodel_dir.split(".")[-1])
                                    for sfmodel_dir in trainable_sfmodel_dirs
                                ]
                                assert len(trainable_sfmodel_indexes) <= len(
                                    filtered_possible_repaired_models
                                )
                                filtered_possible_repaired_models = [
                                    e
                                    for i, e in enumerate(
                                        filtered_possible_repaired_models
                                    )
                                    if i in trainable_sfmodel_indexes
                                ]
                                assert len(filtered_possible_repaired_models) == len(
                                    trainable_sfmodel_dirs
                                )
                                plausible_repaired_models, model_valid_stat = (
                                    validate_filtered_possible_repaired_models(
                                        filtered_possible_repaired_models,
                                        trainable_sfmodel_dirs,
                                        model_valid_api,
                                        os.path.join(
                                            m_out_dir,
                                            "filtered_possible_repaired_models-valid-at_output",
                                        ),
                                        enable_early_stop,
                                    )
                                )
                                # assert len(plausible_repaired_models) > 0  # DON'T ASSERT
                                with open(
                                    plausible_repaired_models_jf, "w", encoding="UTF-8"
                                ) as fp:
                                    json.dump(plausible_repaired_models, fp)
                                with open(
                                    model_valid_stat_jf, "w", encoding="UTF-8"
                                ) as fp:
                                    json.dump(model_valid_stat, fp)
                            else:
                                pgrsu._wlog(
                                    "Skip the model that is being validated by other proc: "
                                    + model_file
                                )
                except BaseException as ex:
                    import traceback

                    # Print full traceback with exception info
                    traceback.print_exc(file=sys.stdout)
                finally:
                    model_valid_api.stop()

        _unprocessed_model_files = []
        _unprocessed_model_out_dirs = []
        for model_file, m_out_dir in zip(
            model_files, model_out_dirs
        ):  # TODO: Add time stat
            plausible_repaired_models_jf = os.path.join(
                m_out_dir, "plausible_repaired_models.json"
            )
            if os.path.exists(plausible_repaired_models_jf):
                pgrsu._wlog(f"Already validated, skip: {model_file}")
                continue
            _unprocessed_model_files.append(model_file)
            _unprocessed_model_out_dirs.append(m_out_dir)
        pgrsu._ilog(
            f"Number of unprocessed model files: {len(_unprocessed_model_files)}"
        )
        pgrsu._ilog(
            f"Number of unprocessed model out dirs: {len(_unprocessed_model_out_dirs)}"
        )
        assert len(_unprocessed_model_files) == len(_unprocessed_model_out_dirs)

        if len(_unprocessed_model_files) == 0:
            pgrsu._wlog("All models are already validated, skip")
        elif model_valid_num_workers == 1:
            pgrsu._ilog("Processing with single worker...")
            log_file = os.path.join(out_dir, "at_model_valid_worker_SINGLE.log")
            if os.path.exists(log_file):
                log_file = log_file.replace(".log", f"_{time.time()}.log")
                assert not os.path.exists(log_file)
            task((0, _unprocessed_model_files, _unprocessed_model_out_dirs, log_file))
        else:
            num_workers = min(
                multiprocessing.cpu_count() // 2,
                24,
                len(_unprocessed_model_files),
                model_valid_num_workers,
            )
            worker_ids = list(range(num_workers))
            model_files_parts = list(
                pgrsu._split_list(_unprocessed_model_files, num_workers)
            )
            model_out_dirs_parts = list(
                pgrsu._split_list(_unprocessed_model_out_dirs, num_workers)
            )
            log_files = [
                os.path.join(out_dir, f"at_model_valid_worker_{i}.log")
                for i in worker_ids
            ]
            for i in range(len(log_files)):
                if os.path.exists(log_files[i]):
                    log_files[i] = os.path.join(
                        out_dir, f"at_model_valid_worker_{i}_{time.time()}.log"
                    )
            assert (
                num_workers
                == len(worker_ids)
                == len(model_files_parts)
                == len(model_out_dirs_parts)
                == len(log_files)
            )

            with multiprocessing.Pool(num_workers) as pool:
                pool.map(
                    task,
                    zip(
                        worker_ids,
                        model_files_parts,
                        model_out_dirs_parts,
                        log_files,
                    ),
                )

    # stat-5. Stat the num of `plausible repaired models` for each buggy model
    if "stat-5" in ops:
        print("Stat the num of `plausible repaired models` for each buggy model...")
        plausible_repaired_models_stat = []
        for model_file, m_out_dir in tqdm.tqdm(
            zip(model_files, model_out_dirs), total=len(model_files)
        ):
            plausible_repaired_models_jf = os.path.join(
                m_out_dir, "plausible_repaired_models.json"
            )
            if not os.path.exists(plausible_repaired_models_jf):
                pgrsu._wlog(f"{plausible_repaired_models_jf} not fount, skip")
                continue
            with open(
                plausible_repaired_models_jf,
                "r",
                encoding="UTF-8",
            ) as f:
                plausible_repaired_models: list = json.load(f)
            plausible_repaired_models_stat.append(
                (model_file, len(plausible_repaired_models))
            )
        plausible_repaired_models_stat_jf = os.path.join(
            out_dir, "plausible_repaired_models_stat.json"
        )
        with open(plausible_repaired_models_stat_jf, "w", encoding="UTF-8") as fp:
            json.dump(plausible_repaired_models_stat, fp)
        print_stat_info(
            "Plausible repaired models stat",
            plausible_repaired_models_stat,
        )

    # stat-rank-5. Stat the rank of the correct model in the `plausible repaired models`
    if "stat-rank-5" in ops:
        print(
            "Stat the rank of the correct model in the `plausible repaired models`..."
        )
        rank_stat = []
        for model_file, m_out_dir in tqdm.tqdm(
            zip(model_files, model_out_dirs), total=len(model_files)
        ):
            plausible_repaired_models_jf = os.path.join(
                m_out_dir, "plausible_repaired_models.json"
            )
            if not os.path.exists(plausible_repaired_models_jf):
                pgrsu._wlog(f"{plausible_repaired_models_jf} not fount, skip")
                continue  # Skip the model that is not validated
            with open(
                plausible_repaired_models_jf,
                "r",
                encoding="UTF-8",
            ) as f:
                plausible_repaired_models: list = json.load(f)
            with open(
                os.path.join(correct_models_dir, os.path.basename(model_file)),
                "r",
                encoding="UTF-8",
            ) as f:
                correct_model = f.read()

            for i, plausible_repaired_model in enumerate(
                plausible_repaired_models, start=1
            ):
                if _semantic_equivalent(plausible_repaired_model, correct_model):
                    rank_stat.append((model_file, i))
                    break
            else:
                rank_stat.append((model_file, -1))
        rank_stat_jf = os.path.join(out_dir, "plausible_repaired_models_rank_stat.json")
        with open(rank_stat_jf, "w", encoding="UTF-8") as fp:
            json.dump(rank_stat, fp)
        no_crroect = sum([x[1] == -1 for x in rank_stat])
        rank_stat_inc_correct = [x for x in rank_stat if x[1] != -1]
        print_stat_info(
            "Rank stat (exclude no correct)",
            rank_stat_inc_correct,
            pre=[
                f"NO CORRECT: {no_crroect} / {len(rank_stat)} = {no_crroect / len(rank_stat):.3f}",
                f"INC CORRECT: {len(rank_stat_inc_correct)} / {len(rank_stat)} = {len(rank_stat_inc_correct) / len(rank_stat):.3f}",
            ],
        )

    # 6. Check the `plausible repaired models` by developer (...gen summary...)
    if "6" in ops:
        print("Generating model repair summary...")
        plausible_repaired_models_details_csv = os.path.join(
            out_dir, "plausible_repaired_models_details.csv"
        )
        plausible_repaired_models_details_html_report_dir = os.path.join(
            out_dir, "plausible_repaired_models_details_html_report"
        )
        os.makedirs(plausible_repaired_models_details_html_report_dir, exist_ok=True)
        prms_index_html = os.path.join(
            plausible_repaired_models_details_html_report_dir, "index.html"
        )
        try:  # Gen details ...
            csv_fp = open(plausible_repaired_models_details_csv, "w", encoding="UTF-8")
            csv_writer = csv.writer(csv_fp)
            index_html_fp = open(prms_index_html, "w", encoding="UTF-8")

            # Generate table header
            ## CSV
            csv_writer.writerow(
                ["File", "Pred", "Tgt", "Is Correct (_semantic_equivalent)"]
            )
            index_html_fp.write("<html>")
            index_html_fp.write(
                "<head><style>table, th, td {border: 1px solid black;}</style></head>"
            )
            ## HTML
            index_html_fp.write("<body><table>")
            index_html_fp.write("<tr>")
            index_html_fp.write("<th>File</th>")
            index_html_fp.write("<th>Pred</th>")
            index_html_fp.write("<th>Tgt</th>")
            index_html_fp.write("<th>acc_e</th>")
            index_html_fp.write("</tr>")
            # Generate table content
            for model_file, m_out_dir in tqdm.tqdm(
                zip(model_files, model_out_dirs),
                total=len(model_files),
                desc="Gen details",
            ):
                with open(model_file, "r", encoding="UTF-8") as f:
                    model_src = f.read()
                with open(
                    os.path.join(correct_models_dir, os.path.basename(model_file)),
                    "r",
                    encoding="UTF-8",
                ) as f:
                    model_tgt = f.read()
                plausible_repaired_models_jf = os.path.join(
                    m_out_dir, "plausible_repaired_models.json"
                )
                if not os.path.exists(plausible_repaired_models_jf):
                    pgrsu._wlog(f"{plausible_repaired_models_jf} not found, skip")
                    continue  # Skip the model that is not validated
                with open(plausible_repaired_models_jf, "r", encoding="UTF-8") as f:
                    plausible_repaired_models = json.load(f)
                # Diff the plausible repaired models with the original buggy model
                bug2correct_diff_html_fn = (
                    f"{os.path.basename(model_file)}-bug-correct.html"
                )
                with open(
                    os.path.join(
                        plausible_repaired_models_details_html_report_dir,
                        bug2correct_diff_html_fn,
                    ),
                    "w",
                    encoding="UTF-8",
                ) as fp:
                    fp.write(
                        pgrsu._make_diff_html(
                            model_src,
                            model_tgt,
                            src_title="bug",
                            trg_title="correct",
                        )
                    )
                # Diff the plausible repaired models with the original buggy model
                for i, plausible_repaired_model in enumerate(plausible_repaired_models):
                    # Gen CSV
                    csv_writer.writerow(
                        [
                            model_file,
                            pgrsu._make_diff(model_src, plausible_repaired_model),
                            pgrsu._make_diff(model_src, model_tgt),
                            int(
                                _semantic_equivalent(
                                    plausible_repaired_model, model_tgt
                                )
                            ),
                        ]
                    )
                    # Gen HTML
                    bug2fix_diff_html_fn = (
                        f"{os.path.basename(model_file)}-bug-fix-{i}.html"
                    )
                    with open(
                        os.path.join(
                            plausible_repaired_models_details_html_report_dir,
                            bug2fix_diff_html_fn,
                        ),
                        "w",
                        encoding="UTF-8",
                    ) as fp:
                        fp.write(
                            pgrsu._make_diff_html(
                                model_src,
                                plausible_repaired_model,
                                src_title="bug",
                                trg_title="fix",
                            )
                        )
                    index_html_fp.write("<tr>")
                    index_html_fp.write(
                        f"<td><b>{os.path.basename(model_file)}</b></td>"
                    )
                    index_html_fp.write(
                        f'<td><a href="{bug2fix_diff_html_fn}">bug-fix-{i}</a></td>'
                    )
                    index_html_fp.write(
                        f'<td><a href="{bug2correct_diff_html_fn}">bug-correct</a></td>'
                    )
                    index_html_fp.write(
                        f"<td>{int(_semantic_equivalent(plausible_repaired_model, model_tgt))}</td>"
                    )
                    index_html_fp.write("</tr>")
            index_html_fp.write("</table></body></html>")
        finally:
            csv_fp.close()
            index_html_fp.close()
        # Gen summary ...
        model_repair_summary = {}
        for model_file, m_out_dir in tqdm.tqdm(
            zip(model_files, model_out_dirs), total=len(model_files), desc="Gen summary"
        ):
            correct_model_file = os.path.join(
                correct_models_dir, os.path.basename(model_file)
            )
            masked_buggy_models_jf = os.path.join(m_out_dir, "masked_buggy_models.json")
            infilled_buggy_models_jf = os.path.join(
                m_out_dir, "possible_repaired_models.json"
            )
            f_possible_repaired_models_jf = os.path.join(
                m_out_dir, "filtered_possible_repaired_models.json"
            )
            plausible_repaired_models_jf = os.path.join(
                m_out_dir, "plausible_repaired_models.json"
            )
            _1_time_cost_jf = os.path.join(m_out_dir, "_1_time_cost.json")
            _2_time_cost_jf = os.path.join(m_out_dir, "_2_time_cost.json")
            _3_time_cost_jf = os.path.join(m_out_dir, "_3_time_cost.json")
            model_valid_stat_jf = os.path.join(m_out_dir, "at_model_valid_stat.json")
            try:
                with open(correct_model_file, "r", encoding="UTF-8") as f:
                    correct_model = f.read()
                with open(masked_buggy_models_jf, "r", encoding="UTF-8") as f:
                    masked_buggy_models = json.load(f)
                with open(infilled_buggy_models_jf, "r", encoding="UTF-8") as f:
                    infilled_buggy_models = json.load(f)
                with open(f_possible_repaired_models_jf, "r", encoding="UTF-8") as f:
                    f_possible_repaired_models = json.load(f)
                with open(plausible_repaired_models_jf, "r", encoding="UTF-8") as f:
                    plausible_repaired_models = json.load(f)
                with open(_1_time_cost_jf, "r", encoding="UTF-8") as f:
                    _1_time_cost = json.load(f)["time_cost"]
                with open(_2_time_cost_jf, "r", encoding="UTF-8") as f:
                    _2_time_cost = json.load(f)["time_cost"]
                with open(_3_time_cost_jf, "r", encoding="UTF-8") as f:
                    _3_time_cost = json.load(f)["time_cost"]
                with open(model_valid_stat_jf, "r", encoding="UTF-8") as f:
                    model_valid_stat = json.load(f)
                model_repair_summary[model_file] = {
                    "count_of_masked": len(masked_buggy_models),
                    "count_of_infilled": len(infilled_buggy_models),
                    "correct_in_infilled": [
                        _semantic_equivalent(
                            x.replace("__mask_0__", mask), correct_model
                        )
                        for x, mask in infilled_buggy_models
                    ],
                    "count_of_filtered": len(f_possible_repaired_models),
                    "correct_in_filtered": [
                        _semantic_equivalent(x, correct_model)
                        for x in f_possible_repaired_models
                    ],
                    "count_of_plausible": len(plausible_repaired_models),
                    "correct_in_plausible": [
                        _semantic_equivalent(x, correct_model)
                        for x in plausible_repaired_models
                    ],
                    "plausible_in_filtered": [
                        any(
                            _semantic_equivalent(x.replace("__mask_0__", mask), p)
                            for p in plausible_repaired_models
                        )
                        for x, mask in f_possible_repaired_models,
                    ],
                    "1_time_cost": _1_time_cost,
                    "2_time_cost": _2_time_cost,
                    "3_time_cost": _3_time_cost,
                    "model_valid_stat": model_valid_stat,
                }
            except FileNotFoundError as ex:
                model_repair_summary[model_file] = {
                    "count_of_masked": -1,
                    "count_of_infilled": -1,
                    "correct_in_infilled": [],
                    "count_of_filtered": -1,
                    "correct_in_filtered": [],
                    "count_of_plausible": -1,
                    "correct_in_plausible": [],
                    "plausible_in_filtered": [],
                    "1_time_cost": -1,
                    "2_time_cost": -1,
                    "3_time_cost": -1,
                    "model_valid_stat": [],
                }
                pgrsu._wlog(f"File not found: {ex.filename}")
        model_repair_summary_jf = os.path.join(out_dir, "model_repair_summary.json")
        model_repair_summary_csv = os.path.join(out_dir, "model_repair_summary.csv")
        with open(model_repair_summary_jf, "w", encoding="UTF-8") as fp:
            json.dump(model_repair_summary, fp)
        with open(model_repair_summary_csv, "w", encoding="UTF-8") as fp:
            csv_writer = csv.writer(fp)
            csv_writer.writerow(
                [
                    "Idx",
                    "File",
                    "Count of masked",
                    "Count of infilled",
                    "Count of correct (infilled)",
                    "Rank of correct (infilled)",
                    "Count of filtered",
                    "Count of correct (filtered)",
                    "Rank of correct (filtered)",
                    "Count of plausible",
                    "Count of correct (plausible)",
                    "Rank of correct (plausible)",
                    "Repaired (plausible)",
                    "Repaired (correct_in_infilled)",
                    "Repaired (correct_in_filtered)",
                    "Repaired (correct_in_plausible)",
                    "Rank of plausible (filtered)",
                    "Time cost (1)",
                    "Time cost (2)",
                    "Time cost (3)",
                    "Time cost (1+2+3: patch generate)",
                    "Time cost (model valid(sum))",
                    "Time cost (model valid(until first plausible))",
                ]
            )
            for idx, (model_file, summary) in enumerate(model_repair_summary.items()):
                model_valid_stat = summary["model_valid_stat"]
                time_cost_of_model_valid = sum(
                    [x["time_cost"] for x in model_valid_stat]
                )
                time_cost_of_model_valid_until_first_plausible = 0
                for s in model_valid_stat:
                    time_cost_of_model_valid_until_first_plausible += s["time_cost"]
                    if s["passed"]:
                        break

                csv_writer.writerow(
                    [
                        idx,  # Idx
                        os.path.basename(model_file),  # File
                        summary["count_of_masked"],  # Count of masked
                        summary["count_of_infilled"],  # Count of infilled
                        summary["correct_in_infilled"].count(
                            True
                        ),  # Count of correct (infilled)
                        (  # Rank of correct (infilled)
                            summary["correct_in_infilled"].index(True) + 1
                            if True in summary["correct_in_infilled"]
                            else -1
                        ),
                        summary["count_of_filtered"],  # Count of filtered
                        summary["correct_in_filtered"].count(
                            True
                        ),  # Count of correct (filtered)
                        (  # Rank of correct (filtered)
                            summary["correct_in_filtered"].index(True) + 1
                            if True in summary["correct_in_filtered"]
                            else -1
                        ),
                        summary["count_of_plausible"],  # Count of plausible
                        summary["correct_in_plausible"].count(True),
                        (  # Rank of correct (plausible)
                            summary["correct_in_plausible"].index(True) + 1
                            if True in summary["correct_in_plausible"]
                            else -1
                        ),
                        int(summary["count_of_plausible"] > 0),  # Repaired (plausible)
                        int(
                            True in summary["correct_in_infilled"]
                        ),  # Repaired (correct_in_infilled)
                        int(
                            True in summary["correct_in_filtered"]
                        ),  # Repaired (correct_in_filtered)
                        int(
                            True in summary["correct_in_plausible"]
                        ),  # Repaired (correct_in_plausible)
                        int(
                            summary["plausible_in_filtered"].index(True) + 1
                            if True in summary["plausible_in_filtered"]
                            else -1
                        ),  # Rank of plausible (filtered)
                        summary["1_time_cost"],  # Time cost (1)
                        summary["2_time_cost"],  # Time cost (2)
                        summary["3_time_cost"],  # Time cost (3)
                        summary["1_time_cost"]
                        + summary["2_time_cost"]
                        + summary["3_time_cost"],  # Time cost (1+2+3: patch generate)
                        time_cost_of_model_valid,  # Time cost (model valid(sum))
                        time_cost_of_model_valid_until_first_plausible,  # Time cost (model valid(until first plausible))
                    ]
                )
        print("Generated files:")
        print(f"  details csv: {plausible_repaired_models_details_csv}")
        print(f"  details html report: {prms_index_html}")
        print(f"  summary json: {model_repair_summary_jf}")
        print(f"  summary csv: {model_repair_summary_csv}")

    # Save time counter result
    pgrsu._ilog("Saving time counter result...")
    pgrsu._append_to_file(
        os.path.join(out_dir, "time_counter.json"),
        json.dumps(pgrsu.TimeCounter.pop_result()) + "\n",
    )
