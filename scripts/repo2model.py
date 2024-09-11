# Model source code written in Keras ->
# Preprocessed source code ->
# "Calls to keras" sequence ->
# Signle-file model source code in Keras
#
#       - -(model2graph)- - >>>    graph

import os
import sys
import ast
import json
import random
import shutil
import argparse
# import h5file2src
from io import StringIO
# from astor import to_source as unparse
from astunparse import unparse

PYTHON_EXE_PATH = 'python -B -W ignore'  # Find in PATH

HOOK_KERAS_MODEL_COMPILE_PYCODE_FMT = """
######################################################################


# def execfile(fn):
#     with open(fn, 'r') as fp:
#         old = locals().copy()
#         try:
#             exec(fp.read())
#         finally:
#             new = locals().copy()
#             for k, v in new.items():
#                 if k not in old:
#                     print(k, v)
#                     globals()[k] = v


import os as __OS
# __OS.system('cp ~/.keras/keras.json.tfcopy ~/.keras/keras.json')
import keras as __K
import tensorflow.keras as __K2
import json as __J
import pickle as __PKL
# import timeout_decorator as _Timed

import signal as _SIG__
class _SIG__TimeOutError(Exception):
    def __init__(self):
        super().__init__('Time out!!')
def _SIG__alarm_handler(signum, frame):
    raise _SIG__TimeOutError()
_SIG__.signal(_SIG__.SIGALRM, _SIG__alarm_handler)

_R2M_FLAG_compiled = False
_R2M_compiled_model = None


def seed(seed):
    import os, random, numpy as np, tensorflow as tf
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed(seed=1234)


def __make_fake_compile(model_h5_filename, compile_args_filename, keras_module):
    # print('Making fake `keras.Model.compile` function')
    assert model_h5_filename.endswith('.h5')
    assert compile_args_filename.endswith(('.pkl', '.json'))
    model_h5_filename = __OS.path.abspath(model_h5_filename)
    compile_args_filename = __OS.path.abspath(compile_args_filename)

    def __fake_compile(self, optimizer, loss=None, metrics=None, **kwargs):
        optimizer = keras_module.optimizers.get(optimizer)
        loss = keras_module.losses.get(loss)
        # metrics = metrics or []
        # Save model
        print(f'\\033[0;32m[I] Saving model\\033[0m({{self}}) as {{model_h5_filename}}')
        keras_module.models.save_model(self, model_h5_filename)
        # Save hyperparamers
        print(f'\\033[0;32m[I] Saving compile-args\\033[0m as {{compile_args_filename}}')
        optimizer_arg = {{
            'class_name': optimizer.__class__.__name__,
            'config': optimizer.get_config()
        }}
        loss_arg = loss
        args = {{
            'optimizer': optimizer_arg,
            'loss': loss_arg,
            'metrics': metrics
        }}
        if compile_args_filename.endswith('.pkl'):
            with open(compile_args_filename, 'wb') as fp:
                __PKL.dump(args, fp)
        else:
            assert False, f'Not supported the file format: {{compile_args_filename}}'
        # Process kwargs
        print(f'\\033[0;32m[I] Ignored\\033[0m {{kwargs}}')
        # Mark this program as compiled
        global _R2M_FLAG_compiled
        global _R2M_compiled_model
        _R2M_FLAG_compiled = True
        _R2M_compiled_model = self

    return __fake_compile


def __make_fake_fit(fit_args_filename, keras_module, fit='fit'):
    assert fit_args_filename.endswith('.pkl')
    fit_args_filename = __OS.path.abspath(fit_args_filename)

    def __fake_fit(self, x=None, y=None, batch_size=None, epochs=1, validation_data=None, **kwargs):
        if not _R2M_FLAG_compiled:
            print(f'\\033[0;33m[W] keras.Model.fit -ing: keras.Model.compile wasn\\'t called\\033[0m')
            return
        if _R2M_compiled_model is not self:
            print(f'\\033[0;33m[W] Skip fit model {{self}}\\033[0m')
            return
        # Preprocess args
        batch_size = batch_size or 32
        # Save args
        print(f'\\033[0;32m[I] Saving fit-args\\033[0m as {{fit_args_filename}}')
        args = {{
            'x': x,
            'y': y,
            'validation_data': validation_data,
            'batch_size': batch_size,
            'epochs': epochs,
            # **kwargs # Ignored
        }}
        if fit_args_filename.endswith('.pkl'):
            with open(fit_args_filename, 'wb') as fp:
                __PKL.dump(args, fp)
        else:
            assert False, f'Not supported the file format: {{fit_args_filename}}'
        # Exit propgram
        print(f'\\033[0;32m[I] Exiting {{__file__}}...\\033[0m')
        exit({exit_code})

    def __fake_fit_generator(self, generator, steps_per_epoch=None, epochs=1, **kwargs):
        return __fake_fit(self, epochs=epochs)

    def __fake_train_on_batch(self, *args, **kwargs):
        return __fake_fit(self)

    if fit == 'fit':
        return __fake_fit
    elif fit == 'fit_generator':
        return __fake_fit_generator
    elif fit == 'train_on_batch':
        return __fake_train_on_batch
    else: assert False


__MODEL_H5_FILENAME = f'{program_name}__MODEL__.h5'
__COMPILE_PARAMS_FILENAME = f'{program_name}__COMPILE_HYP__.pkl'
__FIT_PARAMS_FILENAME = f'{program_name}__FIT_HYP__.pkl'
__K.Model.compile = __make_fake_compile(__MODEL_H5_FILENAME, __COMPILE_PARAMS_FILENAME, __K)
__K2.Model.compile = __make_fake_compile(__MODEL_H5_FILENAME, __COMPILE_PARAMS_FILENAME, __K2)
__K.Model.fit = __make_fake_fit(__FIT_PARAMS_FILENAME, __K)
__K2.Model.fit = __make_fake_fit(__FIT_PARAMS_FILENAME, __K2)
__K.Model.fit_generator = __make_fake_fit(__FIT_PARAMS_FILENAME, __K, fit='fit_generator')
__K2.Model.fit_generator = __make_fake_fit(__FIT_PARAMS_FILENAME, __K2, fit='fit_generator')
__K.Model.train_on_batch = __make_fake_fit(__FIT_PARAMS_FILENAME, __K, fit='train_on_batch')
__K2.Model.train_on_batch = __make_fake_fit(__FIT_PARAMS_FILENAME, __K2, fit='train_on_batch')

# from Emptoraemon import Emptoraemon
# setattr(__K.layers, 'Merge', Emptoraemon('keras.layers.Merge'))
# setattr(__K2.layers, 'Merge', Emptoraemon('keras.layers.Merge'))

import keras

# import pandas as __PD
# __PD.read_csv = Emptoraemon('pandas.read_csv')
######################################################################
"""

PYCODE_TAIL = """
######################################################################
## Unuseful code
# try: __make_fake_fit(__FIT_PARAMS_FILENAME, __K)(_R2M_compiled_model)
# except Exception as ex: __make_fake_fit(__FIT_PARAMS_FILENAME, __K2)(_R2M_compiled_model)
# gls = globals().copy().items()
# for name, fn in gls:
#     try:
#         if name.startswith('test_') and callable(fn):
#             print(name)
#             fn()
#     except Exception as ex:
#         pass
# try:
#     __make_fake_fit(__FIT_PARAMS_FILENAME, __K)(_R2M_compiled_model)
# except Exception as ex:
#     __make_fake_fit(__FIT_PARAMS_FILENAME, __K2)(_R2M_compiled_model)
"""

def make_log_print(prefix=None, postfix=None):
    def _log_print_impl(*args, **kwargs):
        if prefix: print(prefix, end='')
        print(*args, **kwargs, end='')
        if postfix: print(postfix, end='')
        print()  # newline
    return _log_print_impl

log_print = make_log_print()


def _sp_system(cmds, logging=True) -> int:
    import subprocess as sp
    assert isinstance(cmds, (str, tuple, list, set))
    if isinstance(cmds, str):
        cmd = cmds
    else:
        cmds = list(cmd)
        cmd = ' '.join(cmds)
    if logging:
        log_print(f'system: {cmd}')
    try:
        ret = -1
        ps = sp.Popen(
                cmds,
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr,
                shell=True)
        ret = ps.wait()
    except KeyboardInterrupt:
        if 'y' == input('exit? (y/n)').lower():
            exit(0)
    if logging:
        log_print(f'exit {cmd}: with {ret}')
    return ret


def _make_try(node):
    def _make_print_ex():
        return ast.parse('print(f"[EX] Exception: {ex}")').body[0]
        # return ast.Pass()

    def _make_timeout_stmt(node, timeout):
        fmtcode = """
if True:
    _SIG__.alarm({timeout})
    pass
    _SIG__.alarm(0)
        """
        fmtast = ast.parse(fmtcode.format_map({'timeout': timeout}))
        _if = fmtast.body[0]
        _if.body[1] = node
        return _if

    return ast.Try(
        body=[_make_timeout_stmt(node, timeout=30)],
        handlers=[
            ast.ExceptHandler(
                type=ast.Name(id='Exception'),
                name='ex',
                body=[_make_print_ex()])
            ],
        orelse=[],
        finalbody=[])


def _visit_to_insert_try(self, node):
    return ast.copy_location(_make_try(node), node)


class _CollectVar(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.var_names = set()

    def visit_Assign(self, node):
        targets = node.targets
        for t in targets:
            if isinstance(t, ast.Name):
                self.var_names.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for e in t.elts:
                    if isinstance(e, ast.Name):
                        self.var_names.add(e.id)


class _InsertTry(ast.NodeTransformer):
    NAMES = (
        'Return',
        'Delete',
        'Assign',
        'AugAssign',
        'AnnAssign',
        'Raise',
        'Try',
        'Assert',
        'Global',
        'Nonlocal',
        'Expr',
    )


for _name in _InsertTry.NAMES:
    setattr(_InsertTry, f'visit_{_name}', _visit_to_insert_try)


def _preproc_py(src_code: str):
    try:
        src_ast = ast.parse(src_code)
        src_ast = _InsertTry().visit(src_ast)
        return unparse(src_ast)
    except Exception as ex:
        log_print('\n\033[0;33m[W] Preprocess source code failed. Exception: {} \033[0m'.format(ex))
        return src_code


def _read_txt(filename):
    import chardet
    try:
        with open(filename, 'rb') as frb:
            cur_encoding = chardet.detect(frb.read())['encoding']
    except Exception as ex:
        cur_encoding = 'UTF-8'
    try:
        with open(filename, 'r', encoding=cur_encoding) as fp:
            return fp.read()
    except:
        return ""


def replaced_lib_with_emptoraemon(src_code: str):
    # Head:
    #   from Emptoraemon import Emptoraemon
    #   import A              =>  try:
    #                                 import A
    #                             except:
    #                                 A = Emptoraemon('A')
    #   import A as B         =>  ... B = Emptoraemon('B(->A)')
    #   from A import A1, A2  =>  ... A1 = Emptoraemon('A.A1'); A2 = Emptoraemon('A.A2')
    class RecordToModify:
        def __init__(self, start, end, modified_code=None, insert_try=False):
            assert modified_code is None or insert_try == False
            self.range = (start, end)  # [start, end)
            self.modified_code = modified_code
            self.insert_try = insert_try

    lines_to_mod = []  # [RecordToModify(), ]
    lines_needs_at_head = [] # For from __future__ import XXXX
    globals_var_names = set()

    src_code = _preproc_py(src_code)
    src_code_lines = src_code.splitlines()
    src_ast = ast.parse(src_code)
    cv_visitor = _CollectVar()
    cv_visitor.visit(src_ast)
    globals_var_names = {*globals_var_names, *cv_visitor.var_names}
    ast_body = src_ast.body
    if ast_body:
        last_lineno = len(src_code_lines)
        ast_body.append(ast.AST(lineno=last_lineno + 1))  # dummy
        for i in range(len(ast_body) - 1):
            item = ast_body[i]
            start_l = item.lineno
            end_l = ast_body[i + 1].lineno
            if isinstance(item, ast.Expr) and \
                    not isinstance(item.value, (ast.Call, ast.Yield, ast.YieldFrom)):
                pass
            elif isinstance(item, ast.Import):
                builder = StringIO()
                for n in item.names:
                    module_name = n.asname or n.name
                    builder.write(f'try: import {n.name}')
                    builder.write(f' as {module_name}' if n.asname else '')
                    builder.write('\n')
                    # builder.write(f'except: {module_name} = Emptoraemon(\'{module_name}\')\n')
                    builder.write(f'except: {module_name} = \'{module_name}\'\n')
                moded_src = builder.getvalue()
                lines_to_mod.append(RecordToModify(start_l, end_l, modified_code=moded_src))
                del builder
            elif isinstance(item, ast.ImportFrom):
                module, level = item.module, item.level
                builder = StringIO()
                if module == '__future__':
                    builder.write('# This line was moved to the beginning of the file\n')
                    for n in item.names:
                        assert n.asname is None
                        lines_needs_at_head.append(f'from __future__ import {n.name}\n')
                else:
                    for n in item.names:
                        obj_name = n.asname or n.name
                        if n.name == '*':
                            assert n.asname is None
                            builder.write(f'from {"."*level}{module} import *\n')
                        else:
                            builder.write(f'try: from {"."*level}{module} import {n.name}')
                            builder.write(f' as {n.asname}' if n.asname else '')
                            builder.write('\n')
                            # builder.write(f'except: {obj_name} = Emptoraemon(f\'{obj_name}\')\n')
                            builder.write(f'except: {obj_name} = f\'{obj_name}\'\n')
                moded_src = builder.getvalue()
                lines_to_mod.append(RecordToModify(start_l, end_l, modified_code=moded_src))
                del builder
            elif isinstance(item, ast.Assign):
                targets = item.targets
                for t in targets:
                    if isinstance(t, ast.Name):
                        globals_var_names.add(t.id)
                    elif isinstance(t, (ast.Tuple, ast.List)):
                        for e in t.elts:
                            if isinstance(e, ast.Name):
                                globals_var_names.add(e.id)
                lines_to_mod.append(RecordToModify(start_l, end_l, insert_try=True))
            else:
                lines_to_mod.append(RecordToModify(start_l, end_l, insert_try=True))

    # import_emptoraemon = 'from Emptoraemon import Emptoraemon\n'
    import_emptoraemon = '# from Emptoraemon import Emptoraemon\n'
    if lines_to_mod:
        # dummy = RecordToModify(
        #     start=last_lineno + 1024,  # LARGE
        #     end=last_lineno + 2048,  # LARGE
        #     modified_code=None)
        # lines_to_mod.append(dummy)  # dummy
        # mod_iter = iter(lines_to_mod)
        # mod = next(mod_iter)
        lines = src_code_lines
        modified_src_builder = StringIO()
        modified_src_builder.write(import_emptoraemon)
        for var_name in globals_var_names:
            modified_src_builder.write(f'{var_name} = None\n')
        # for i, l in enumerate(lines, start=1):
        #     start, end = mod.range
        #     if i == end:
        #         mod = next(mod_iter)
        #         start, end = mod.range
        #     if i < start:
        #         modified_src_builder.write(l)
        #         modified_src_builder.write('\n')
        #     elif i == start:
        #         modified_src_builder.write(mod.modified_code)
        #     elif i > start or i < end:
        #         pass
        for mod in lines_to_mod:
            start, end = mod.range
            ins_try = mod.insert_try
            mod_src = mod.modified_code
            if ins_try:
                assert mod_src is None
                block_lines = lines[start - 1: end - 1]
                # log_print(f'[{start}, {end})', '\n===={\n', "\n".join(block_lines), '\n}====\n')
                modified_src_builder.write('try:\n')
                for line in block_lines:
                    modified_src_builder.write(f'    {line}\n')
                modified_src_builder.write('except Exception as ex:print("Exception: ", ex)\n')
            else:
                assert mod_src is not None
                modified_src_builder.write(f'{mod_src}\n')


        return lines_needs_at_head, modified_src_builder.getvalue()

    return lines_needs_at_head, import_emptoraemon + src_code


def py2model(program_name: str, src_code: str, out_model_fp, *, disable_ignore_exception):
    def proc_run_hooked_failure(h5_f, c_args_f, f_args_f, ret):
        fail, fail_msg = False, '[W] Fail to run hooked model:\n'
        if ret != exit_code:
            # with open(f_args_f, 'w', encoding='UTF-8') as fp:
            #     fp.write('{"batch_size": 32, "epochs": 1}')
            fail = True
            fail_msg += '\t* fake keras.Model.fit wasn\'t called successfully\n'
        if not os.path.exists(h5_f):
            fail = True
            fail_msg += f'\t* not found {h5_f}\n'
        if not os.path.exists(c_args_f):
            fail = True
            fail_msg += f'\t* not found {c_args_f}\n'
        if not os.path.exists(f_args_f):
            fail = True
            fail_msg += f'\t* not found {f_args_f}\n'
        return fail, fail_msg
    
    try:
        # Hook python code
        exit_code = random.randint(100, 255)
        hook_header = HOOK_KERAS_MODEL_COMPILE_PYCODE_FMT.format_map(
            {'exit_code': exit_code, 'program_name': os.path.abspath(program_name).replace('\\', '/')})
        if not disable_ignore_exception:  # Ignore exception
            header_lines, mod_src = replaced_lib_with_emptoraemon(src_code)
            hooked_src = ''.join(header_lines) + hook_header + mod_src + PYCODE_TAIL
        else:  # Not ignore exception
            hooked_src = hook_header + src_code + PYCODE_TAIL
        hooked_filename = f'{program_name}__HOOKED__.py_'
        with open(hooked_filename, 'w', encoding='UTF-8') as hooked_f:
            hooked_f.write(hooked_src)
        # Run hooked python code
        ret = 0
        try:
            # os.system('cp ~/.keras/keras.json.tfcopy ~/.keras/keras.json')
            # ret = os.system(f'cd "{os.path.dirname(hooked_filename)}" && timeout 90s {PYTHON_EXE_PATH} "{os.path.basename(hooked_filename)}"')
            timeout_cmd = '' if disable_ignore_exception else 'timeout 90s '
            ret = _sp_system(f'cd "{os.path.dirname(hooked_filename)}" && {timeout_cmd}{PYTHON_EXE_PATH} "{os.path.basename(hooked_filename)}"')
        except KeyboardInterrupt:
            _cmd = input('exit? (y/n)')
            if _cmd.lower() == 'y': exit(0)
        finally:
            os.remove(hooked_filename)
        h5_model_filename = f'{program_name}__MODEL__.h5'
        compile_args_filename = f'{program_name}__COMPILE_HYP__.pkl'
        fit_args_filename = f'{program_name}__FIT_HYP__.pkl'
        fail, fail_msg = proc_run_hooked_failure(h5_f=h5_model_filename,
                                                 c_args_f=compile_args_filename,
                                                 f_args_f=fit_args_filename,
                                                 ret=ret)
        if fail:
            log_print(f'\033[0;33m{fail_msg}\033[0m')
            return False, fail_msg, None
        if False:
            # H5 to python code
            try:
                ## Model-define body
                model_var_name = h5file2src.main(h5_model_filename, out_model_fp)
                ## Model-compile
                with open(compile_args_filename, 'r', encoding='UTF-8') as fp:
                    j = json.load(fp)
                args_string = f'optimizer={j["optimizer"]}, loss=\'{j["loss"]["loss_name"]}\', metrics={j["metrics"]}'
                out_model_fp.write(f'{model_var_name}.compile({args_string})\n')
                ## Model-fit
                with open(fit_args_filename, 'r', encoding='UTF-8') as fp:
                    j = json.load(fp)
                args_string = f'x=None, y=None, batch_size={j["batch_size"]}, epochs={j["epochs"]}'
                out_model_fp.write(f'{model_var_name}.fit({args_string})\n')
            except:
                pass
        else:
            out_model_fp.write('# nothing ...\n')
    except Exception as ex:
        import traceback as tb
        from sys import exc_info
        from io import StringIO
        fail = True
        fail_msg = f'\t* Exception ({str(ex)}) :\n'
        tb_msg_builder = StringIO()
        _, _, exec_tb = exc_info()
        tb.print_tb(exec_tb, file=tb_msg_builder)
        fail_msg += '\t\t|' + tb_msg_builder.getvalue().replace(
            '\n', '\n\t\t|')
        return False, fail_msg, None
    return True, '', (h5_model_filename, compile_args_filename, fit_args_filename)


class SFModel:
    def __init__(self, name: str, path: str, sf_model_f: str, h5_f: str, compile_args_f: str, fit_args_f: str, origin_filename: str):
        self.name = name
        self.path = path
        self.sf_model_f = sf_model_f
        self.h5_f = h5_f
        self.compile_args_f = compile_args_f
        self.fit_args_f = fit_args_f
        self.origin_filename = origin_filename
        # self.commit = None  # Set/Get in gen_dataset.py


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
            parts = unparse(node).strip().split(".")
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


def main(repo_path: str, out_dir: str, log_path: str, t_repo_path: str = None, out_subdir_name: str = None, disble_ignore_exception=False):
    def proc_file(filename, log_fp, disable_ignore_exception):
        log_print('\n\033[1;34m============ Start ===============\033[0m')
        log_print(f'\033[0;32m[I] Processing py\033[0m: {filename}')
        prog_name = filename[:-3]
        # print('===>@@@', prog_name, repo_path, os.path.relpath(prog_name, start=repo_path))
        sfm_dir = out_subdir_name or os.path.relpath(f'{prog_name}.py', repo_path).replace('\\', '.').replace('/', '.').replace(':', '.')
        sfm_dir = os.path.join(out_dir, sfm_dir)
        os.makedirs(sfm_dir, exist_ok=True)
        sf_model_f = os.path.join(sfm_dir, '__SF_MODEL__.py')
        h5_f = os.path.join(sfm_dir, '__MODEL__.h5')
        compile_args_f = os.path.join(sfm_dir, '__COMPILE_HYP__.pkl')
        fit_args_f = os.path.join(sfm_dir, '__FIT_HYP__.pkl')
        if bool(eval(os.environ.get('R2M_SKIP_EXISTING', 'False'))):
            if os.path.exists(h5_f) and os.path.exists(compile_args_f) and os.path.exists(fit_args_f):
                log_print(f'\033[1;32m[I] (Skip)     Gen-ing sf model\033[0m: {sf_model_f}')
                return True, SFModel('<unknown-name>', sfm_dir, sf_model_f, h5_f, compile_args_f, fit_args_f, filename)
        # with open(filename, 'r', encoding='UTF-8') as fp:
        #     src_code = fp.read()
        src_code = _read_txt(filename)
        if bool(eval(os.getenv('R2M_FIX_SFMODEL_CODE', '0'))):
            src_code = _fix_sfmodel_code(src_code)
        with open(sf_model_f, 'w', encoding='UTF-8') as sf_model_fp:
            log_print(f'\033[1;32m[I] (Start)    Gen-ing sf model\033[0m: {sf_model_f}')
            ok, msg, files = py2model(prog_name, src_code, sf_model_fp, disable_ignore_exception=disable_ignore_exception)
            if not ok:
                log_print(f'\033[1;33m[W] (Failed)   Gen-ing sf model\033[0m: {sf_model_f}')
                log_fp.write(f'Process {filename} failed: {msg}')
                log_fp.flush()
            else:
                shutil.move(files[0], h5_f)
                shutil.move(files[1], compile_args_f)
                shutil.move(files[2], fit_args_f)
                log_print(f'\033[1;32m[I] (Finished) Gen-ing sf model\033[0m: {sf_model_f}')
        if not ok:  # Remove empty files
            os.remove(sf_model_f)
            os.rmdir(sfm_dir)
        log_print('\033[1;34m============ End ===============\033[0m')
        return ok, SFModel('<unknown-name>', sfm_dir, sf_model_f, h5_f, compile_args_f, fit_args_f, filename)

    def try_early_stop(enable):
        if enable and eval(os.environ.get('R2M_EARLY_STOP', 'False')):
            exit(-1)

    log_print(f'\033[1;32m[I] (Start)    Processing repo\033[0m: {repo_path}')
    os.makedirs(out_dir, exist_ok=True)
    sfmodels = []
    with open(log_path, 'a', encoding='UTF-8') as log_fp:
        if os.path.isfile(repo_path):
            if repo_path.endswith('.py'):
                filename = repo_path
                repo_path = t_repo_path or os.path.dirname(os.path.abspath(repo_path))
                ok, sfm = proc_file(filename, log_fp, disable_ignore_exception=disble_ignore_exception)
                try_early_stop(not ok)
                return [sfm] if ok else []
            return

        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    filename = os.path.join(root, file)
                    ok, sfm = proc_file(filename, log_fp, disable_ignore_exception=disble_ignore_exception)
                    try_early_stop(not ok)
                    if ok: sfmodels.append(sfm)
    log_print(f'\033[1;32m[I] (Finished) Processing repo\033[0m: {repo_path}')
    return sfmodels


__all__ = ['main', 'py2model']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--repo-path', type=str, required=True)
    parser.add_argument('-t', '--t-repo-path', type=str, required=False, default=None)
    parser.add_argument('-o', '--out-dir', type=str, required=True)
    parser.add_argument('-l', '--log-path', type=str)
    parser.add_argument('--disable-ignore-exception', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    if not args.log_path:
        args.log_path = os.path.join(args.out_dir, 'fail-to-gen.log')

    main(args.repo_path,
         args.out_dir,
         args.log_path,
         args.t_repo_path,
         disble_ignore_exception=args.disable_ignore_exception)
