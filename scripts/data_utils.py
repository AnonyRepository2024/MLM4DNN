import io
import sys
import ast
import tokenize
import random
import functools
import _rs_utils as pgrsu
from dataclasses import dataclass
from typing import Callable, Tuple, List, Set, Dict, Union, TextIO, Any

ast_unparse = ast.unparse

##+ AST Utils

class FakeOp(ast.AST):
    _fields = ('op', 'precedence')


class ReplaceXWithY(ast.NodeTransformer):
    def __init__(self, x_cls_names: list, is_x, make_y, ignore_parents=None) -> None:
        super().__init__()
        assert callable(make_y)
        assert callable(is_x)
        ignore_parents = ignore_parents or []
        self.is_x = is_x
        self.make_y = make_y

        def echo_v(node):
            return node

        for name in ignore_parents:
            setattr(self, f'visit_{name}', echo_v)

        def visit_X(node):
            self.generic_visit(node)
            return self.make_y(node) if self.is_x(node) else node

        for name in x_cls_names:
            setattr(self, f'visit_{name}', visit_X)


def _clone_ast(root):
    if isinstance(root, ast.Module):
        mod = root
        ret = lambda m: m
    elif isinstance(root, (tuple, list)):
        mod = ast.Module(body=root, type_ignores=[])
        ret = lambda m: m.body
    elif isinstance(root, ast.expr):
        mod = ast.Module(body=[ast.Expr(value=root)], type_ignores=[])
        ret = lambda m: m.body[0].value
    else:
        mod = ast.Module(body=[root], type_ignores=[])
        ret = lambda m: m.body[0]

    return ret(ast.parse(ast.unparse(mod)))


def _set_height(root: ast.AST) -> ast.AST:
    subtree_max_height = 0
    for child in ast.iter_child_nodes(root):
        subtree_max_height = max(subtree_max_height,
                                 _set_height(child)._pg_height)
    root._pg_height = subtree_max_height + 1
    return root


def _set_node_id(root, init_id=0):
    for i, n in enumerate(ast.walk(root), start=init_id):
        n._pg_node_id = i
    return root


def _set_parent(root):
    if not hasattr(root, 'parent'):
        root.parent = None
    for child in ast.iter_child_nodes(root):
        child.parent = root
        _set_parent(child)
    return root


def _iter_parent_chain(node, with_self = True):
    if with_self: yield node
    p = node
    while p.parent:
        yield p.parent
        p = p.parent


def _check_parent_chain(node, checker, mode='any', with_self=False):
    assert callable(checker)
    assert mode in ('any', 'all')
    pchain = list(_iter_parent_chain(node, with_self=with_self))
    pchain = [checker(p) for p in pchain]
    return bool(eval(f'{mode}(pchain)'))


def _check_call_member_fn(fn_name: str, node: ast.Call, nargs: int = None, kwargkeys: set = None):
    return isinstance(node, ast.Call) and \
            isinstance(node.func, ast.Attribute) and \
            node.func.attr == fn_name and \
            (nargs is None or len(node.args) == nargs) and \
            (kwargkeys is None or set(kwargkeys) == {e.arg for e in node.keywords})


def _find_x(node, is_x, ignore_sub_scope=False) -> list:
    assert callable(is_x)
    ck = lambda n: not _check_parent_chain(n,
                                           checker=lambda p: p != node and isinstance(p, (ast.FunctionDef, ast.ClassDef)),
                                           mode='any',
                                           with_self=False) \
         if ignore_sub_scope else lambda n: True
    return [n for n in ast.walk(node) if is_x(n) if ck(n)]


def _random_sample_n_subtrees(nodes: List[ast.AST], k: int) -> Set[ast.AST]:
    nodes = nodes.copy()
    result = set()
    while nodes and len(result) < k:
        n = random.choice(nodes)
        result.add(n)
        nodes.remove(n)
        for parent in _iter_parent_chain(n, with_self=False):
            try:
                nodes.remove(parent)
            except:
                pass
        for child in ast.walk(n):
            try:
                nodes.remove(child)
            except:
                pass
    return result


def _is_block_stmt(node):
    block_stmts = (
        ast.For,  # BODY: body, orelse
        ast.AsyncFor,  # BODY: body, orelse
        ast.While,  # BODY: body, orelse
        ast.If,  # BODY: body, orelse
        ast.With,  # BODY: body
        ast.AsyncWith,  # BODY: body
        ast.Try,  # BODY: body, orelse, finalbody
        ast.TryStar,  # BODY: body, orelse, finalbody
        ast.ExceptHandler,  # BODY: body
        ast.FunctionDef,  # BODY: body
        ast.ClassDef,  # BODY: body
        ast.Module,  # BODY: body
    )
    return isinstance(node, block_stmts)


def _foreach_stmt(block):
    EMP = []
    yield from getattr(block, 'body', EMP)
    yield from getattr(block, 'orelse', EMP)
    yield from getattr(block, 'finalbody', EMP)


_OP_SYMBOL_TABLE = {
    # ast.operator
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.Mod: '%',
    ast.Pow: '**',
    ast.LShift: '<<',
    ast.RShift: '>>',
    ast.BitOr: '|',
    ast.BitXor: '^',
    ast.BitAnd: '&',
    ast.MatMult: '@',
    # ast.boolop
    ast.And: 'and',
    ast.Or: 'or',
    # ast.unaryop
    ast.UAdd: '+',
    ast.USub: '-',
    ast.Not: 'not',
    ast.Invert: '~',
    # ast.cmpop
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
    ast.Is: 'is',
    ast.IsNot: 'is not',
    ast.In: 'in',
    ast.NotIn: 'not in'
}
def _get_op_symbol(a):
    return _OP_SYMBOL_TABLE[a.__class__]


_SHARED_CLASSES = (*_OP_SYMBOL_TABLE.keys(), ast.Load, ast.Store, ast.Del)
def _fix_single_ins(node):
    # Note When a string is parsed by ast.parse(), operator nodes (subclasses of
    # ast.operator, ast.unaryop, ast.cmpop, ast.boolop and ast.expr_context) on 
    # the returned tree will be singletons. Changes to one will be reflected in
    # all other occurrences of the same value (e.g. ast.Add).
    return ReplaceXWithY(x_cls_names=[c.__name__ for c in _SHARED_CLASSES],
                         is_x=lambda x: isinstance(x, _SHARED_CLASSES),
                         make_y=lambda x: type(x)()).visit(node)

def _is_name_node(node: ast.AST):
    if isinstance(node, ast.Name):
        return True
    elif isinstance(node, ast.Attribute):
        return _is_name_node(node.value)
    return False


def _check_call_member_fn(fn_name: str, node: ast.Call, nargs: int = None, kwargkeys: set = None):
    return isinstance(node, ast.Call) and \
            isinstance(node.func, ast.Attribute) and \
            node.func.attr == fn_name and \
            (nargs is None or len(node.args) == nargs) and \
            (kwargkeys is None or set(kwargkeys) == {e.arg for e in node.keywords})


##+ Tokenize

_MASK_TOKEN_PREFIX = '<mask:'
_MASK_TOKEN_SUFFIX = '>'
_MASK_TOKEN_FMT = f'{_MASK_TOKEN_PREFIX}{{i}}{_MASK_TOKEN_SUFFIX}'
_MASK_MARKER_PREFIX = '__mask_'
_MASK_MARKER_SUFFIX = '__'
_MASK_MARKER_FMT = f'{_MASK_MARKER_PREFIX}{{i}}{_MASK_MARKER_SUFFIX}'

_NIL_TOKEN = '<nil>'
_NIL_MARKER = '__nil__'
_UNK_TOKEN = '<unk>'
_UNK_MARKER = '__unk__'


def _make_mask_token(i: int):
    return _MASK_TOKEN_FMT.format(i=i)


def _is_mask_token(t: str):
    return t.startswith(_MASK_TOKEN_PREFIX) and \
            t.endswith(_MASK_TOKEN_SUFFIX)


def _parse_mask_token(t: str):
    assert _is_mask_token(t)
    return int(t[len(_MASK_TOKEN_PREFIX):
                 -len(_MASK_TOKEN_SUFFIX)])


def _make_mask_marker(i: int):
    return _MASK_MARKER_FMT.format(i=i)


def _is_mask_marker(m: str):
    return m.startswith(_MASK_MARKER_PREFIX) and \
            m.endswith(_MASK_MARKER_SUFFIX)


def _parse_mask_marker(m: str):
    assert _is_mask_marker(m)
    return int(m[len(_MASK_MARKER_PREFIX):
                 -len(_MASK_MARKER_SUFFIX)])


def _is_nil_marker(m: str):
    return m == _NIL_MARKER


def _make_nil_marker():
    return _NIL_MARKER


def _is_nil_token(t: str):
    return t == _NIL_TOKEN


def _make_nil_token():
    return _NIL_TOKEN


def _make_mask_hint_token(hint):
    return f'__mhint_{hint}__'


def _is_unk_marker(m):
    return m == _UNK_MARKER


def _make_unk_token():
    return _UNK_TOKEN


_make_token_marker = lambda x: f'<token_mark:{x}>'

_TOKM_NUM_START = _make_token_marker('num_start')
_TOKM_NUM_END = _make_token_marker('num_end')
_TOKM_NEWLINE = _make_token_marker('newline')
_TOKM_DEDENT = _make_token_marker('dedent')
_TOKM_NL = _make_token_marker('nl')
# _TOKM_FSTRING_START = ...
# _TOKM_FSTRING_MIDDLE = ...
# _TOKM_FSTRING_END = ...

# <token_mark:indent:i>
_is_indent_token_marker = lambda t: t.startswith('<token_mark:indent:') and t.endswith('>')
_make_indent_token_marker = lambda i: _make_token_marker(f'indent:{i}')
_parse_indent_token_marker = lambda t: int(t[len('<token_mark:indent:'): -len('>')])

## FIXME: Many bugs (!!!reimpl it!!!)
# def _untokenize_python(tokens, indent=None, fmt=False):
#     indent = indent or '    '
#     sb = io.StringIO()
#     indent_q = []
#     last_tk = None
#     for tk in tokens:
#         if _is_mask_token(tk):
#             sb.write(f' {_make_mask_marker(i=_parse_mask_token(tk))} ')
#         elif _is_nil_token(tk):
#             sb.write(f' {_make_nil_marker()} ')
#         elif tk in (_TOKM_NUM_START, _TOKM_NUM_END, _TOKM_NL):
#             pass
#         elif _is_indent_token_marker(tk):
#             indent_q.append(indent)
#             sb.write(indent)
#         elif tk == _TOKM_DEDENT:
#             indent_q.pop()
#         elif tk == _TOKM_NEWLINE:
#             sb.write(f'\n')
#             sb.write(''.join(indent_q))
#         else:
#             sb.write(f'{tk} ')
#         last_tk = tk
#     src = sb.getvalue()
#     return src if not fmt else ast_unparse(ast.parse(src))


def _tokenize_python(fp):
    def get_tokens(tk: tokenize.TokenInfo):
        # Process [mask] & number
        if _is_mask_marker(tk.string):
            return (_make_mask_token(i=_parse_mask_marker(tk.string)), )
        elif _is_nil_marker(tk.string):
            return (_make_nil_token(), )
        elif _is_unk_marker(tk.string):
            return (_make_unk_token(), )
        elif tag_b := _parse_tag_begin(tk.string):
            return (_make_tag_begin_token(tag_b), )
        elif tag_e := _parse_tag_end(tk.string):
            return (_make_tag_end_token(tag_e), )
        # elif tk.type == tokenize.NUMBER:
        #     return (_TOKM_NUM_START, *tuple(tk.string), _TOKM_NUM_END)
        elif tk.type == tokenize.ENDMARKER:
            return ()
        elif tk.type == tokenize.NEWLINE:
            return (_TOKM_NEWLINE, )
        elif tk.type == tokenize.NL:
            return (_TOKM_NL, )
        elif tk.type == tokenize.INDENT:
            return (_make_indent_token_marker(len(tk.string)), )
        elif tk.type == tokenize.DEDENT:
            return (_TOKM_DEDENT, )
        return (tk.string, )

    tokens = [t for tk in tokenize.generate_tokens(fp.readline) \
                    for t in get_tokens(tk)]
    while tokens and tokens[-1] == _TOKM_NEWLINE:
        tokens.pop()
    return tokens


def _tokenize_python_string(string: str):
    assert string is not None
    fp = io.StringIO(string)
    return _tokenize_python(fp)


def _tokenize_python_file(filename: str):
    with tokenize.open(filename) as fp:
        return _tokenize_python(fp)


##+ Mask

_MHINT_K_PADDING = 'K_padding'
_MHINT_K_INITIALIZER = 'K_initializer'
_MHINT_K_CONSTRAINT = 'K_constraint'
_MHINT_K_ACTIVATION = 'K_activation'
_MHINT_K_DATAFORMAT = 'K_dataformat'
_MHINT_K_LOSS = 'K_loss'
_MHINT_K_OPTIMIZER = 'K_optimizer'
_MHINT_K_METRIC = 'K_metric'
_MHINT_K_REGULARIZER = 'K_regularizer'
_MHINT_K_LAYER = 'K_layer'
_MHINT_K_MODEL_COMPILE = 'K_model_compile'
_MHINT_K_MODEL_FIT = 'K_model_fit'


def _mhint_is_keras_component_hint(hint: str):
    return hint.startswith('K_')


# From https://github.com/keras-team/keras/blob/master/keras/...
_keras_padding_names = ['valid', 'same', 'causal']
_keras_initializer_names = ['Zeros', 'Ones', 'RandomUniform', 'TruncatedNormal', 'Identity', 'Orthogonal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal', 'he_uniform']
_keras_constraint_names = ['Constraint', 'MaxNorm', 'NonNeg', 'UnitNorm', 'MinMaxNorm', 'constraint', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm']
_keras_activation_names = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
_keras_dataformat_names = ['channels_first', 'channels_last']
_keras_loss_names = ['BCE', 'MSE', 'MAE', 'MAPE', 'MSLE', 'KLD', 'bce', 'mse', 'mae', 'mape', 'msle', 'kld', 'logcosh', 'huber_loss', 'BinaryCrossentropy', 'CategoricalCrossentropy', 'CategoricalHinge', 'CosineSimilarity', 'Hinge', 'Huber', 'KLDivergence', 'LogCosh', 'LossFunctionWrapper', 'MeanAbsoluteError', 'MeanAbsolutePercentageError', 'MeanSquaredError', 'MeanSquaredLogarithmicError', 'Poisson', 'SparseCategoricalCrossentropy', 'SquaredHinge', 'binary_crossentropy', 'categorical_crossentropy', 'categorical_hinge', 'cosine_proximityLoss', 'cosine_similarity', 'hinge', 'huber', 'kl_divergence', 'kullback_leibler_divergence', 'log_cosh', 'logcosh', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'mean_squared_logarithmic_error', 'poisson', 'sparse_categorical_crossentropy', 'squared_hinge']
_keras_optimizer_names = ['Adadelta', 'Adafactor', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'Ftrl', 'Lion', 'LossScaleOptimizer', 'Nadam', 'Optimizer', 'RMSprop', 'SGD', 'adadelta', 'adafactor', 'adagrad', 'adam', 'adamax', 'adamw', 'ftrl', 'lion', 'lossscaleoptimizer', 'nadam', 'optimizer', 'rmsprop', 'sgd']
_keras_metric_names = ['accuracy', 'acc', 'crossentropy', 'ce', 'AUC', 'Accuracy', 'BCE', 'BinaryAccuracy', 'BinaryCrossentropy', 'BinaryIoU', 'CategoricalAccuracy', 'CategoricalCrossentropy', 'CategoricalHinge', 'CosineSimilarity', 'F1Score', 'FBetaScore', 'FalseNegatives', 'FalsePositives', 'Hinge', 'IoU', 'KLDivergence', 'LogCoshError', 'MAE', 'MAPE', 'MSE', 'MSLE', 'Mean', 'MeanAbsoluteError', 'MeanAbsolutePercentageError', 'MeanIoU', 'MeanMetricWrapper', 'MeanSquaredError', 'MeanSquaredLogarithmicError', 'Metric', 'OneHotIoU', 'OneHotMeanIoU', 'Poisson', 'Precision', 'PrecisionAtRecall', 'R2Score', 'Recall', 'RecallAtPrecision', 'RootMeanSquaredError', 'SensitivityAtSpecificity', 'SparseCategoricalAccuracy', 'SparseCategoricalCrossentropy', 'SparseTopKCategoricalAccuracy', 'SpecificityAtSensitivity', 'SquaredHinge', 'Sum', 'TopKCategoricalAccuracy', 'TrueNegatives', 'TruePositives', 'accuracy', 'auc', 'bce', 'binary_accuracy', 'binary_crossentropy', 'binary_io_u', 'categorical_accuracy', 'categorical_crossentropy', 'categorical_hinge', 'cosine_similarity', 'f1_score', 'f_beta_score', 'false_negatives', 'false_positives', 'hinge', 'io_u', 'kl_divergence', 'log_cosh_error', 'mae', 'mape', 'mean', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_io_u', 'mean_metric_wrapper', 'mean_squared_error', 'mean_squared_logarithmic_error', 'metric', 'mse', 'msle', 'one_hot_io_u', 'one_hot_mean_io_u', 'poisson', 'precision', 'precision_at_recall', 'r2_score', 'recall', 'recall_at_precision', 'root_mean_squared_error', 'sensitivity_at_specificity', 'sparse_categorical_accuracy', 'sparse_categorical_crossentropy', 'sparse_top_k_categorical_accuracy', 'specificity_at_sensitivity', 'squared_hinge', 'sum', 'top_k_categorical_accuracy', 'true_negatives', 'true_positives']
_keras_regularizer_names = ['L1', 'L1L2', 'L2', 'OrthogonalRegularizer', 'Regularizer', 'l1', 'l1l2', 'l2', 'orthogonal_regularizer', 'regularizer']
_keras_possible_inits_api_root = ['__root__.tensorflow.initializers',
                                  '__root__.keras.initializers',
                                  '__root__.keras_extensions.initializers',
                                  '__root__.tensorflow.keras.initializers',
                                  '__root__.keras_core.initializers',
                                  '__root__.tensorflow.python.keras.initializers']
_keras_possible_constraints_api_root = ['__root__.tensorflow.compat.v1.keras.constraints',
                                        '__root__.tensorflow.contrib.keras.constraints',
                                        '__root__.toolkit4nlp.backend.keras.constraints',
                                        '__root__.bert4keras.backend.keras.constraints',
                                        '__root__.keras_xlnet.backend.keras.constraints',
                                        '__root__.tensorflow.python.keras.constraints',
                                        '__root__.tensorflow.keras.constraints',
                                        '__root__.keras.constraints']
_keras_possible_activations_api_root = ['__root__.tensorflow.keras.activations',
                                        '__root__.tensorflow_addons.activations',
                                        '__root__.keras.activations',
                                        '__root__.keras_core.activations']
_keras_possible_losses_api_root = ['__root__.keras_cv.losses',
                                    '__root__.tensorflow.python.keras.losses',
                                    '__root__.keras_contrib.losses',
                                    '__root__.tensorflow.losses',
                                    '__root__.keras_maskrcnn.losses',
                                    '__root__.keras_fsl.losses',
                                    '__root__.segmentation_models.losses',
                                    '__root__.keras_retinanet.losses',
                                    '__root__.tensorflow_similarity.losses',
                                    '__root__.tensorflow_addons.losses',
                                    '__root__.keras_uncertainty.losses',
                                    '__root__.tensorflow.keras.losses',
                                    '__root__.keras.losses',
                                    '__root__.losses',
                                    '__root__.keras_core.losses',
                                    '__root__.keras.backend.tf.losses']
_keras_possible_optimizers_api_root = ['__root__.bert4keras.optimizers',
                                       '__root__.keras.optimizers',
                                       '__root__.keras.optimizers.optimizers',
                                       '__root__.keras.preprocessing.image.optimizers',
                                       '__root__.keras_core.optimizers',
                                       '__root__.keras_exp.multigpu.optimizers',
                                       '__root__.keras_rewiring.optimizers',
                                       '__root__.keras_xlnet.backend.keras.optimizers',
                                       '__root__.kungfu.tensorflow.optimizers',
                                       '__root__.tensorflow.compat.v1.keras.optimizers',
                                       '__root__.tensorflow.keras.optimizers',
                                       '__root__.tensorflow.optimizers',
                                       '__root__.tensorflow.python.keras.optimizers',
                                       '__root__.tensorflow_addons.optimizers',
                                       '__root__.toolkit4nlp.optimizers']
_keras_possible_metrics_api_root = ['__root__.deepkt.metrics',
                                    '__root__.keras_fsl.metrics',
                                    '__root__.keras.metrics',
                                    '__root__.kgcnn.metrics',
                                    '__root__.tensorflow.keras.metrics',
                                    '__root__.toolkit4nlp.backend.keras.metrics',
                                    '__root__.keras_nlp.metrics',
                                    '__root__.keras_core.metrics',
                                    '__root__.bert4keras.backend.keras.metrics',
                                    '__root__.segmentation_models.metrics',
                                    '__root__.tensorflow_addons.metrics',
                                    '__root__.tensorflow.metrics',
                                    '__root__.utils.metrics',
                                    '__root__.keras_contrib.metrics',
                                    '__root__.sklearn.metrics',
                                    '__root__.kgcnn.metrics.metrics']
_keras_possible_regularizers_api_root = ['__root__.regularizers',
                                         '__root__.tensorflow.keras.regularizers',
                                         '__root__.keras_core.regularizers',
                                         '__root__.keras.regularizers',
                                         '__root__.tensorflow.python.keras.regularizers']
_keras_possible_layers_api_root = ['__root__.bert4keras.backend.keras.layers',
                                   '__root__.bert4keras.layers',
                                   '__root__.keras.layers',
                                   '__root__.keras.legacy.layers',
                                   '__root__.keras_bert.layers',
                                   '__root__.keras_compressor.layers', 
                                   '__root__.keras_contrib.layers',
                                   '__root__.keras_core.layers',
                                   '__root__.keras_cv.layers',
                                   '__root__.keras_dgl.layers',
                                   '__root__.keras_fsl.layers',
                                   '__root__.keras_nlp.layers',
                                   '__root__.keras_uncertainty.backend.layers',
                                   '__root__.keras_uncertainty.layers',
                                   '__root__.keras_xlnet.backend.keras.layers',
                                   '__root__.tensorflow.compat.v1.keras.layers',
                                   '__root__.tensorflow.contrib.keras.api.keras.layers',
                                   '__root__.tensorflow.keras.layers',
                                   '__root__.tensorflow.python.keras.layers']
# _keras_possible_api_root = {root \
#                             for root in [*_keras_possible_inits_api_root,
#                                          *_keras_possible_activations_api_root,
#                                          *_keras_possible_losses_api_root,
#                                          *_keras_possible_optimizers_api_root,
#                                          *_keras_possible_metrics_api_root,
#                                          *_keras_possible_regularizers_api_root,
#                                          *_keras_possible_layers_api_root] if root.split('.')[-2] == 'keras'}
_keras_model_compile_infix = '.compile('  #FIXME: More strict checking if needed
_keras_model_fit_infix = '.fit('  #FIXME: More strict checking if needed
def _get_possible_ast_hint(node: ast.AST) -> str:
    # DNN component releated
    if isinstance(node, ast.Constant) and isinstance((val := node.value), str):
        if val in _keras_padding_names: return _MHINT_K_PADDING
        if val in _keras_initializer_names: return _MHINT_K_INITIALIZER
        if val in _keras_constraint_names: return _MHINT_K_CONSTRAINT
        if val in _keras_activation_names: return _MHINT_K_ACTIVATION
        if val in _keras_dataformat_names: return _MHINT_K_DATAFORMAT
        if val in _keras_loss_names: return _MHINT_K_LOSS
        if val in _keras_optimizer_names: return _MHINT_K_OPTIMIZER
        if val in _keras_metric_names: return _MHINT_K_METRIC
        if val in _keras_regularizer_names: return _MHINT_K_REGULARIZER
    elif isinstance(node, ast.Call) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        line = ast_unparse(node)
        if _keras_model_compile_infix in line: return _MHINT_K_MODEL_COMPILE
        if _keras_model_fit_infix in line: return _MHINT_K_MODEL_FIT
        if any([line.startswith(p) for p in _keras_possible_layers_api_root]): return _MHINT_K_LAYER
        if any([line.startswith(p) for p in _keras_possible_inits_api_root]): return _MHINT_K_INITIALIZER
        if any([line.startswith(p) for p in _keras_possible_constraints_api_root]): return _MHINT_K_CONSTRAINT
        if any([line.startswith(p) for p in _keras_possible_losses_api_root]): return _MHINT_K_LOSS
        if any([line.startswith(p) for p in _keras_possible_optimizers_api_root]): return _MHINT_K_OPTIMIZER
        if any([line.startswith(p) for p in _keras_possible_metrics_api_root]): return _MHINT_K_METRIC
        if any([line.startswith(p) for p in _keras_possible_regularizers_api_root]): return _MHINT_K_REGULARIZER
        # !!!NO!!!: if any([p in line for p in _keras_possible_api_root]): return 'kerasApiCall'
    elif _is_name_node(node):
        line = ast_unparse(node)
        if any([line.startswith(p) for p in _keras_possible_inits_api_root]): return _MHINT_K_INITIALIZER
        if any([line.startswith(p) for p in _keras_possible_constraints_api_root]): return _MHINT_K_CONSTRAINT
        if any([line.startswith(p) for p in _keras_possible_activations_api_root]): return _MHINT_K_ACTIVATION
        if any([line.startswith(p) for p in _keras_possible_losses_api_root]): return _MHINT_K_LOSS
        if any([line.startswith(p) for p in _keras_possible_metrics_api_root]): return _MHINT_K_METRIC
    # Name
    if _is_name_node(node):  # Name or Attribute(root is Name)
        return 'Name'
    # Constant: more detailed hint
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex, bool)): return 'Num'
        if isinstance(node.value, (str, bytes)): return 'Str'
        return 'Constant'
    # cmpop,unaryop,operator,boolop: more uni hint
    if isinstance(node, (ast.cmpop, ast.unaryop, ast.operator, ast.boolop)):
        return 'op'
    # expr,stmt
    if isinstance(node, ast.expr):
        return 'expr'
    elif isinstance(node, ast.stmt):
        return 'stmt'
    return node.__class__.__name__


def _parse_tag_begin(t: str):
    if t.startswith('__') and t.endswith('_TagBegin__'):
        return t[len('__'): -len('_TagBegin__')]
    return None

def _parse_tag_end(t: str):
    if t.startswith('__') and t.endswith('_TagEnd__'):
        return t[len('__'): -len('_TagEnd__')]
    return None


def _make_tag_begin_token(tag: str):
    return f'<tag:{tag}>'


def _make_tag_end_token(tag: str):
    return f'</tag:{tag}>'


def _set_ast_tag(root: ast.AST):
    for node in ast.walk(root):
        if not isinstance(node, (ast.Expr, )):
            tag = _get_possible_ast_hint(node)
            if tag.startswith('K_'):
                node._pg_tag_begin = ast.Name(id=f'__{tag}_TagBegin__', ctx=ast.Load())
                node._pg_tag_end = ast.Name(id=f'__{tag}_TagEnd__', ctx=ast.Load())
    return root


@dataclass
class MaskedPython:
    string: str
    python_wmt_ast: ast.AST  # python_wmt (PYTHON _ With Mask Tokens)
    masked_cfs_ast: List[ast.AST]  # masked_cfs (MASKED _ Code FragmentS)

    def __hash__(self) -> int:
        return hash(self.string)
    
    def __eq__(self, value: object) -> bool:
        return isinstance(value, self.__class__) and \
                self.string == value.string
    
    def _to_debug_string(self, fn=None) -> str:
        tks = (fn or self.to_tokens)()
        in_ = ' '.join(tks['python_wmt_tokens'])
        out_ = ' <seq> '.join([
                        ' '.join(cf) for cf in tks['masked_cfs_tokens']
                        ])
        return f'IN: "{in_}"\nOUT: "{out_}"'
    
    def to_tokens(self) -> dict:
        python_wmt_tokens = _tokenize_python_string(ast_unparse(ast.fix_missing_locations(self.python_wmt_ast)))
        masked_cfs_tokens = [_tokenize_python_string(ast_unparse(ast.fix_missing_locations(c)))
                                for c in self.masked_cfs_ast]
        masks = [(i, tk) for i, tk in enumerate(python_wmt_tokens) if _is_mask_token(tk)]
        if len(masked_cfs_tokens) != len(masks):
            print('1111 1 1111>>> ', ast.dump(self.python_wmt_ast))
            print('1111 2 1111>>> ', [ast.dump(e) for e in self.masked_cfs_ast])
            print('1111 3 1111>>> ', python_wmt_tokens)
            print('1111 4 1111>>> ', masked_cfs_tokens)
            print('1111 5 1111>>> ', masks)
        assert len(masked_cfs_tokens) == len(masks)
        masked_cfs_tokens, old_masked_cfs_tokens = [None] * len(masks), masked_cfs_tokens
        for new_id, (idx, m) in enumerate(masks):  # Reorder <mask:i> by left-to-right
            old_id = _parse_mask_token(m)
            python_wmt_tokens[idx] = _make_mask_token(new_id)
            masked_cfs_tokens[new_id] = old_masked_cfs_tokens[old_id]
        return {
            'python_wmt_tokens': python_wmt_tokens,
            'masked_cfs_tokens': masked_cfs_tokens,
        }
    
    def to_itks_oasts(self) -> dict:
        python_wmt_tokens = _tokenize_python_string(ast_unparse(ast.fix_missing_locations(self.python_wmt_ast)))
        masked_cfs_asts = [_tokenize_python_string(ast.dump(c)) for c in self.masked_cfs_ast]
        masks = [(i, tk) for i, tk in enumerate(python_wmt_tokens) if _is_mask_token(tk)]
        if len(masked_cfs_asts) != len(masks):
            print('1111 1 1111>>> ', ast.dump(self.python_wmt_ast))
            print('1111 2 1111>>> ', [ast.dump(e) for e in self.masked_cfs_ast])
            print('1111 3 1111>>> ', python_wmt_tokens)
            print('1111 4 1111>>> ', masked_cfs_asts)
            print('1111 5 1111>>> ', masks)
        assert len(masked_cfs_asts) == len(masks)
        python_wmt_tokens, old_python_wmt_tokens = [], python_wmt_tokens
        for tk in old_python_wmt_tokens:
            if _is_mask_token(tk):
                tk_id = _parse_mask_token(tk)
                masked_ast = self.masked_cfs_ast[tk_id]
                python_wmt_tokens.append(tk)
                python_wmt_tokens.append(_make_mask_hint_token(masked_ast.__class__.__name__))
            else:
                python_wmt_tokens.append(tk)
        masked_cfs_asts, old_masked_cfs_asts = [None] * len(masks), masked_cfs_asts
        for new_id, (idx, m) in enumerate(masks):  # Reorder <mask:i> by left-to-right
            old_id = _parse_mask_token(m)
            python_wmt_tokens[idx] = _make_mask_token(new_id)
            masked_cfs_asts[new_id] = old_masked_cfs_asts[old_id]

        return {
            'python_wmt_tokens': python_wmt_tokens,
            'masked_cfs_tokens': masked_cfs_asts,
        }
    
    def to_tokens_with_mask_hint(self) -> dict:
        python_wmt_tokens = _tokenize_python_string(ast_unparse(ast.fix_missing_locations(self.python_wmt_ast)))
        masked_cfs_tokens = [_tokenize_python_string(ast_unparse(ast.fix_missing_locations(c)))
                                for c in self.masked_cfs_ast]
        masks = [(i, tk) for i, tk in enumerate(python_wmt_tokens) if _is_mask_token(tk)]
        if len(masked_cfs_tokens) != len(masks):
            print('1111 1 1111>>> ', ast.dump(self.python_wmt_ast))
            print('1111 2 1111>>> ', [ast.dump(e) for e in self.masked_cfs_ast])
            print('1111 3 1111>>> ', python_wmt_tokens)
            print('1111 4 1111>>> ', masked_cfs_tokens)
            print('1111 5 1111>>> ', masks)
        assert len(masked_cfs_tokens) == len(masks)
        python_wmt_tokens, old_python_wmt_tokens = [], python_wmt_tokens
        for tk in old_python_wmt_tokens:
            if _is_mask_token(tk):
                tk_id = _parse_mask_token(tk)
                masked_ast = self.masked_cfs_ast[tk_id]
                hint = _get_possible_ast_hint(masked_ast)
                python_wmt_tokens.append(tk)
                python_wmt_tokens.append(_make_mask_hint_token(hint))
            else:
                python_wmt_tokens.append(tk)
        masked_cfs_tokens, old_masked_cfs_tokens = [None] * len(masks), masked_cfs_tokens
        for new_id, (idx, m) in enumerate(masks):  # Reorder <mask:i> by left-to-right
            old_id = _parse_mask_token(m)
            python_wmt_tokens[idx] = _make_mask_token(new_id)
            masked_cfs_tokens[new_id] = old_masked_cfs_tokens[old_id]

        return {
            'python_wmt_tokens': python_wmt_tokens,
            'masked_cfs_tokens': masked_cfs_tokens,
        }

    def to_istr_ostr(self) -> dict:
        python_wmt_string = ast_unparse(ast.fix_missing_locations(self.python_wmt_ast))
        _python_wmt_tokens = _tokenize_python_string(python_wmt_string)
        _masks = [(i, tk) for i, tk in enumerate(_python_wmt_tokens) if _is_mask_token(tk)]
        # if len(self.masked_cfs_ast) != len(_masks):
        #     print('1111 1 1111>>> ', len(self.masked_cfs_ast), len(_masks))
        # if len(self.masked_cfs_ast) != 1:
        #     print('1111 2 1111>>> ', len(self.masked_cfs_ast))
        assert len(self.masked_cfs_ast) == len(_masks) and len(self.masked_cfs_ast) <= 1
        del _python_wmt_tokens, _masks
        if self.masked_cfs_ast:
            masked_cfs_string = ast_unparse(ast.fix_missing_locations(self.masked_cfs_ast[0]))
        else:
            masked_cfs_string = ""
        return {
            'python_wmt_string': python_wmt_string,
            'masked_cfs_string': masked_cfs_string,
        }


@dataclass
class MaskTemplate:
    name: str
    generator: Callable[[ast.AST], list[MaskedPython]]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, value: Union[str, object]) -> bool:
        return self.name == value or \
                (isinstance(value, self.__class__) and \
                self.name == value.name)


def _make_mtg_chain(*mtg):
    def chain(codeast: ast.AST, **kwargs):
        mask_id_start = kwargs.pop('mask_id_start', 0)
        clone_codeast = kwargs.pop('clone_codeast', True)

        if clone_codeast:
            codeast = _set_parent(_set_height(_fix_single_ins(_clone_ast(codeast))))

        masked_cfs_ast = []
        for g in mtg:
            mp = g(codeast, mask_id_start=mask_id_start, clone_codeast=False, **kwargs)
            codeast = mp.python_wmt_ast
            mask_id_start += len(mp.masked_cfs_ast)
            masked_cfs_ast.extend(mp.masked_cfs_ast)
        return MaskedPython(string=ast_unparse(codeast),
                            python_wmt_ast=codeast,
                            masked_cfs_ast=masked_cfs_ast)
    return chain


def _make_mask_node(masked_node: ast.AST) -> Callable[[ast.Name], ast.AST]:
    def _is_cannotbe_masked_ast(node):
        return _check_parent_chain(node,
                                   checker=lambda p: isinstance(p, ast.JoinedStr),
                                   mode='any',
                                   with_self=False) or \
              (isinstance(node, ast.Name) and _is_mask_marker(node.id))
    if _is_cannotbe_masked_ast(masked_node):
        raise ValueError(f'Don\'t support mask the `{masked_node}`')
    elif isinstance(masked_node, ast.expr):
        return lambda m: m
    elif isinstance(masked_node, ast.stmt):
        return lambda m: ast.Expr(value=m)
    elif isinstance(masked_node, ast.mod):
        return lambda m: ast.Module(body=[ast.Expr(value=m)], type_ignores=[])
    elif isinstance(masked_node, (ast.boolop, ast.operator, ast.unaryop, ast.cmpop)):
        raise NotImplementedError('Don\'t use this now!')
    elif isinstance(masked_node, ast.arg):
        return lambda m: ast.arg(arg=m.id)
    elif isinstance(masked_node, ast.keyword):
        return lambda m: m
    elif isinstance(masked_node, ast.alias):
        raise NotImplementedError('Don\'t use this now!')
    elif isinstance(masked_node, ast.withitem):
        raise NotImplementedError('Don\'t use this now!')
    elif isinstance(masked_node, ast.match_case):
        raise NotImplementedError('Don\'t use this now!')
    elif isinstance(masked_node, (ast.expr_context,
                                  ast.comprehension,
                                  ast.excepthandler,
                                  ast.arguments,
                                  ast.pattern,
                                  ast.type_ignore)):
        raise ValueError(f'Don\'t support mask the `{masked_node}`')
    raise NotImplementedError('unreachable')


def _random_mask_subtree(codeast: ast.AST,
                             mask_id_start: int,
                             max_height_factor: float,
                             max_selected_num_factor: float,
                             max_selected_num: int,
                             clone_codeast: bool = True,
                             **kwargs) -> MaskedPython:
    # `Replace With Of Ast [for] <ast-cls>` functions
    def _rwoa_alias(n: ast.alias, m: str) -> Tuple[ast.alias, ast.alias]:
        return ast.alias(name=m), n
    
    def _rwoa_arg(n: ast.arg, m: str) -> Tuple[ast.arg, ast.arg]:
        ##NOTE: nomask n.annotation
        return ast.arg(arg=m), n

    def _rwoa_arguments(n: ast.arguments, m: str) -> Tuple[ast.arguments, ast.arguments]:
        return ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=m)],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]), n

    # def _rwoa_generic_op(n: Any, m: str) -> Tuple[FakeOp, Any]:
    #     return FakeOp(op=m, precedence=_get_op_precedence(n)), n

    # _rwoa_boolop = _rwoa_generic_op
    # _rwoa_cmpop = _rwoa_generic_op
    # _rwoa_operator = _rwoa_generic_op

    # def _rwoa_unaryop(n: ast.unaryop, m: str) -> Tuple[FakeOp, ast.unaryop]:
    #     prec = _get_op_precedence(n)
    #     if prec == Precedence.FACTOR:
    #         prec = -int(Precedence.FACTOR)  # __mask__ A instead of __mask__A
    #     return FakeOp(op=m, precedence=prec), n

    def _rwoa_expr(n: ast.expr, m: str) -> Tuple[ast.Name, ast.expr]:
        ctx = getattr(n, 'ctx', None) or ast.Load()
        return ast.Name(id=m, ctx=ctx), n
    
    def _rwoa_keyword(n: ast.keyword, m: str) -> Tuple[ast.Name, ast.keyword]:
        return ast.Name(id=m, ctx=ast.Load()), n

    def _rwoa_stmt(n: ast.stmt, m: str) -> Tuple[ast.Expr, ast.stmt]:
        return ast.Expr(value=ast.Name(id=m, ctx=ast.Load())), n
    
    def _rwoa_withitem(n: ast.withitem, m: str) -> Tuple[ast.Name, ast.withitem]:
        return ast.Name(id=m, ctx=ast.Load()), n

    rwoa_fn_table = {n: fn for n, fn in locals().copy().items() if n.startswith('_rwoa_')}

    def _resolve_rwoa_func(x, fn_table):
        for cls in x.__class__.__mro__:
            try:
                return eval(f'_rwoa_{cls.__name__}', fn_table)
            except:
                pass
        raise NotImplementedError('un...')

    CANBE_REPLACED_AST = (ast.alias,
                          ast.arg,
                          ast.arguments,
                        #   ast.boolop,
                        #   ast.cmpop,
                        #   ast.operator,
                        #   ast.unaryop,
                          ast.expr,
                          ast.keyword,
                          ast.stmt,
                          ast.withitem)

    def _is_cannotbe_replaced_ast(node):
        return _check_parent_chain(node,
                                   checker=lambda p: isinstance(p, ast.JoinedStr),
                                   mode='any',
                                   with_self=False) or \
              (isinstance(node, ast.Name) and _is_mask_marker(node.id))

    if clone_codeast:
        codeast = _set_parent(_set_height(_fix_single_ins(_clone_ast(codeast))))

    max_h = int(max_height_factor * codeast._pg_height)
    possible_nodes = [node for node in ast.walk(codeast)
                        if isinstance(node, CANBE_REPLACED_AST) and \
                            not _is_cannotbe_replaced_ast(node) and \
                            node._pg_height <= max_h]

    masked_cfs = None
    if possible_nodes:
        max_s_num = min(int(len(possible_nodes) * max_selected_num_factor), max_selected_num)
        s_num = random.randint(0, max_s_num)
        nodes = list(_random_sample_n_subtrees(possible_nodes, k=s_num))
        replace_node_map = {}
        masked_cfs = [None] * len(nodes)
        for i, n in enumerate(nodes):
            rwoa_f = _resolve_rwoa_func(n, rwoa_fn_table)
            replace_node_map[n], masked_cfs[i] = rwoa_f(n, _make_mask_marker(i + mask_id_start))
        codeast = ReplaceXWithY(x_cls_names=[n.__class__.__name__ for n in replace_node_map.keys()],
                                is_x=lambda x: x in replace_node_map,
                                make_y=lambda x: replace_node_map.pop(x)).visit(codeast) 
        codeast = ast.fix_missing_locations(codeast)

    return MaskedPython(string=ast_unparse(codeast),
                        python_wmt_ast=codeast,
                        masked_cfs_ast=masked_cfs or [])


def _random_mask_nil(codeast: ast.AST,
                         mask_id_start: int,
                         max_inserted_num_factor: float,
                         max_inserted_num: int,
                         clone_codeast: bool = True,
                         **kwargs) -> MaskedPython:
    DUMMY = id(codeast)  # any value

    if clone_codeast:
        codeast = _set_parent(_fix_single_ins(_clone_ast(codeast)))

    _mask_id_counter = -1
    def make_mask_id():
        nonlocal _mask_id_counter
        _mask_id_counter += 1
        return _mask_id_counter
    
    def make_inserter(L: list, mk_val):
        def I():
            i = random.randint(0, len(L))
            L.insert(i, mk_val())
        return I

    def _is_cannotbe_inserted_ast(node):
        return _check_parent_chain(node,
                                   checker=lambda p: isinstance(p, ast.JoinedStr),
                                   mode='any',
                                   with_self=False) or \
              (isinstance(node, ast.Name) and _is_mask_marker(node.id))

    possible_inserters = []
    possible_containers = [node for node in ast.walk(codeast)
                                if _is_block_stmt(node) or isinstance(node, (ast.Call, ast.Tuple, ast.List, ast.Set))
                                if not _is_cannotbe_inserted_ast(node)]
    possible_containers = list(_random_sample_n_subtrees(possible_containers, k=len(possible_containers)))
    total_insert_points = 0
    make_mask_stmt = lambda: ast.Expr(ast.Name(id=_make_mask_marker(make_mask_id() + mask_id_start), ctx=ast.Load()))
    make_mask_expr = lambda: ast.Name(id=_make_mask_marker(make_mask_id() + mask_id_start), ctx=ast.Load())
    for c in possible_containers:
        if _is_block_stmt(c):
            if hasattr(c, 'body'):
                possible_inserters.append(make_inserter(c.body, make_mask_stmt))
                total_insert_points += len(c.body) + 1
            if hasattr(c, 'orelse'):
                possible_inserters.append(make_inserter(c.orelse, make_mask_stmt))
                total_insert_points += len(c.orelse) + 1
            if hasattr(c, 'finalbody'):
                possible_inserters.append(make_inserter(c.finalbody, make_mask_stmt))
                total_insert_points += len(c.finalbody) + 1
        elif isinstance(c, ast.Call):
            possible_inserters.append(make_inserter(c.args, make_mask_expr))
            total_insert_points += len(c.args) + 1
            possible_inserters.append(make_inserter(c.keywords, make_mask_expr))
            total_insert_points += len(c.keywords) + 1
        elif isinstance(c, (ast.Tuple, ast.List, ast.Set)):
            possible_inserters.append(make_inserter(c.elts, make_mask_expr))
            total_insert_points += len(c.elts) + 1

    max_inserted_num = min(max_inserted_num, int(max_inserted_num_factor * total_insert_points))
    i_num = random.randint(0, max_inserted_num)

    inserter_index = 0
    while _mask_id_counter + 1 < i_num:
        if random.randint(0, 1) == 0:
            possible_inserters[inserter_index]()
        inserter_index = (inserter_index + 1) % len(possible_inserters)

    return MaskedPython(string=ast_unparse(codeast),
                        python_wmt_ast=codeast,
                        masked_cfs_ast=[ast.Name(id='__nil__', ctx=ast.Load())] * (_mask_id_counter + 1))


class MaskMaker:
    def __init__(self, mask_id_start):
        self.mask_id_start = mask_id_start
        self.mask_id_counter = -1

    def make_mask_node(self):
        self.mask_id_counter += 1
        return ast.Name(id=_make_mask_marker(self.mask_id_counter + self.mask_id_start),
                        ctx=ast.Load())

    def num_masks(self):
        return self.mask_id_counter + 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.mask_id_counter = -1


_glb_mask_templates = {}
def _reg_mask_template(name: str):
    def inner(fn):
        @functools.wraps(fn)
        def wrapped(codeast: ast.AST,
                    mask_id_start: int = 0,
                    clone_codeast: bool = True,
                    **kwargs):
            mask_maker = MaskMaker(mask_id_start)
            if clone_codeast:
                codeast = _fix_single_ins(_clone_ast(codeast))
            return fn(codeast,
                      mask_maker=mask_maker,
                      **kwargs)
        _glb_mask_templates[name] = MaskTemplate(name=name,
                                                 generator=wrapped)
        return wrapped
    return inner


def _mask_t(name: str):
    return _glb_mask_templates[name]


class SetResetHelper:
    def __init__(self, fn_list: List[Callable[[Any], Any]], new_values):
        assert len(fn_list) == len(new_values)
        self.fn_list = fn_list
        self.new_values = new_values
        self.old_values = None

    def _set(self, values):
        assert len(self.fn_list) == len(values)
        old_values = [None] * len(values)
        for i, (fn, val) in enumerate(zip(self.fn_list, values)):
            if val is not None:
                old_values[i] = fn(val)
        return old_values

    def __enter__(self):
        self.old_values = self._set(self.new_values)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_values:
            self._set(self.old_values)


# T (temp for testing)
@_reg_mask_template("T_temp_act")
def _replace_act_arg_with_mask(codeast: ast.AST, mask_maker: MaskMaker, **kwargs) -> list[MaskedPython]:
    #1. activation=xxx
    #2. __root__.keras.layers.Activation(xxx)
    calls: list[ast.Call] = _find_x(codeast, is_x=lambda x: isinstance(x, ast.Call))
    setreset_fns = []
    for c in calls:
        if (act := tuple(k for k in c.keywords if k.arg == 'activation')):
            assert len(act) == 1
            def make_fn(kw):
                def fn(val):
                    old_val, kw.value = kw.value, val
                    return old_val
                return fn
            setreset_fns.append(make_fn(act[0]))
        elif isinstance(c.func, ast.Attribute) and c.func.attr == 'Activation':
            if len(c.args) == 1:
                def make_fn(args):
                    def fn(val):
                        old_val, args[0] = args[0], val
                        return old_val
                    return fn
                setreset_fns.append(make_fn(c.args))
    masked_data: List[MaskedPython] = []
    for slot in range(len(setreset_fns)):
        with mask_maker as mm:
            with SetResetHelper(setreset_fns, [mm.make_mask_node() if i == slot else None \
                                                    for i in range(len(setreset_fns))]) as h:
                    masked_data.append(MaskedPython(string=ast_unparse(codeast),
                                                    python_wmt_ast=_clone_ast(codeast),
                                                    masked_cfs_ast=[_clone_ast(h.old_values[slot])]))
            assert mm.num_masks() <= 1
    return masked_data


def _replace_arg_with_mask(codeast: ast.AST,
                           mask_maker: MaskMaker,
                           is_call: Callable[[ast.Call], bool],
                           cared_arg_indexes: List[int],
                           cared_arg_keys: List[str]) -> list[MaskedPython]:
    calls: list[ast.Call] = _find_x(codeast, is_x=is_call)
    setreset_fns = []
    for c in calls:
        for i, a in enumerate(c.args):
            if i in cared_arg_indexes:
                def make_fn(args, i):
                    def fn(val):
                        old_val, args[i] = args[i], val
                        return old_val
                    return fn
                setreset_fns.append(make_fn(c.args, i))
        for k in c.keywords:
            if k.arg in cared_arg_keys:
                def make_fn(keywords, k):
                    def fn(val):
                        old_val, k.value = k.value, val
                        return old_val
                    return fn
                setreset_fns.append(make_fn(c.keywords, k))
    masked_data: List[MaskedPython] = []
    for slot in range(len(setreset_fns)):
        with mask_maker as mm:
            with SetResetHelper(setreset_fns, [mm.make_mask_node() if i == slot else None \
                                                    for i in range(len(setreset_fns))]) as h:
                    masked_data.append(MaskedPython(string=ast_unparse(codeast),
                                                    python_wmt_ast=_clone_ast(codeast),
                                                    masked_cfs_ast=[_clone_ast(h.old_values[slot])]))
            assert mm.num_masks() <= 1
    return masked_data


def _replace_node_with_mask(codeast: ast.AST,
                            mask_maker: MaskMaker,
                            is_node: Callable[[ast.AST], bool],
                            max_generate_num: int = None,
                            **kwargs) -> list[MaskedPython]:
    codeast = _set_node_id(_set_parent(codeast))
    nodes: list[ast.AST] = _find_x(codeast, is_x=is_node)
    if max_generate_num is not None:
        nodes = random.sample(nodes, min(max_generate_num, len(nodes)))
    masked_data: List[MaskedPython] = []
    for n in nodes:
        with mask_maker as mm:
            try:
                mask_node = _make_mask_node(n)(mm.make_mask_node())
            except (NotImplementedError, ValueError):
                continue # Skip this node
            mcodeast = ReplaceXWithY(x_cls_names=[n.__class__.__name__],
                                     is_x=lambda x: x._pg_node_id == n._pg_node_id,
                                     make_y=lambda x: mask_node).visit(_set_node_id(_clone_ast(codeast)))
            masked_data.append(MaskedPython(string=ast_unparse(mcodeast),
                                            python_wmt_ast=mcodeast,
                                            masked_cfs_ast=[_clone_ast(n)]))
            assert mm.num_masks() <= 1
    return masked_data


@_reg_mask_template("T_temp_all_keras_components")
def _replace_keras_component_with_mask(codeast: ast.AST, mask_maker: MaskMaker, *, max_generate_num, **kwargs) -> list[MaskedPython]:
    from copy import deepcopy

    codeast = _set_node_id(_set_parent(codeast))
    k_node_ids = set()  # {(id, node), ...}
    for node in ast.walk(codeast):
        if _mhint_is_keras_component_hint(mhint := _get_possible_ast_hint(node)):
            assert node._pg_node_id not in k_node_ids
            # k_node_ids.update({*map(lambda n: (n._pg_node_id, n), ast.walk(node))})
            k_node_ids.add((mhint, node._pg_node_id, node))
    if max_generate_num is not None:
        k_node_ids = random.sample(sorted(k_node_ids), min(max_generate_num, len(k_node_ids)))
    masked_data: List[MaskedPython] = []
    for hint, i, n in k_node_ids:
        with mask_maker as mm:
            try:
                mask_node = _make_mask_node(n)(mm.make_mask_node())
            except (NotImplementedError, ValueError):
                continue  # Skip this node
            mcodeast = ReplaceXWithY(x_cls_names=[n.__class__.__name__],
                    is_x=lambda x: x._pg_node_id == i,
                    make_y=lambda x: mask_node).visit(_set_node_id(_clone_ast(codeast)))
            masked_data.append(MaskedPython(string=ast_unparse(mcodeast),
                                            python_wmt_ast=mcodeast,
                                            masked_cfs_ast=[_clone_ast(n)]))
            assert mm.num_masks() <= 1
    return masked_data


@_reg_mask_template("T_mask_keras_metric_arg")
def _replace_keras_metric_arg_with_mask(codeast: ast.AST, mask_maker: MaskMaker, **kwargs) -> list[MaskedPython]:
    #1. ...compile(...metrics=xxx...), keys=['metrics']
    return _replace_arg_with_mask(codeast,
                                  mask_maker,
                                  is_call=lambda n: _check_call_member_fn('compile', n),
                                  cared_arg_indexes=[],
                                  cared_arg_keys=['metrics'])


@_reg_mask_template("T_mask_keras_optimizer_arg")
def _replace_keras_optimizer_arg_with_mask(codeast: ast.AST, mask_maker: MaskMaker, **kwargs) -> list[MaskedPython]:
    #1. ...compile(...optimizer=<opt>...), keys=['optimizer']
    #2. ...compile(<opt>, ...), indexes=[0]
    return _replace_arg_with_mask(codeast,
                                  mask_maker,
                                  is_call=lambda n: _check_call_member_fn('compile', n),
                                  cared_arg_indexes=[0],
                                  cared_arg_keys=['optimizer'])


@_reg_mask_template("T_mask_keras_epochs_arg")
def _replace_keras_epochs_arg_with_mask(codeast: ast.AST, mask_maker: MaskMaker, **kwargs) -> list[MaskedPython]:
    #1. ...fit(...epochs=<epochs>...), keys=['epochs']
    #2. ...fit(...nb_epoch=<epochs>...), keys=['nb_epoch']
    #3. ...fit(<0>, <1>, <2>, <epochs>, ...), indexes=[3]
    return _replace_arg_with_mask(codeast,
                                  mask_maker,
                                  is_call=lambda n: _check_call_member_fn('fit', n),
                                  cared_arg_indexes=[3],
                                  cared_arg_keys=['epochs', 'nb_epoch'])


@_reg_mask_template("T_mask_keras_loss_arg")
def _replace_keras_loss_arg_with_mask(codeast: ast.AST, mask_maker: MaskMaker, **kwargs) -> list[MaskedPython]:
    #1. ...compile(...loss=<loss>...), keys=['loss']
    #2. ...compile(<0>, <loss>, ...), indexes=[1]
    return _replace_arg_with_mask(codeast,
                                  mask_maker,
                                  is_call=lambda n: _check_call_member_fn('compile', n),
                                  cared_arg_indexes=[1],
                                  cared_arg_keys=['loss'])


@_reg_mask_template("T_mask_keras_activation_arg")
def _replace_keras_activation_arg_with_mask(codeast: ast.AST, mask_maker: MaskMaker, **kwargs) -> list[MaskedPython]:
    #1. ...<Layer>(...activation=<activation>...), keys=['activation']
    #2. ...Activation(<activation>), keys=['activation'], indexes=[0]
    in_Layer = _replace_arg_with_mask(codeast,
                                      mask_maker,
                                      is_call=lambda n: isinstance(n, ast.Call) and _get_possible_ast_hint(n) == _MHINT_K_LAYER,
                                      cared_arg_indexes=[],
                                      cared_arg_keys=['activation'])
    in_Activation = _replace_arg_with_mask(codeast,
                                           mask_maker,
                                           is_call=lambda n: _check_call_member_fn('Activation', n),
                                           cared_arg_indexes=[0],
                                           cared_arg_keys=['activation'])
    return in_Layer + in_Activation


@_reg_mask_template("T_mask_keras_initializer_arg")
def _replace_keras_initializer_arg_with_mask(codeast: ast.AST, mask_maker: MaskMaker, **kwargs) -> list[MaskedPython]:
    #1. ...<Layer>(...kernel_initializer=<initializer>...), keys=['kernel_initializer']
    #2. ...<Layer>(...bias_initializer=<initializer>...), keys=['bias_initializer']
    #3. ...<Layer>(...init=<initializer>...), keys=['init']
    return _replace_arg_with_mask(codeast,
                                  mask_maker,
                                  is_call=lambda n: isinstance(n, ast.Call) and _get_possible_ast_hint(n) == _MHINT_K_LAYER,
                                  cared_arg_indexes=[],
                                  cared_arg_keys=['kernel_initializer', 'bias_initializer', 'init'])


@_reg_mask_template("T_mask_keras_Layer")
def _replace_keras_layer_with_mask(codeast: ast.AST, mask_maker: MaskMaker, **kwargs) -> list[MaskedPython]:
    #1. ...<Layer>(...)
    return _replace_node_with_mask(codeast,
                                   mask_maker,
                                   is_node=lambda n: isinstance(n, ast.Call) and _get_possible_ast_hint(n) == _MHINT_K_LAYER,
                                   **kwargs)


@_reg_mask_template("T_mask_keras_learning_rate_arg")
def _replace_learning_rate_arg_with_mask(codeast: ast.AST, mask_maker: MaskMaker, **kwargs) -> list[MaskedPython]:
    #1. ...<Optimizer>(...learning_rate=<lr>...), keys=['learning_rate']
    #2. ...<Optimizer>(...lr=<lr>...), keys=['lr']
    #3. ...<Optimizer>(<lr>), indexes=[0]
    return _replace_arg_with_mask(codeast,
                                  mask_maker,
                                  is_call=lambda n: isinstance(n, ast.Call) and _get_possible_ast_hint(n) == _MHINT_K_OPTIMIZER,
                                  cared_arg_indexes=[0],
                                  cared_arg_keys=['learning_rate', 'lr'])


@_reg_mask_template("T_mask_keras_batch_size_arg")
def _replace_batch_size_arg_with_mask(codeast: ast.AST, mask_maker: MaskMaker, **kwargs) -> list[MaskedPython]:
    #1. ...fit(...batch_size=<batch_size>...), keys=['batch_size']
    #3. ...fit(<0>, <1>, <batch_size>, ...), indexes=[2]
    return _replace_arg_with_mask(codeast,
                                  mask_maker,
                                  is_call=lambda n: _check_call_member_fn('fit', n),
                                  cared_arg_indexes=[2],
                                  cared_arg_keys=['batch_size'])


## args of keras.Model.compile
_MT_mask_keras_metric_arg: MaskTemplate = _mask_t('T_mask_keras_metric_arg')
_MT_mask_keras_learning_rate_arg: MaskTemplate = _mask_t('T_mask_keras_learning_rate_arg')
_MT_mask_keras_optimizer_arg: MaskTemplate = _mask_t('T_mask_keras_optimizer_arg')
_MT_mask_keras_loss_arg: MaskTemplate = _mask_t('T_mask_keras_loss_arg')
## args of keras.Model.fit
_MT_mask_keras_epochs_arg: MaskTemplate = _mask_t('T_mask_keras_epochs_arg')
_MT_mask_keras_batch_size_arg: MaskTemplate = _mask_t('T_mask_keras_batch_size_arg')
## keras.layers.Layer / args of keras.layers.Layer
_MT_mask_keras_Layer: MaskTemplate = _mask_t('T_mask_keras_Layer')
_MT_mask_keras_activation_arg: MaskTemplate = _mask_t('T_mask_keras_activation_arg')
_MT_mask_keras_initializer_arg: MaskTemplate = _mask_t('T_mask_keras_initializer_arg')


def _generate_masked_python(codeast: ast.AST,
                            templates: List[MaskTemplate],
                            expected_number_of_data_generated_by_templ: List[int],
                            max_tries_per_templ: int, 
                            **ext_kwargs) -> Dict[str, List[MaskedPython]]:
    result = {}
    for templ, num in zip(templates, expected_number_of_data_generated_by_templ):
        ext_kwargs_of_t = {(k[len(templ.name)+1:] if k.startswith(f'{templ.name}_') else k): v \
                                for k, v in ext_kwargs.items()}
        result_of_t = set()
        for _ in range(max_tries_per_templ):
            elts = templ.generator(codeast, **ext_kwargs_of_t)
            if isinstance(elts, (list, tuple, set)):
                result_of_t.update(elts)
            else:
                result_of_t.add(elts)
            if len(result_of_t) >= num:
                break
        result[templ.name] = list(result_of_t)
    return result


def _generate_masked_python_from_string(string, 
                                        templates: List[MaskTemplate],
                                        expected_numbers: List[int],
                                        max_tries_per_templ: int, 
                                        **ext_kwargs):
    return _generate_masked_python(codeast=ast.parse(string),
                                   templates=templates,
                                   expected_number_of_data_generated_by_templ=expected_numbers,
                                   max_tries_per_templ=max_tries_per_templ,
                                   **ext_kwargs)


def _debug_print_generated_masked_python(m: Dict[MaskTemplate, List[MaskedPython]], fn = None, fp: TextIO = None):
    _print = lambda *args, **kwargs: print(*args, **kwargs, file=fp or sys.stdout)
    for k, mp_list in m.items():
        _print(f'==> Templ: {k}')
        for m in mp_list:
            _print(m._to_debug_string(fn=None if not fn else getattr(m, fn)))


def _trans_to_mask_line(_python_wmt_string: str, _masked_cfs_string: str):
    python_wmt_string = _python_wmt_string
    masked_cfs_string = _masked_cfs_string
    line_with_mask0 = list(
        filter(lambda x: "__mask_0__" in x, python_wmt_string.splitlines())
    )
    assert len(line_with_mask0) == 1
    assert line_with_mask0[0].count("__mask_0__") == 1
    line_with_mask0 = line_with_mask0[0]  # NOTE: DO NOT `strip()`
    python_wmt_string = python_wmt_string.replace(line_with_mask0, "__mask_0__")
    masked_cfs_string = line_with_mask0.replace("__mask_0__", masked_cfs_string)
    assert python_wmt_string.replace("__mask_0__", masked_cfs_string) == \
            _python_wmt_string.replace("__mask_0__", _masked_cfs_string)
    return python_wmt_string, masked_cfs_string
