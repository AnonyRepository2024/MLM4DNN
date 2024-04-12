import os
import io
import re
import sys
import ast
import glob
import json
import pickle
import argparse


try:
    import pyan
except ImportError:
    pyan = None


class R2MSError(Exception):
    pass


#+ Uni utils
def _to_snake_case(name):
    name = re.sub(r"\W+", "", name)
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub("([a-z])([A-Z])", r"\1_\2", name).lower()
    return name

if pyan is not None:
    #+ Call-Graph utils (using pyan)
    def _pyan_get_module_name(filename, root: str = None):
        from pyan.anutils import get_module_name
        return get_module_name(os.path.abspath(filename), root=root)

    def _pyan_analysis_call(
        filenames: list[str] | str,
        root: str = None,
        function: str = None,
        namespace: str = None,
        max_iter: int = 1000,
    ) -> pyan.analyzer.CallGraphVisitor:
        if isinstance(filenames, str):
            filenames = [filenames]
        elif not isinstance(filenames, (list, tuple)):
            filenames = list(filenames)

        v = pyan.analyzer.CallGraphVisitor(filenames, root=root)
        if function or namespace:
            if function:
                function_name = function.split(".")[-1]
                function_namespace = ".".join(function.split(".")[:-1])
                node = v.get_node(function_namespace, function_name)
            else:
                node = None
            v.filter(node=node, namespace=namespace, max_iter=max_iter)
        return v


    def _pyan_build_call_graph(
        visitor: pyan.analyzer.CallGraphVisitor
    ) -> dict:
        """Return {node_table:, uses_from2to:, uses_to2from:, defines_from2to:, defines_to2from:, name2nodes:}"""
        node_table = [e for name, n in visitor.nodes.items() for e in n]
        node2index = {n: i for i, n in enumerate(node_table)}
        
        uses_from2to, uses_to2from = {}, {}  # int => [int, ...]
        for f_node, t_nodes in visitor.uses_edges.items():
            f = node2index[f_node]
            if f not in uses_from2to:
                uses_from2to[f] = []
            for t_node in t_nodes:
                t = node2index[t_node]
                if t not in uses_to2from:
                    uses_to2from[t] = []
                uses_from2to[f].append(t)
                uses_to2from[t].append(f)
        
        defines_from2to, defines_to2from = {}, {}  # int => [int, ...]
        for f_node, t_nodes in visitor.defines_edges.items():
            f = node2index[f_node]
            if f not in defines_from2to:
                defines_from2to[f] = []
            for t_node in t_nodes:
                t = node2index[t_node]
                if t not in defines_to2from:
                    defines_to2from[t] = []
                defines_from2to[f].append(t)
                defines_to2from[t].append(f)

        name2nodes = {name: [node2index[n] for n in nodes] for name, nodes in visitor.nodes.items()}

        return {
            'node_table': node_table,
            'uses_from2to': uses_from2to,
            'uses_to2from': uses_to2from,
            'defines_from2to': defines_from2to,
            'defines_to2from': defines_to2from,
            'name2nodes': name2nodes,
        }


    def _pyan_draw_call_graph(
            visitor: pyan.analyzer.CallGraphVisitor,
            format: str = "svg",
            rankdir: str = "LR",
            nested_groups: bool = True,
            draw_defines: bool = True,
            draw_uses: bool = True,
            colored: bool = True,
            grouped_alt: bool = False,
            annotated: bool = False,
            grouped: bool = True) -> str:
        """
        Draw callgraph from CallGraphVisitor

        Args:
            rankdir: direction of graph, e.g. "LR" for horizontal or "TB" for vertical
            nested_groups: if to group by modules and submodules
            draw_defines: if to draw defines edges (functions that are defines)
            draw_uses: if to draw uses edges (functions that are used)
            colored: if to color graph
            grouped_alt: if to use alternative grouping
            annotated: if to annotate graph with filenames
            grouped: if to group by modules

        Returns:
            str: callgraph
        """

        graph_options = {
            "draw_defines": draw_defines,
            "draw_uses": draw_uses,
            "colored": colored,
            "grouped_alt": grouped_alt,
            "grouped": grouped,
            "nested_groups": nested_groups,
            "annotated": annotated,
        }

        graph = pyan.visgraph.VisualGraph.from_visitor(visitor, options=graph_options)

        stream = io.StringIO()
        if format == "dot":
            writer = pyan.writers.DotWriter(graph, options=["rankdir=" + rankdir], output=stream)
            writer.run()

        elif format == "html":
            writer = pyan.writers.HTMLWriter(graph, options=["rankdir=" + rankdir], output=stream)
            writer.run()

        elif format == "svg":
            writer = pyan.writers.SVGWriter(graph, options=["rankdir=" + rankdir], output=stream)
            writer.run()
        else:
            raise ValueError(f"format {format} is unknown")

        return stream.getvalue()


    def _pyan_find_callers_by_nodes(possible_nodes, cg: dict, transform=True) -> list:
        node_table = cg['node_table']
        uses_to2from = cg['uses_to2from']
        try:
            possible_callers = [caller for n in possible_nodes if n in uses_to2from for caller in uses_to2from[n]]
        except KeyError:
            return []
        return possible_callers if not transform else [node_table[n] for n in possible_callers]


    def _pyan_find_callers(namespace: str, funcname: str, cg: dict, transform=True) -> list:
        node_table = cg['node_table']
        name2nodes = cg['name2nodes']
        checker = lambda n: (namespace is None or n.namespace == namespace) and (n.name == funcname)
        try:
            possible_nodes = [n for n in name2nodes[funcname] if checker(node_table[n])]
            return _pyan_find_callers_by_nodes(possible_nodes, cg, transform=transform)
        except KeyError:
            return []


    def _pyan_find_callers_recursively(namespace: str, funcname: str, max_tries: int, cg: dict, transform=True) -> list:
        node_table = cg['node_table']
        trans_fn = lambda n: node_table[n]
        callers = _pyan_find_callers(namespace, funcname, cg, transform=False)
        all_callers = {*callers}
        for _ in range(max_tries):
            callers = set(_pyan_find_callers_by_nodes(callers, cg, transform=False))
            if not callers or callers.issubset(all_callers): break
            all_callers.update(callers)
        return list(all_callers) if not transform else [trans_fn(n) for n in all_callers]


    class PyanCallGraph:
        def __init__(self, cg: dict) -> None:
            self.__cg = cg

        def save_as_pkl(self, filename: str):
            with open(filename, 'wb') as fp:
                pickle.dump(self.__cg, fp)

        def to_json(self) -> dict:
            cg = self.__cg.copy()
            make_n = lambda n: {
                'namespace': str(n.namespace),
                'name': str(n.name),
                # 'ast_node': n.ast_node,
                'filename': str(n.filename),
                'flavor': str(n.flavor.value),
                'defined': bool(n.defined)}
            cg['node_table'] = [make_n(n) for n in cg['node_table']]
            return cg

        def save_as_json(self, filename: str):
            with open(filename, 'w', encoding='UTF-8') as fp:
                json.dump(self.to_json(), fp)

        def find_callers(self, namespace: str, funcname: str, transform=True) -> list:
            return _pyan_find_callers(namespace, funcname, self.__cg, transform)

        def find_callers_r(self, namespace: str, funcname: str, max_tries: int = 128, transform=True) -> list:
            return _pyan_find_callers_recursively(namespace, funcname, max_tries, self.__cg, transform)

        def get_entrys(self, functions: list, max_find_callers_tries: int = 128):
            entrys = []
            for fn in functions:
                if not fn.name and fn.namespace:
                    fn.name, fn.namespace = fn.namespace, fn.name
                for caller in self.find_callers_r(namespace=fn.namespace,
                                                    funcname=fn.name,
                                                    max_tries=max_find_callers_tries):
                    if caller.flavor == pyan.analyzer.Flavor.MODULE:
                        entrys.append(caller)
            return entrys

        @classmethod
        def from_visitor(cls, visitor: pyan.analyzer.CallGraphVisitor):
            return cls(_pyan_build_call_graph(visitor))
        
        @classmethod
        def from_pkl(cls, filename: str):
            with open(filename, 'rb') as fp:
                cg = pickle.load(fp)
            assert isinstance(cg, dict)
            assert 'node_table' in cg
            assert 'uses_from2to' in cg
            assert 'uses_to2from' in cg
            assert 'defines_from2to' in cg
            assert 'defines_to2from' in cg
            assert 'name2nodes' in cg
            return cls(cg)

        @classmethod
        def from_json(cls, *args):
            raise NotImplemented('Use from_pkl')

#+ AST utils

class NodeVisitorInRunOrder(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.__current_cnt = -1

    @property
    def current_cnt(self):
        return self.__current_cnt

    def visit(self, node):
        self.__current_cnt += 1
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        for field, value in _iter_fields_in_run_order(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)


class NodeTransformerInRunOrder(NodeVisitorInRunOrder):
    def __init__(self):
        super().__init__()
    
    def generic_visit(self, node):
        for field, old_value in _iter_fields_in_run_order(node):
            if isinstance(old_value, list):
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node    


class ImportedModule:
    def __init__(self, fullname: str, module: ast.Module, symbol_map: dict | None = None) -> None:
        self.fullname = fullname
        self.module = module
        self.symbol_map = symbol_map  # ori => renamed

    @classmethod
    def make(cls, fullname: str, module: ast.Module, symbol_map: dict | None = None):
        if fullname is None or module is None: return None
        return cls(fullname, module, symbol_map)


class ModuleTable:
    def __init__(self, roots: list) -> None:
        self.__roots = roots
        self.__modules = {}  # fullname => ast.Module

    @property
    def all_module_glbs(self) -> set:
        return {glb for m in self.__modules.values() if (m and m.symbol_map) for glb in m.symbol_map.values()}

    def __find_module(self, module_name: str) -> ImportedModule | None:
        if module_name not in self.__modules:
            module_relpath = '/'.join(module_name.split('.'))
            module: ast.Module | None = None
            for root in self.__roots:
                if os.path.isfile(mpy := f'{root}/{module_relpath}.py'):
                    try:
                        with open(mpy, 'r', encoding='UTF-8') as fp:
                            code = fp.read()
                        module = ast.parse(code)
                        break
                    except (UnicodeDecodeError, FileNotFoundError, SyntaxError):
                        continue
                elif os.path.isdir(mdir := f'{root}/{module_relpath}') and \
                        os.path.isfile(mpy := f'{mdir}/__init__.py'):
                    try:
                        with open(mpy, 'r', encoding='UTF-8') as fp:
                            code = fp.read()
                        module = ast.parse(code)
                        break
                    except (UnicodeDecodeError, FileNotFoundError, SyntaxError):
                        continue
            self.__modules[module_name] = ImportedModule.make(module_name, module)
        # print(f'Find module: {module_name}, {self.__modules[module_name]}')
        return self.__modules[module_name]
    
    def __process_import(self, import_: ast.Import):
        assert len(import_.names) == 1
        imported_module = import_.names[0].name
        return self.__find_module(imported_module)

    def __process_import_from(self, import_: ast.ImportFrom):
        assert len(import_.names) == 1
        if not import_.module or import_.level != 0:
            return None
        imported_module = import_.module
        imported_symbol = import_.names[0].name
        try_import_module_1 = f'{imported_module}.{imported_symbol}'
        try_import_module_2 = imported_module
        return self.__find_module(try_import_module_1) or \
                self.__find_module(try_import_module_2)

    def lookup(self, import_) -> ImportedModule:
        assert isinstance(import_, (ast.Import, ast.ImportFrom))
        if isinstance(import_, ast.Import):
            return self.__process_import(import_)
        elif isinstance(import_, ast.ImportFrom):
            return self.__process_import_from(import_)

    @classmethod
    def make(cls, roots: list):
        return cls(roots)


# ToDo-List:
## [x] 1. Functions defined in the code && name=ast.Name(...) (def f(...; f(...))
## [ ] 4. Member functions defined in the code ...
#FIXME: More detailed (consider assign ...)
class FuncTable:
    def __init__(self, init_func_table: dict) -> None:
        assert isinstance(init_func_table, dict)  # name -> ast.FunctionDef
        self.func_table = init_func_table

    def lookup(self, name) -> ast.FunctionDef:
        # `name` <=> `ast.Call().func`
        result = None
        if isinstance(name, ast.Name):
            funcname = name.id
            result = self.func_table.get(funcname)
        return result

    def update(self, codeast: ast.Module):
        self.func_table = self.__class__.find_funcdefs(codeast)

    @classmethod
    def find_funcdefs(cls, codeast: ast.Module):
        # 1.
        class FindFDef(ast.NodeVisitor):
            def __init__(self) -> None:
                self.fdefs = []
            def visit_FunctionDef(self, node: ast.FunctionDef):
                self.fdefs.append(node)
        ffd = FindFDef()
        ffd.visit(codeast)
        return {e.name: _set_parent(_clone_ast(e)) for e in ffd.fdefs if e.parent == codeast}  ##TODO: temp

    @classmethod
    def make(cls, codeast: ast.Module):
        assert isinstance(codeast, ast.Module)
        return cls(init_func_table=cls.find_funcdefs(codeast))


class FindImports(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports = []

    def visit_Import(self, node: ast.Import):
        self.imports.append(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.imports.append(node)


class GetIDs(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.ids = set()

    def visit_Name(self, node: ast.Name):
        self.ids.add(node.id)

    def visit_alias(self, node: ast.alias):
        self.ids.add(node.asname or node.name)


class GetRWRefIDs(NodeVisitorInRunOrder):
    def __init__(self, recursive) -> None:
        super().__init__()
        self.__rids = dict()
        self.__wids = dict()
        self.__initids = dict()
        self.__scope_stack = []
        self.__recursive = recursive

    def __make_fullname(self, id_: str):
        fullname = f'{".".join(self.__scope_stack)}.{id_}'
        if fullname[0] == '.': fullname = fullname[1:]
        return fullname

    @property
    def rids(self):
        # print(self.__rids)
        return self.__rids.keys()

    @property
    def wids(self):
        # print(self.__wids)
        return self.__wids.keys()

    @property
    def initids(self):
        return self.__initids.keys()

    @property
    def rid_nodes(self):
        return self.__rids

    @property
    def wid_nodes(self):
        return self.__wids

    @property
    def initid_nodes(self):
        return self.__initids

    @property
    def allr_after_i_ids(self):
        return (self.initids - self.rids) | \
            {e for e in (self.rids & self.initids) \
                    if min([i[0] for i in self.rid_nodes[e]]) > max([i[0] for i in self.initid_nodes[e]])}

    @property
    def local_id2nodes(self) -> dict: # id => (idnode => seq)
        def per_id(id_: str) -> dict: # idnode => seq
            if id_ not in self.initid_nodes: return {}
            assert min([e[0] for e in self.initid_nodes[id_]]) == self.initid_nodes[id_][0][0]
            first_init_seq = self.initid_nodes[id_][0][0]
            all_nodes_of_id = [
                *self.wid_nodes[id_],
                *self.initid_nodes[id_],
                *self.rid_nodes.get(id_, {})
            ]
            return {I[1][0]: I[0] for I in all_nodes_of_id if I[0] >= first_init_seq}

        assert set(self.initids).issubset(self.wids)
        return {k: per_node2seq for k in (self.rids | self.wids) if (per_node2seq := per_id(k))}

    def _add_id(self, ids: dict, id_: str, info, fullname):
        id_ = fullname or self.__make_fullname(id_)
        info = (self.current_cnt, info)
        if id_ in ids:
            ids[id_].append(info)
        else:
            ids[id_] = [info]

    @classmethod
    def _check_w(cls, node):
        if isinstance(node, ast.Attribute) and \
            isinstance(node.parent, ast.Call) and \
            node == node.parent.func:
            return True, True, node.parent  #NOTE: NEED MORE ANALYSIS
        if hasattr(node, 'ctx') and isinstance(node.ctx, ast.Store):
            return True, False, node
        if (isinstance(node.parent, ast.Attribute) and node == node.parent.value) or \
            (isinstance(node.parent, ast.Subscript) and node == node.parent.value) or \
            (isinstance(node.parent, ast.Starred) and node == node.parent.value) or \
            (isinstance(node.parent, (ast.Tuple, ast.List)) and node in node.parent.elts):
            return cls._check_w(node.parent)
        return False, True, None

    @classmethod
    def _check_i(cls, node):
        if isinstance(node.parent, ast.Assign) and node in node.parent.targets:
            return True, node.parent
        if isinstance(node.parent, ast.AnnAssign) and \
                       node == node.parent.target and \
                       node.parent.value is not None:
            return True, node.parent
        if isinstance(node.parent, ast.For) and node == node.parent.target:
            return True, node.parent
        if isinstance(node.parent, ast.withitem) and node == node.parent.optional_vars:
            return True, node.parent
        if isinstance(node.parent, (ast.Tuple, ast.List)) and node in node.parent.elts:
            return cls._check_i(node.parent)
        return False, None

    def _add_rid(self, id_: str, node, rnode, fullname=None):
        self._add_id(self.__rids, id_, (node, rnode), fullname)

    def _add_wid(self, id_: str, node, wnode, fullname=None):
        self._add_id(self.__wids, id_, (node, wnode), fullname)

    def _add_initid(self, id_: str, node, inode, fullname=None):
        self._add_id(self.__initids, id_, (node, inode), fullname)

    def visit_Name(self, node: ast.Name):
        written, read, wnode = self.__class__._check_w(node)
        fullname = self.__make_fullname(node.id).split('.')
        if written:
            isinit, inode = self.__class__._check_i(node)
            if isinit:
                self._add_initid(node.id, node, inode)
                self._add_wid(node.id, node, wnode)
            else:
                for i in range(len(fullname)):
                    possible_fullname = '.'.join(fullname[i:])
                    if possible_fullname in self.initid_nodes:
                        self._add_wid(node.id, node, wnode, fullname=possible_fullname)
                        break
                else:
                    self._add_wid(node.id, node, wnode, fullname=possible_fullname)
        if read:
            for i in range(len(fullname)):
                possible_fullname = '.'.join(fullname[i:])
                if possible_fullname in self.initid_nodes:
                    self._add_rid(node.id, node, wnode, fullname=possible_fullname)
                    break
            else:
                self._add_rid(node.id, node, wnode, fullname=possible_fullname)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._add_wid(node.name, node, node)
        self._add_initid(node.name, node, node)
        #NOTE: DON'T visit body of the node
        if self.__recursive:
            self.__scope_stack.append(node.name)
            self.generic_visit(node)
            self.__scope_stack.pop()

    def visit_arg(self, node: ast.arg):
        self._add_wid(node.arg, node, node)
        self._add_initid(node.arg, node, node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self._add_wid(node.name, node, node)
        self._add_initid(node.name, node, node)
        if self.__recursive:
            self.__scope_stack.append(node.name)
            self.generic_visit(node)
            self.__scope_stack.pop()

    def visit_alias(self, node: ast.alias):
        assert isinstance(node.parent, (ast.Import, ast.ImportFrom))
        id_ = node.asname or node.name
        self._add_wid(id_, node, node.parent)
        self._add_initid(id_, node, node.parent)


#TODO: More strict
class VerifyKerasSFModel(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.with_compile = False
        self.with_fit = False

    def valid(self):
        return self.with_compile and self.with_fit

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ('fit', 'fit_generator', 'train_on_batch'):
                self.with_fit = True
            elif node.func.attr == 'compile':
                self.with_compile = True
        self.generic_visit(node)


#TODO: More strict
class FindPytorchModelTrainLoop(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.loops = []

    def _visit_loop(self, node):
        call_step = _find_x(node, is_x=lambda n: _check_call_member_fn('step', n, nargs=0, kwargkeys=[]))
        call_zero_grad = _find_x(node, is_x=lambda n: _check_call_member_fn('zero_grad', n))
        call_backward = _find_x(node, is_x=lambda n: _check_call_member_fn('backward', n))
        if call_step and call_zero_grad and call_backward:
            self.loops.append(node)

    def visit_For(self, node: ast.For):
        return self._visit_loop(node)

    def visit_While(self, node: ast.While):
        return self._visit_loop(node)


class FindCalls(ast.NodeVisitor):
    def __init__(self, recursive, ignore_parents) -> None:
        super().__init__()
        self.calls = []
        self.__recursive = recursive
        self.__ignore_parents = ignore_parents

        def empty_visit(node):
            pass

        for p in self.__ignore_parents:
            setattr(self, f'visit_{p}', empty_visit)

    def visit_Call(self, node: ast.Call):
        if self.__recursive:
            self.calls.append(node)
            self.generic_visit(node)
        else:
            self.calls.append(node)


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


def _get_import_root(filename, root: str = None):
    # Adapted from pyan.anutils.get_module_name
    def _inner(filename, root: str = None):
        """Try to determine the full module name of a source file, by figuring out
        if its directory looks like a package (i.e. has an __init__.py file or
        there is a .py file in it )."""

        if os.path.basename(filename) == "__init__.py":
            # init file means module name is directory name
            module_path = os.path.dirname(filename)
        else:
            # otherwise it is the filename without extension
            module_path = filename.replace(".py", "")

        # find the module root - walk up the tree and check if it contains .py files - if yes. it is the new root
        directories = [(module_path, True)]
        if root is None:
            while directories[0][0] != os.path.dirname(directories[0][0]):
                potential_root = os.path.dirname(directories[0][0])
                is_root = any([f == "__init__.py" for f in os.listdir(potential_root)])
                directories.insert(0, (potential_root, is_root))

            # keep directories where itself of parent is root
            while not directories[0][1]:
                directories.pop(0)

        else:  # root is already known - just walk up until it is matched
            while directories[0][0] != root:
                potential_root = os.path.dirname(directories[0][0])
                directories.insert(0, (potential_root, True))

        assert directories
        return os.path.dirname(directories[0][0])
    return _inner(os.path.abspath(filename), root)


# _AST_FIELDS_IN_RUN_ORDER (for visiting the given node in run-order)
## MAP: ast_class_name => (field_name0, field_name1, ...)
### Modified from <ast-class>._fields, python.__version__ == 3.11.4
_AST_FIELDS_IN_RUN_ORDER = {
    'AST': (),
    'Add': (),
    'And': (),
    'AnnAssign': ('value', 'target', 'annotation', 'simple'),
    'Assert': ('test', 'msg'),
    'Assign': ('value', 'targets', 'type_comment'),
    'AsyncFor': ('iter', 'target', 'body', 'orelse', 'type_comment'),
    'AsyncFunctionDef': ('name', 'args', 'body', 'decorator_list', 'returns', 'type_comment'),
    'AsyncWith': ('items', 'body', 'type_comment'),
    'Attribute': ('value', 'attr', 'ctx'),
    'AugAssign': ('value', 'target', 'op'),
    'AugLoad': (),
    'AugStore': (),
    'Await': ('value',),
    'BinOp': ('left', 'op', 'right'),
    'BitAnd': (),
    'BitOr': (),
    'BitXor': (),
    'BoolOp': ('op', 'values'),
    'Break': (),
    'Bytes': ('s',),
    'Call': ('func', 'args', 'keywords'),
    'ClassDef': ('name', 'bases', 'keywords', 'body', 'decorator_list'),
    'Compare': ('left', 'ops', 'comparators'),
    'Constant': ('value', 'kind'),
    'Continue': (),
    'Del': (),
    'Delete': ('targets',),
    'Dict': ('keys', 'values'),  # FIXME: 1, 2 in zip(keys, values)
    'DictComp': ('generators', 'key', 'value'), 
    'Div': (),
    'Ellipsis': (),
    'Eq': (),
    'ExceptHandler': ('type', 'name', 'body'),
    'Expr': ('value',),
    'Expression': ('body',),
    'ExtSlice': (),
    'FloorDiv': (),
    'For': ('iter', 'target', 'body', 'orelse', 'type_comment'),
    'FormattedValue': ('value', 'conversion', 'format_spec'), ##
    'FunctionDef': ('name', 'args', 'body', 'decorator_list', 'returns', 'type_comment'),
    'FunctionType': ('argtypes', 'returns'),
    'GeneratorExp': ('generators', 'elt'),
    'Global': ('names',),
    'Gt': (),
    'GtE': (),
    'If': ('test', 'body', 'orelse'),
    'IfExp': ('test', 'body', 'orelse'),
    'Import': ('names',),
    'ImportFrom': ('module', 'names', 'level'),
    'In': (),
    'Index': (),
    'Interactive': ('body',),
    'Invert': (),
    'Is': (),
    'IsNot': (),
    'JoinedStr': ('values',),
    'LShift': (),
    'Lambda': ('args', 'body'),
    'List': ('elts', 'ctx'),
    'ListComp': ('generators', 'elt'),
    'Load': (),
    'Lt': (),
    'LtE': (),
    'MatMult': (),
    'Match': ('subject', 'cases'), ##
    'MatchAs': ('pattern', 'name'), ##
    'MatchClass': ('cls', 'patterns', 'kwd_attrs', 'kwd_patterns'), ##
    'MatchMapping': ('keys', 'patterns', 'rest'), ##
    'MatchOr': ('patterns',), ##
    'MatchSequence': ('patterns',), ##
    'MatchSingleton': ('value',), ##
    'MatchStar': ('name',), ##
    'MatchValue': ('value',), ##
    'Mod': (),
    'Module': ('body', 'type_ignores'),
    'Mult': (),
    'Name': ('id', 'ctx'),
    'NameConstant': ('value', 'kind'),
    'NamedExpr': ('target', 'value'),
    'Nonlocal': ('names',),
    'Not': (),
    'NotEq': (),
    'NotIn': (),
    'Num': ('n',),
    'Or': (),
    'Param': (),
    'Pass': (),
    'Pow': (),
    'RShift': (),
    'Raise': ('exc', 'cause'),
    'Return': ('value',),
    'Set': ('elts',),
    'SetComp': ('generators', 'elt'),
    'Slice': ('lower', 'upper', 'step'),
    'Starred': ('value', 'ctx'),
    'Store': (),
    'Str': ('s',),
    'Sub': (),
    'Subscript': ('value', 'slice', 'ctx'),
    'Suite': (),
    'Try': ('body', 'handlers', 'orelse', 'finalbody'),
    'TryStar': ('body', 'handlers', 'orelse', 'finalbody'),
    'Tuple': ('elts', 'ctx'),
    'TypeIgnore': ('lineno', 'tag'),
    'UAdd': (),
    'USub': (),
    'UnaryOp': ('op', 'operand'),
    'While': ('test', 'body', 'orelse'),
    'With': ('items', 'body', 'type_comment'),
    'Yield': ('value',),
    'YieldFrom': ('value',),
    'alias': ('name', 'asname'),
    'arg': ('arg', 'annotation', 'type_comment'),
    'arguments': ('posonlyargs', 'args', 'vararg', 'kwonlyargs', 'kw_defaults', 'kwarg', 'defaults'),
    'boolop': (),
    'cmpop': (),
    'comprehension': ('iter', 'target', 'ifs', 'is_async'),
    'excepthandler': (),
    'expr': (),
    'expr_context': (),
    'keyword': ('arg', 'value'),
    'match_case': ('pattern', 'guard', 'body'),
    'mod': (),
    'operator': (),
    'pattern': (),
    'slice': (),
    'stmt': (),
    'type_ignore': (),
    'unaryop': (),
    'withitem': ('context_expr', 'optional_vars'),
}


def _get_version_hash() -> str:
    return "UTI5d2VYSnBaMmgwSUNoaktTQXlNREl6TENCTmFXNW5iV2x1Wnl" \
            "CYWFHRnVaeTRnUVd4c0lISnBaMmgwY3lCeVpYTmxjblpsWkM0PQ=="


def _iter_fields_in_run_order(node):
    def _get_fields(N):
        return _AST_FIELDS_IN_RUN_ORDER.get(
            N.__class__.__name__)

    for field in _get_fields(node):
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


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


def _is_in_block(stmt, block):
    def body_stmts(b):
        return {
            *getattr(b, 'body', {}),
            *getattr(b, 'orelse', {}),
            *getattr(b, 'finalbody', {})
        }
    return _is_block_stmt(block) and stmt.parent == block and stmt in body_stmts(block)


def _is_blockdef(node):
    blockdef_nodes = (
        ast.Module,
        ast.FunctionDef,
        ast.ClassDef,
    )
    return isinstance(node, blockdef_nodes)


def _get_relname(name: str, start: str, d: str = None) -> str | None:
    assert name is not None and start is not None
    d = d or '.'
    name_l = name.split(d)
    start_l = start.split(d)
    if len(name_l) < len(start_l): return None
    if name_l[:len(start_l)] != start_l: return None
    return '.'.join(name_l[len(start_l):])


def _get_imported_symbol_and_id(node: ast.Import | ast.ImportFrom) -> str:
    assert len(node.names) == 1
    name0 = node.names[0]
    if isinstance(node, ast.Import):
        return name0.name, name0.asname or name0.name
    elif isinstance(node, ast.ImportFrom):
        return f"{'.'*node.level}{f'{node.module}.' or ''}{name0.name}", name0.asname or name0.name


def _is_name_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return True
    elif isinstance(node, ast.Attribute):
        return _is_name_node(node.value)
    return False


def _make_name_node(*, name: str=None, name_l: list=None, ctx=None):
    if name is not None:
        name_l = name.split('.')
    if len(name_l) == 1:
        return ast.Name(id=name_l[0], ctx=ctx)
    return ast.Attribute(value=_make_name_node(name_l=name_l[:-1], ctx=ast.Load()), attr=name_l[-1], ctx=ctx)


def _remove_fn_head(codeast: ast.Module) -> ast.Module:
    assert len(codeast.body) == 1 and isinstance(codeast.body[0], ast.FunctionDef)
    assert isinstance(codeast.body[0].body[-1], ast.Return)
    codeast.body = codeast.body[0].body[:-1]  # remove return
    return ast.fix_missing_locations(codeast)


def _name_equals(a: ast.AST, b: ast.AST) -> bool:
    if a == b:
        return True
    elif type(a) is not type(b):
        return False
    elif isinstance(a, ast.Name):
        assert isinstance(b, ast.Name)
        return a.id == b.id
    elif isinstance(a, ast.Attribute):
        assert isinstance(b, ast.Attribute)
        return a.attr == b.attr and _name_equals(a.value, b.value)
    return False


def _find_defined_in(node):
    assert hasattr(node, 'parent')
    paths = []
    last_p = node
    for p in _iter_parent_chain(node, with_self=False):
        if isinstance(p, (ast.FunctionDef, ast.ClassDef)) and _is_in_block(stmt=last_p, block=p):
            paths.append(p)
        last_p = p
    return '.'.join(reversed([f.name for f in paths]))


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

def _make_if_true_block(stmts: list):
    assert isinstance(stmts, list)
    return ast.If(
        test=ast.Constant(value=True),
        body=stmts,
        orelse=[])


# -> (found, node)
# NOTE: NOT WORKING ON `ast.Load(), ...` (single instance)
def _set_parent(root):
    if not hasattr(root, 'parent'):
        root.parent = None
    for child in ast.iter_child_nodes(root):
        child.parent = root
        _set_parent(child)
    return root


def _set_depth(root, init_depth=0):
    root._pg_depth = init_depth
    for child in ast.iter_child_nodes(root):
        _set_depth(child, init_depth + 1)
    return root


def _set_node_id(root, init_id=0):
    for i, n in enumerate(ast.walk(root), start=init_id):
        n._pg_node_id = i
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


def _get_ids(node, ignores=None):
    if node is None: return set()
    ignores = ignores or ()
    if isinstance(node, (list, tuple, set)):
        ids = set()
        for n in node:
            ids.update(_get_ids(n, ignores=ignores))
        return ids
    else:
        gi = GetIDs()
        gi.visit(node)
        ids = gi.ids
        ids -= set(ignores)
        return ids


def _get_rwref_ids(node,
                   *, 
                   recursive=True,
                   rignores=None, 
                   wignores=None, 
                   iignores=None, 
                   raiignores=None, 
                   localignores=None,
                   ignores=None, 
                   get_locals=False,
                   get_all=False):
    assert node is not None
    rignores = rignores or ignores or set()
    wignores = wignores or ignores or set()
    iignores = iignores or ignores or set()
    raiignores = raiignores or ignores or set()
    localignores = localignores or ignores or set()
    gi = GetRWRefIDs(recursive=recursive)
    gi.visit(node)
    if get_all: return gi
    rids, wids, initids, raiids = gi.rids, gi.wids, gi.initids, gi.allr_after_i_ids
    rids -= set(rignores)
    wids -= set(wignores)
    initids -= set(iignores)
    raiids -= set(raiignores)
    if not get_locals:
        return rids, wids, initids, raiids
    local_id2nodes = {
        k: v
        for k, v in gi.local_id2nodes.items()
            if k not in localignores
    }
    return rids, wids, initids, raiids, local_id2nodes


def _get_local_id2nodes(node, ignores=None) -> dict: # id => (node => seq)
    if isinstance(node, (tuple, list)):
        node = _make_if_true_block(node)
    node = _set_parent(node)
    r, w, i, r, locals_ = _get_rwref_ids(node, ignores=ignores, get_locals=True)
    return locals_


def _find_imports(node):
    if node is None: return {}
    if isinstance(node, (list, tuple, set)):
        imports = []
        for n in node:
            imports.extend(_find_imports(n))
        return imports
    else:
        fi = FindImports()
        fi.visit(node)
        return fi.imports


def _find_calls(node, recursive, ignore_parents):
    if node is None: return []
    if isinstance(node, (list, tuple, set)):
        calls = []
        for n in node:
            calls.extend(_find_calls(n, recursive=recursive, ignore_parents=ignore_parents))
        return calls
    fc = FindCalls(recursive=recursive, ignore_parents=ignore_parents)
    fc.visit(node)
    return fc.calls


def _find_if_name_eq_main(node):
    class FindIfNEqM(ast.NodeVisitor):
        def __init__(self):
            self.ifs = []
        def visit_If(self, node):
            if self.check(node):
                self.ifs.append(node)
        def check(self, node):
            assert isinstance(node, ast.If)
            # If(test=Compare(left=Name(id='__name__', ctx=Load()), ops=[Eq()],
            #                comparators=[Constant(value='__main__')]),
            #    body=[...],
            #    orelse=[])
            return isinstance(node.test, ast.Compare) and \
                    isinstance(node.test.left, ast.Name) and \
                    node.test.left.id == '__name__' and \
                    len(node.test.ops) == 1 and \
                    isinstance(node.test.ops[0], ast.Eq) and \
                    len(node.test.comparators) == 1 and \
                    isinstance(node.test.comparators[0], ast.Constant) and \
                    node.test.comparators[0].value == '__main__'

    finem = FindIfNEqM()
    finem.visit(node)
    return finem.ifs


def _get_nearest_block_stmt(node, use_vblock=False):
    last_p = node
    for p in _iter_parent_chain(node, with_self=False):
        if use_vblock and hasattr(last_p, '_pg_vblock'):
            return last_p._pg_vblock, last_p
        elif _is_in_block(stmt=last_p, block=p):
            return p, last_p
        last_p = p
    assert False, 'Unreachable'


def _force_untab_block(codeast, block_nodes):
    # assert hasattr(block_nodes[i], 'body')
    # block_node -> block_node.body
    block_nodes = set(block_nodes)
    codeast = ReplaceXWithY(
        x_cls_names={e.__class__.__name__ for e in block_nodes},
        is_x=lambda n: n in block_nodes,
        make_y=lambda n: n.body).visit(codeast)
    return _set_parent(ast.fix_missing_locations(codeast))


def _remove_useless_stmts(codeast, level=0):
    assert level == 0

    def is_name(x):
        return isinstance(x, (ast.Name, ast.Constant)) or \
                (isinstance(x, ast.Attribute) and is_name(x.value)) or \
                (isinstance(x, (ast.Tuple, ast.List, ast.Set)) and all([is_name(xx) for xx in x.elts]))

    def is_name_stmt(x):
        return isinstance(x, ast.Expr) and is_name(x.value) \
            and '__mask_0__' not in ast.unparse(x) \
            and 'seed(' not in ast.unparse(x)
        # return isinstance(x, ast.Expr) and is_name(x.value)

    def get_new_node(x: ast.Expr):
        if x in (body := getattr(x.parent, 'body', {})) and len(body) == 1:
            return ast.Pass()
        if x in (body := getattr(x.parent, 'orelse', {})) and len(body) == 1:
            return ast.Pass()
        if x in (body := getattr(x.parent, 'finalbody', {})) and len(body) == 1:
            return ast.Pass()
        return None

    nodes = set(_find_x(codeast, is_x=is_name_stmt))
    codeast = ReplaceXWithY(
        x_cls_names={e.__class__.__name__ for e in nodes},
        is_x=lambda n: n in nodes,
        make_y=lambda n: get_new_node(n)).visit(codeast)
    return _set_parent(ast.fix_missing_locations(codeast))


def _remove_constant_assign(codeast):
    # Use with `_inline_all_vars_l1`

    def is_constant_assign(x):
        # a = b = c = 1
        return isinstance(x, ast.Assign) and \
                all([isinstance(t, ast.Name) for t in x.targets]) and \
                isinstance(x.value, ast.Constant)

    def get_new_node(x: ast.Assign):
        if x in (body := getattr(x.parent, 'body', {})) and len(body) == 1:
            return ast.Pass()
        if x in (body := getattr(x.parent, 'orelse', {})) and len(body) == 1:
            return ast.Pass()
        if x in (body := getattr(x.parent, 'finalbody', {})) and len(body) == 1:
            return ast.Pass()
        return None
    
    nodes = set(_find_x(codeast, is_x=is_constant_assign))
    codeast = ReplaceXWithY(
        x_cls_names=['Assign'],
        is_x=lambda n: n in nodes,
        make_y=lambda n: get_new_node(n)).visit(codeast)
    return _set_parent(ast.fix_missing_locations(codeast))


def _std_calls(codeast, idx_maker):
    assert callable(idx_maker)
    # std calls:
    ## ast.Expr(ast.Call()), f(), node.parent == BLOCK
    ## ast.Assign(target=..., value=ast.Call()), ret = f(), node.parent == BLOCK
    codeast = _set_depth(codeast)
    calls = _find_calls(codeast, recursive=True, ignore_parents=[])
    calls.sort(key=lambda e: -e._pg_depth)  # sort by -depth; rq:stable-sort
    soft_func_name = lambda c: c.func.id if isinstance(c.func, ast.Name) else ''
    replace_node_map = {}  # node => [stmt0, stmt1, ...]
    for c in calls:
        if isinstance(c.parent, (ast.Expr, ast.Assign, ast.AnnAssign)) and \
                                        _is_in_block(c.parent, c.parent.parent):
            continue
        block, c_pstmt = _get_nearest_block_stmt(c)
        sc_tmp_var = f'sc_T{soft_func_name(c)}{idx_maker()}'
        ass_stmt = ast.Assign(
            targets=[ast.Name(id=sc_tmp_var, ctx=ast.Store())],
            value=c,
            type_comment=None)
        proced_c_pstmt = ReplaceXWithY(
            x_cls_names=['Call'],
            is_x=lambda n: n == c,
            make_y=lambda n: ast.Name(id=sc_tmp_var, ctx=ast.Load())
            ).visit(c_pstmt)
        if c_pstmt not in replace_node_map:
            replace_node_map[c_pstmt] = []
        replace_node_map[c_pstmt].append(ass_stmt)
    codeast = ReplaceXWithY(
        x_cls_names={x.__class__.__name__ for x in replace_node_map.keys()},
        is_x=lambda n: n in replace_node_map,
        make_y=lambda n: [*replace_node_map[n], n]).visit(codeast)
    return _set_parent(ast.fix_missing_locations(codeast))


def _trans_comp_to_loop(codeast, idx_maker):
    assert callable(idx_maker)
    # <id0> = [<expr0> for <id1> in <expr1> ... ]
    ##-->
    # <id0> = []
    # for <id1> in <expr1>:
    #     ...
    #     <id0>.append(<expr0>)

    def _trans_generators_to_loop(generators, innermost_body):
        if not generators: return innermost_body
        g0: ast.comprehension = generators[0]
        body = _trans_generators_to_loop(generators[1:], innermost_body)
        if g0.ifs:
            body = ast.If(
                test=ast.BoolOp(op=ast.And(), values=g0.ifs),
                body=[body],
                orelse=[])
        return ast.For(
            target=g0.target,
            iter=g0.iter,
            body=[body],
            orelse=[],
            type_comment=None
        )
    
    def _make_ass(target_id, comp):
        if isinstance(comp, ast.ListComp):
            ty = 'list'
        elif isinstance(comp, ast.SetComp):
            ty = 'set'
        elif isinstance(comp, ast.DictComp):
            ty = 'dict'
        else:
            assert False, 'expect ListComp, SetComp or DictComp'
        # <id0> = list() # set(), or dict()
        return ast.Assign(
            targets=[ast.Name(id=target_id, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Name(id=ty, ctx=ast.Load()),
                args=[],
                keywords=[]),
            type_comment=None)

    def _make_update_container(target_id, comp):
        if isinstance(comp, ast.ListComp):
            return ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                    value=ast.Name(id=target_id, ctx=ast.Load()),
                    attr='append',
                    ctx=ast.Load()),
                    args=[comp.elt],
                    keywords=[]))
        elif isinstance(comp, ast.SetComp):
            return ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                    value=ast.Name(id=target_id, ctx=ast.Load()),
                    attr='add',
                    ctx=ast.Load()),
                    args=[comp.elt],
                    keywords=[]))
        elif isinstance(comp, ast.DictComp):
            return ast.Assign(
                targets=[
                    ast.Subscript(
                    value=ast.Name(id=target_id, ctx=ast.Load()),
                    slice=comp.key,
                    ctx=ast.Store())],
                value=comp.value)
        else:
            assert False, 'expect ListComp, SetComp or DictComp'

    codeast = _set_depth(codeast)
    comps = _find_x(codeast,
                    is_x=lambda x: isinstance(x, (ast.ListComp, ast.SetComp, ast.DictComp)))
    comps.sort(key=lambda e: e._pg_depth)
    levels = set(c._pg_depth for c in comps)
    levels2comps = {k: [] for k in levels}
    for c in comps:
        levels2comps[c._pg_depth].append(c)
    for lv in levels:
        replace_node_map = {}  # node => [stmt0, stmt1, ...]
        for c in levels2comps[lv]:
            block, c_pstmt = _get_nearest_block_stmt(c)
            tc2l_tmp_var = f'tc2l_T{idx_maker()}'
            ass_stmt = _make_ass(tc2l_tmp_var, c)
            forloop = _trans_generators_to_loop(
                            generators=c.generators,
                            innermost_body=_make_update_container(tc2l_tmp_var, c))
            proced_c_pstmt = ReplaceXWithY(
                x_cls_names=[c.__class__.__name__],
                is_x=lambda n: n == c,
                make_y=lambda n: ast.Name(id=tc2l_tmp_var, ctx=ast.Load())
                ).visit(c_pstmt)
            if c_pstmt not in replace_node_map:
                replace_node_map[c_pstmt] = []
            replace_node_map[c_pstmt].append(ass_stmt)
            replace_node_map[c_pstmt].append(forloop)
        codeast = _set_parent(ReplaceXWithY(
            x_cls_names={x.__class__.__name__ for x in replace_node_map.keys()},
            is_x=lambda n: n in replace_node_map,
            make_y=lambda n: [*replace_node_map[n], n]).visit(codeast))
    return _set_parent(ast.fix_missing_locations(codeast))


def _inline_all_vars_l1(codeast):
    class VarValTracer(NodeTransformerInRunOrder):
        def __init__(self):
            super().__init__()
            self.var_table = {}  # node => val node (xxx or List/Tuple[xxx]) (must be leaf)
            self.val_ref_table = {}  # val node => [name table, ] (sorted by ref-order)
            self.scope_stack = []
            self.init_names = set()
            self.rw_only1 = {}  # r/w node => val node

            def visit_branch(node):
                return node
            
            branches = [ast.If, ast.For, ast.While, ast.Match]
            for b in branches:
                setattr(self, f'visit_{b.__name__}', visit_branch)

        def fullname(self, id_: str):
            fullname = f'{".".join(self.scope_stack)}.{id_}'
            if fullname[0] == '.': fullname = fullname[1:]
            return fullname
        
        def resolve_val(self, val: ast.AST, index: int = None):
            if isinstance(val, ast.Name):
                fullname = self.fullname(val.id)
                if (v := self.var_table.get(fullname, None)):
                    if index is None:
                        return v
                    elif isinstance(v, tuple) and index < len(v.elts):
                        return v[index]
            elif isinstance(val, (ast.Tuple, ast.List)):
                if index is None:
                    return tuple(self.resolve_val(e) for e in val.elts)
                elif index < len(val.elts):
                    return self.resolve_val(val.elts[index])
            if index is None:
                return val  # the val is leaf
            return None  # resolve failed

        def update_var_table(self, var: ast.AST, val):
            resolved_val = self.resolve_val(val)
            self.update_val_ref_table(resolved_val, var)
            if isinstance(var, ast.Name):
                self.var_table[self.fullname(var.id)] = resolved_val
            elif isinstance(var, (ast.List, ast.Tuple)):
                for i, elt in enumerate(var.elts):
                   self.update_var_table(elt, self.resolve_val(resolved_val, i))

        def update_val_ref_table(self, val, new_var):
            old_val = self.resolve_val(new_var)
            # Add new
            if val not in self.val_ref_table:
                self.val_ref_table[val] = []
            self.val_ref_table[val].append(new_var)
            if isinstance(val, tuple):
                for i, elt_val in enumerate(val):
                    if isinstance(new_var, (ast.List, ast.Tuple)):
                        sub_var = new_var.elts[i]
                        self.update_val_ref_table(elt_val, sub_var)
            # Del old
            if old_val and old_val in self.val_ref_table:
                var_list: list = self.val_ref_table[old_val]
                #FIXME: Consider more...
                self.val_ref_table[old_val] = list(filter(lambda x: ast.unparse(ast.fix_missing_locations(x)) != ast.unparse(ast.fix_missing_locations(new_var)), var_list))

        def get_current_ref_of_val(self, val):
            # print({(ast.unparse(k), id(k)): [ast.unparse(e) for e in v] for k, v in self.val_ref_table.items()})
            if (vars := self.val_ref_table.get(val, None)):
                return vars[0]
            return None

        def update_init_names(self, node):
            if not isinstance(node, (list, tuple)):
                node = (node, )
            for n in node:
                if isinstance(n, ast.Name):
                    self.init_names.add(n)
                elif isinstance(n, (ast.Tuple, ast.List)):
                    for elt in n.elts:
                        self.update_init_names(elt)

        def make_leaf_node(self, original_node: ast.AST):
            node = _clone_ast(original_node)
            if hasattr(node, 'ctx'):
                node.ctx = ast.Load()
            return node            

        def visit_FunctionDef(self, node: ast.FunctionDef):
            self.scope_stack.append(node.name)
            node = self.generic_visit(node)
            self.scope_stack.pop()
            return node

        def visit_ClassDef(self, node: ast.ClassDef):
            self.scope_stack.append(node.name)
            node = self.generic_visit(node)
            self.scope_stack.pop()
            return node

        def visit_Assign(self, node: ast.Assign):
            self.update_init_names(node.targets)
            node = self.generic_visit(node)
            for t in node.targets:
                self.update_var_table(t, node.value)
            return node

        def visit_AnnAssign(self, node: ast.AnnAssign):
            if node.value:
                self.update_init_names(node.target)
                node = self.generic_visit(node)
                self.update_var_table(node.target, node.value)
            return node

        op_table = {
            # ast.operator
            ast.Add: lambda a, b: a + b,
            ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b,
            ast.Div: lambda a, b: a / b,
            ast.FloorDiv: lambda a, b: a // b,
            ast.Mod: lambda a, b: a % b,
            ast.Pow: lambda a, b: a ** b,
            ast.LShift: lambda a, b: a << b,
            ast.RShift: lambda a, b: a >> b,
            ast.BitOr: lambda a, b: a | b,
            ast.BitXor: lambda a, b: a ^ b,
            ast.BitAnd: lambda a, b: a & b,
            ast.MatMult: lambda a, b: a @ b,
            # ast.boolop
            ast.And: lambda a, b: a and b,
            ast.Or: lambda a, b: a or b,
            # ast.unaryop
            ast.UAdd: lambda a: a,
            ast.USub: lambda a: -a,
            ast.Not: lambda a: not a,
            ast.Invert: lambda a: ~a,
            # ast.cmpop
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a >= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
            ast.Is: lambda a, b: a is b,
            ast.IsNot: lambda a, b: a is not b,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b
        }

        def visit_BinOp(self, node: ast.BinOp):
            node = self.generic_visit(node)
            left = node.left
            right = node.right
            if not isinstance(left, ast.Constant) or not isinstance(right, ast.Constant):
                return node
            try:
                return ast.Constant(value=self.op_table[node.op.__class__](left.value, right.value))
            except:
                return node

        def visit_BoolOp(self, node: ast.BoolOp):
            node = self.generic_visit(node)
            vals = node.values
            if any(not isinstance(v, ast.Constant) for v in vals):
                return node
            try:
                assert len(vals) >= 2
                val = vals[0]
                for i in range(1, len(vals)):
                    val = self.op_table[node.op.__class__](val, vals[i])
                return ast.Constant(value=val)
            except:
                return node
            
        def visit_UnaryOp(self, node: ast.UnaryOp):
            node = self.generic_visit(node)
            if not isinstance(node.operand, ast.Constant):
                return node
            try:
                # - 1 => -1, whatever, the ast_unparse work well...
                return ast.Constant(value=self.op_table[node.op.__class__](node.operand))
            except:
                return node
        
        def visit_Compare(self, node: ast.Compare):
            node = self.generic_visit(node)
            vals = (node.left, *node.comparators)
            if any(not isinstance(v, ast.Constant) for v in vals):
                return node
            try:
                assert len(vals) >= 2
                val = vals[0]
                for i in range(1, len(vals)):
                    val = self.op_table[node.ops[i - 1].__class__](val, vals[i])
                return ast.Constant(value=val)
            except:
                return node

        def visit_Name(self, node: ast.Name):
            if node not in self.init_names and \
               (val := self.resolve_val(node)):
                if isinstance(val, ast.Constant):
                    return self.make_leaf_node(val)
                elif other_name := self.get_current_ref_of_val(val):
                    return self.make_leaf_node(other_name)
            return node

    vvt = VarValTracer()
    codeast = vvt.visit(codeast)
    return _set_parent(ast.fix_missing_locations(codeast))


def _inline_all_vars_l2(codeast):
    gi: GetRWRefIDs = _get_rwref_ids(codeast, get_all=True)
    ass_list: list[ast.Assign] = _find_x(codeast, is_x=lambda x: isinstance(x, ast.Assign), ignore_sub_scope=True)
    replace_node_map = {}  # node => new node
    for ass in ass_list:
        if len(ass.targets) == 1:
            t = ass.targets[0]
            if isinstance(t, ast.Name) and \
               t.id in gi.rid_nodes and \
               len(gi.rid_nodes[t.id]) == 1 and \
               t.id in gi.wid_nodes and \
               len(gi.wid_nodes[t.id]) == 1:
                assert isinstance(gi.rid_nodes[t.id][0][1][0], ast.Name)
                replace_node_map[gi.rid_nodes[t.id][0][1][0]] = ass.value
                replace_node_map[ass] = None
    codeast = ReplaceXWithY(x_cls_names=['Name', 'Assign'],
                            is_x=lambda x: x in replace_node_map,
                            make_y=lambda x: replace_node_map.pop(x)).visit(codeast)
    return _set_parent(ast.fix_missing_locations(codeast))


def _norm_local_ids(codeast, id_maker, ignores=None):
    assert callable(id_maker)
    ignores = ignores or {}
    local_id2nodes: dict = _get_local_id2nodes(codeast, ignores=ignores)  # id => (node => seq)
    local_id2nodes = sorted([(k, v) for k, v in local_id2nodes.items()], key=lambda x: min(x[1].values()))
    for i, p in local_id2nodes:  # Change nodes in place
        i_l = i.split('.')
        name = id_maker()
        if len(i_l) > 1: pass  # Rename all locals
        for node in p.keys():
            if isinstance(node, ast.Name):
                assert node.id == i_l[-1]
                node.id = name
            elif isinstance(node, ast.FunctionDef):
                assert node.name == i_l[-1]
                node.name = name
            elif isinstance(node, ast.ClassDef):
                assert node.name == i_l[-1]
                node.name = name
            elif isinstance(node, ast.alias) and i_l[-1] != '*':
                assert (node.asname or node.name) == i_l[-1]
                node.asname = name
            elif isinstance(node, ast.arg):
                assert node.arg == i_l[-1]
                node.arg = name
    return codeast


def _sort_call_kwargs(codeast, key=None):
    key = lambda kw: kw.arg
    calls: list[ast.Call] = _find_calls(codeast, recursive=True, ignore_parents=[])
    for c in calls:
        c.keywords.sort(key=key)
    codeast = _set_parent(ast.fix_missing_locations(codeast))
    return codeast


def _verify_keras_sfmodel(node):
    if node is None: return False
    v = VerifyKerasSFModel()
    v.visit(node)
    return v.valid()


def _verify_pt_sfmodel(node):
    if node is None: return False
    f = FindPytorchModelTrainLoop()
    f.visit(node)
    return len(f.loops) > 0


_compile_kwargs_keys = ['optimizer',
                        'loss',
                        'metrics',
                        'loss_weights',
                        'weighted_metrics',
                        'run_eagerly',
                        'steps_per_execution',
                        'jit_compile',
                        'pss_evaluation_shards',]
_fit_kwargs_keys = ['x',
                    'y',
                    'batch_size',
                    'epochs', 'nb_epoch',
                    'verbose',
                    'callbacks',
                    'validation_split',
                    'validation_data',
                    'shuffle',
                    'class_weight',
                    'sample_weight',
                    'initial_epoch',
                    'steps_per_epoch',
                    'validation_steps',
                    'validation_batch_size',
                    'validation_freq',
                    'max_queue_size',
                    'workers',
                    'use_multiprocessing',]
_fitg_kwargs_keys = ['generator',
                     'steps_per_epoch',
                     'epochs', 'nb_epoch',
                     'verbose',
                     'callbacks',
                     'validation_data',
                     'validation_steps',
                     'validation_freq',
                     'class_weight',
                     'max_queue_size',
                     'workers',
                     'use_multiprocessing',
                     'shuffle',
                     'initial_epoch',]
_train_on_batch_kwargs_keys = ['x',
                               'y',
                               'sample_weight',
                               'class_weight',
                               'reset_metrics',
                               'return_dict',]
def _std_keras_compile_and_fit_kwargs(codeast):
    def compile_key(kw):
        try:
            return _compile_kwargs_keys.index(kw.arg)
        except ValueError:
            return 1024

    def fit_key(kw):
        try:
            return _fit_kwargs_keys.index(kw.arg)
        except ValueError:
            return 1024
        
    def fitg_key(kw):
        try:
            return _fitg_kwargs_keys.index(kw.arg)
        except ValueError:
            return 1024
       
    def tob_key(kw):
        try:
            return _train_on_batch_kwargs_keys.index(kw.arg)
        except ValueError:
            return 1024

    compile_calls = _find_x(codeast, is_x=lambda n: _check_call_member_fn('compile', n))
    for call in compile_calls:
        kwargs: list[ast.keyword] = call.keywords
        kwargs.sort(key=compile_key)

    fit_calls = _find_x(codeast, is_x=lambda n: _check_call_member_fn('fit', n))
    for call in fit_calls:
        kwargs: list[ast.keyword] = call.keywords
        kwargs.sort(key=fit_key)

    fitg_calls = _find_x(codeast, is_x=lambda n: _check_call_member_fn('fit_generator', n))
    for call in fitg_calls:
        kwargs: list[ast.keyword] = call.keywords
        kwargs.sort(key=fitg_key)

    tob_calls = _find_x(codeast, is_x=lambda n: _check_call_member_fn('train_on_batch', n))
    for call in tob_calls:
        kwargs: list[ast.keyword] = call.keywords
        kwargs.sort(key=tob_key)

    return _set_parent(ast.fix_missing_locations(codeast))


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
                                   '__root__.keras.layers.core',
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
                                   '__root__.tensorflow.python.keras.layers',
                                   '__root__.tensorflow.python.keras.layers.core']
_keras_possible_Sequential_api_root = ['__root__.keras.models.Sequential',
                                       '__root__.keras.Sequential',
                                       '__root__.tensorflow.contrib.keras.api.keras.models.Sequential',
                                       '__root__.tensorflow.keras.models.Sequential',
                                       '__root__.tensorflow.keras.Sequential',
                                       '__root__.tensorflow.python.keras.models.Sequential',
                                       '__root__.tensorflow.python.keras.Sequential']
_keras_possible_Model_api_root = ['__root__.bert4keras.backend.keras.models.Model',
                                  '__root__.gandlf.Model',
                                  '__root__.keras.Model',
                                  '__root__.keras.engine.Model',
                                  '__root__.keras.engine.training.Model',
                                  '__root__.keras.models.Model',
                                  '__root__.keras4torch.Model',
                                  '__root__.keras_bert.backend.keras.models.Model',
                                  '__root__.tarantella.Model',
                                  '__root__.tensorflow.compat.v2.keras.models.Model',
                                  '__root__.tensorflow.contrib.keras.models.Model',
                                  '__root__.tensorflow.contrib.keras.python.keras.models.Model',
                                  '__root__.tensorflow.keras.Model',
                                  '__root__.tensorflow.keras.layers.Model',
                                  '__root__.tensorflow.keras.models.Model',
                                  '__root__.tensorflow.python.keras.Model',
                                  '__root__.tensorflow.python.keras._impl.keras.models.Model',
                                  '__root__.tensorflow.python.keras.models.Model',
                                  '__root__.toolkit4nlp.models.Model']
_keras_possible_api_root = list({'.'.join(paths[:paths.index('keras') + 1]) \
                            for root in [*_keras_possible_inits_api_root,
                                         *_keras_possible_activations_api_root,
                                         *_keras_possible_losses_api_root,
                                         *_keras_possible_optimizers_api_root,
                                         *_keras_possible_metrics_api_root,
                                         *_keras_possible_regularizers_api_root,
                                         *_keras_possible_layers_api_root] if 'keras' in (paths := root.split('.'))})
_keras_possible_inits_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_possible_constraints_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_possible_activations_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_possible_losses_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_possible_optimizers_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_possible_metrics_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_possible_regularizers_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_possible_layers_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_possible_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_possible_Sequential_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_possible_Model_api_root.sort(key=lambda x: -len(x.split('.')))  # Try long first
_keras_inits_std_api_root = '__root__.keras.initializers'
_keras_constraints_std_api_root = '__root__.keras.constraints'
_keras_activations_std_api_root = '__root__.keras.activations'
_keras_losses_std_api_root = '__root__.keras.losses'
_keras_optimizers_std_api_root = '__root__.keras.optimizers'
_keras_metrics_std_api_root = '__root__.keras.metrics'
_keras_regularizers_std_api_root = '__root__.keras.regularizers'
_keras_layers_std_api_root = '__root__.keras.layers'
_keras_Sequential_std_api_root = '__root__.keras.models.Sequential'
_keras_Model_std_api_root = '__root__.keras.models.Model'
_keras_std_api_root = '__root__.keras'
_std_keras_api_root_table = [(_keras_possible_inits_api_root, _keras_inits_std_api_root),
                             (_keras_possible_constraints_api_root, _keras_constraints_std_api_root),
                             (_keras_possible_activations_api_root, _keras_activations_std_api_root),
                             (_keras_possible_losses_api_root, _keras_losses_std_api_root),
                             (_keras_possible_optimizers_api_root, _keras_optimizers_std_api_root),
                             (_keras_possible_metrics_api_root, _keras_metrics_std_api_root),
                             (_keras_possible_regularizers_api_root, _keras_regularizers_std_api_root),
                             (_keras_possible_layers_api_root, _keras_layers_std_api_root),
                             (_keras_possible_Sequential_api_root, _keras_Sequential_std_api_root),
                             (_keras_possible_Model_api_root, _keras_Model_std_api_root),
                             (_keras_possible_api_root, _keras_std_api_root)]
_std_keras_api_root_table = tuple((tuple(_make_name_node(name=e) for e in a), (_make_name_node(name=b))) \
                                            for a, b in _std_keras_api_root_table)
def _std_keras_api_root(codeast):
    def name_match(node, pattern) -> ast.Attribute | ast.Name | None:
        if isinstance(node, ast.Name) and isinstance(pattern, ast.Name) and node.id == pattern.id:
            return node
        elif isinstance(node, ast.Attribute) and isinstance(pattern, ast.Attribute):
            if node.attr == pattern.attr and name_match(node.value, pattern.value):
                return node
            return name_match(node.value, pattern)
        return None

    def make_std_root(std_root: ast.Attribute, ctx):
        assert isinstance(std_root, ast.Attribute)
        std_root = _clone_ast(std_root)
        std_root.ctx = ctx
        return std_root

    def per_attribute(node: ast.Attribute):
        for possible_roots, std_root in _std_keras_api_root_table:
            for r in possible_roots:
                if (matched_node := name_match(node, r)) and not _name_equals(matched_node, std_root):
                    assert isinstance(matched_node, ast.Attribute)
                    return matched_node, make_std_root(std_root, matched_node.ctx)
        return None

    def is_long_attribute(x: ast.AST) -> bool:
        return isinstance(x, ast.Attribute) and not isinstance(x.parent, ast.Attribute)

    replace_node_map = {}  # node => new node
    attributes: list[ast.Attribute] = _find_x(codeast, is_x=is_long_attribute)
    for attr in attributes:
        if ret := per_attribute(attr):
            replace_node_map[ret[0]] = ret[1]
    codeast = ReplaceXWithY(x_cls_names=['Attribute'],
                            is_x=lambda x: x in replace_node_map,
                            make_y=lambda x: replace_node_map.pop(x)).visit(codeast)
    codeast = _set_parent(ast.fix_missing_locations(codeast))
    return codeast


# From https://github.com/keras-team/keras/blob/master/keras/...
_keras_padding_names = ['valid', 'same', 'causal']
_keras_initializer_names = ['Zeros', 'Ones', 'RandomUniform', 'TruncatedNormal', 'Identity', 'Orthogonal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'lecun_normal', 'he_uniform']
_keras_constraint_names = ['Constraint', 'MaxNorm', 'NonNeg', 'UnitNorm', 'MinMaxNorm', 'constraint', 'max_norm', 'non_neg', 'unit_norm', 'min_max_norm']
_keras_activation_names = ['elu', 'exponential', 'gelu', 'hard_sigmoid', 'leaky_relu', 'linear', 'log_softmax', 'mish', 'relu', 'relu6', 'selu', 'sigmoid', 'silu', 'softmax', 'softplus', 'softsign', 'tanh']
_keras_dataformat_names = ['channels_first', 'channels_last']
_keras_loss_names = ['BCE', 'MSE', 'MAE', 'MAPE', 'MSLE', 'KLD', 'bce', 'mse', 'mae', 'mape', 'msle', 'kld', 'logcosh', 'huber_loss', 'BinaryCrossentropy', 'CategoricalCrossentropy', 'CategoricalHinge', 'CosineSimilarity', 'Hinge', 'Huber', 'KLDivergence', 'LogCosh', 'LossFunctionWrapper', 'MeanAbsoluteError', 'MeanAbsolutePercentageError', 'MeanSquaredError', 'MeanSquaredLogarithmicError', 'Poisson', 'SparseCategoricalCrossentropy', 'SquaredHinge', 'binary_crossentropy', 'categorical_crossentropy', 'categorical_hinge', 'cosine_proximityLoss', 'cosine_similarity', 'hinge', 'huber', 'kl_divergence', 'kullback_leibler_divergence', 'log_cosh', 'logcosh', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_squared_error', 'mean_squared_logarithmic_error', 'poisson', 'sparse_categorical_crossentropy', 'squared_hinge']
_keras_optimizer_names = ['Adadelta', 'Adafactor', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'Ftrl', 'Lion', 'LossScaleOptimizer', 'Nadam', 'Optimizer', 'RMSprop', 'SGD', 'adadelta', 'adafactor', 'adagrad', 'adam', 'adamax', 'adamw', 'ftrl', 'lion', 'lossscaleoptimizer', 'nadam', 'optimizer', 'rmsprop', 'sgd']
_keras_metric_names = ['accuracy', 'acc', 'crossentropy', 'ce', 'AUC', 'Accuracy', 'BCE', 'BinaryAccuracy', 'BinaryCrossentropy', 'BinaryIoU', 'CategoricalAccuracy', 'CategoricalCrossentropy', 'CategoricalHinge', 'CosineSimilarity', 'F1Score', 'FBetaScore', 'FalseNegatives', 'FalsePositives', 'Hinge', 'IoU', 'KLDivergence', 'LogCoshError', 'MAE', 'MAPE', 'MSE', 'MSLE', 'Mean', 'MeanAbsoluteError', 'MeanAbsolutePercentageError', 'MeanIoU', 'MeanMetricWrapper', 'MeanSquaredError', 'MeanSquaredLogarithmicError', 'Metric', 'OneHotIoU', 'OneHotMeanIoU', 'Poisson', 'Precision', 'PrecisionAtRecall', 'R2Score', 'Recall', 'RecallAtPrecision', 'RootMeanSquaredError', 'SensitivityAtSpecificity', 'SparseCategoricalAccuracy', 'SparseCategoricalCrossentropy', 'SparseTopKCategoricalAccuracy', 'SpecificityAtSensitivity', 'SquaredHinge', 'Sum', 'TopKCategoricalAccuracy', 'TrueNegatives', 'TruePositives', 'accuracy', 'auc', 'bce', 'binary_accuracy', 'binary_crossentropy', 'binary_io_u', 'categorical_accuracy', 'categorical_crossentropy', 'categorical_hinge', 'cosine_similarity', 'f1_score', 'f_beta_score', 'false_negatives', 'false_positives', 'hinge', 'io_u', 'kl_divergence', 'log_cosh_error', 'mae', 'mape', 'mean', 'mean_absolute_error', 'mean_absolute_percentage_error', 'mean_io_u', 'mean_metric_wrapper', 'mean_squared_error', 'mean_squared_logarithmic_error', 'metric', 'mse', 'msle', 'one_hot_io_u', 'one_hot_mean_io_u', 'poisson', 'precision', 'precision_at_recall', 'r2_score', 'recall', 'recall_at_precision', 'root_mean_squared_error', 'sensitivity_at_specificity', 'sparse_categorical_accuracy', 'sparse_categorical_crossentropy', 'sparse_top_k_categorical_accuracy', 'specificity_at_sensitivity', 'squared_hinge', 'sum', 'top_k_categorical_accuracy', 'true_negatives', 'true_positives']
_keras_regularizer_names = ['L1', 'L1L2', 'L2', 'OrthogonalRegularizer', 'Regularizer', 'l1', 'l1l2', 'l2', 'orthogonal_regularizer', 'regularizer']
_keras_model_compile_infix = '.compile('  #FIXME: More strict checking if needed
_keras_model_fit_infix = '.fit('  #FIXME: More strict checking if needed
# Copy from IM4DNN/data_utils.py
def _get_possible_ast_hint(node: ast.AST) -> str:
    # DNN component releated
    if isinstance(node, ast.Constant) and isinstance((val := node.value), str):
        if val in _keras_padding_names: return 'K_padding'
        if val in _keras_initializer_names: return 'K_initializer'
        if val in _keras_constraint_names: return 'K_constraint'
        if val in _keras_activation_names: return 'K_activation'
        if val in _keras_dataformat_names: return 'K_dataformat'
        if val in _keras_loss_names: return 'K_loss'
        if val in _keras_optimizer_names: return 'K_optimizer'
        if val in _keras_metric_names: return 'K_metric'
        if val in _keras_regularizer_names: return 'K_regularizer'
    elif isinstance(node, ast.Call) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
        line = ast.unparse(node)
        if _keras_model_compile_infix in line: return 'K_model_compile'
        if _keras_model_fit_infix in line: return 'K_model_fit'
        if any([line.startswith(p) for p in _keras_possible_layers_api_root]): return 'K_layer'
        if any([line.startswith(p) for p in _keras_possible_inits_api_root]): return 'K_initializer'
        if any([line.startswith(p) for p in _keras_possible_constraints_api_root]): return 'K_constraint'
        if any([line.startswith(p) for p in _keras_possible_losses_api_root]): return 'K_loss'
        if any([line.startswith(p) for p in _keras_possible_optimizers_api_root]): return 'K_optimizer'
        if any([line.startswith(p) for p in _keras_possible_metrics_api_root]): return 'K_metric'
        if any([line.startswith(p) for p in _keras_possible_regularizers_api_root]): return 'K_regularizer'
        # !!!NO!!!: if any([p in line for p in _keras_possible_api_root]): return 'kerasApiCall'
    elif _is_name_node(node):
        line = ast.unparse(node)
        if any([line.startswith(p) for p in _keras_possible_inits_api_root]): return 'K_initializer'
        if any([line.startswith(p) for p in _keras_possible_constraints_api_root]): return 'K_constraint'
        if any([line.startswith(p) for p in _keras_possible_activations_api_root]): return 'K_activation'
        if any([line.startswith(p) for p in _keras_possible_losses_api_root]): return 'K_loss'
        if any([line.startswith(p) for p in _keras_possible_metrics_api_root]): return 'K_metric'
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


# [call/name ->] string -> snake name
def _std_keras_initializer_usage_style(node) -> ast.AST | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if (name := node.func.attr) in _keras_initializer_names:
            return ast.Constant(value=_to_snake_case(name))
    elif _is_name_node(node) and isinstance(node, ast.Attribute):
        if (name := node.attr) in _keras_initializer_names:
            return ast.Constant(value=_to_snake_case(name))
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ast.Constant(value=_to_snake_case(node.value))
    return None


# [call/name ->] string -> snake name
def _std_keras_constraint_usage_style(node) -> ast.AST | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if (name := node.func.attr) in _keras_constraint_names:
            return ast.Constant(value=_to_snake_case(name))
    elif _is_name_node(node) and isinstance(node, ast.Attribute):
        if (name := node.attr) in _keras_constraint_names:
            return ast.Constant(value=_to_snake_case(name))
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ast.Constant(value=_to_snake_case(node.value))
    return None

# name -> string
def _std_keras_activation_usage_style(node) -> ast.AST | None:
    if _is_name_node(node) and isinstance(node, ast.Attribute):
        if (name := node.attr) in _keras_activation_names:
            return ast.Constant(value=_to_snake_case(name))
    return None


# [call/name ->] string -> original name -> snake name
_keras_loss_alias2string = {"bce": 'binary_crossentropy',
                            "BCE": 'binary_crossentropy',
                            "kld": 'kl_divergence',
                            "KLD": 'kl_divergence',
                            "mae": 'mean_absolute_error',
                            "MAE": 'mean_absolute_error',
                            "mse": 'mean_squared_error',
                            "MSE": 'mean_squared_error',
                            "mape": 'mean_absolute_percentage_error',
                            "MAPE": 'mean_absolute_percentage_error',
                            "msle": 'mean_squared_logarithmic_error',
                            "MSLE": 'mean_squared_logarithmic_error'}
def _std_keras_loss_usage_style(node) -> ast.AST | None:
    def to_original_name(name: str):
        return _keras_loss_alias2string.get(name, name)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if (name := node.func.attr) in _keras_loss_names:
            return ast.Constant(value=_to_snake_case(to_original_name(name)))
    elif _is_name_node(node) and isinstance(node, ast.Attribute):
        if (name := node.attr) in _keras_loss_names:
            return ast.Constant(value=_to_snake_case(to_original_name(name)))
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ast.Constant(value=_to_snake_case(to_original_name(node.value)))
    return None


# string -> call; call -> std call
_keras_optimizer_alias2call = {'adam': f'{_keras_optimizers_std_api_root}.Adam(learning_rate=0.001)',
                               'sgd': f'{_keras_optimizers_std_api_root}.SGD(learning_rate=0.01)',
                               'rmsprop': f'{_keras_optimizers_std_api_root}.RMSprop(learning_rate=0.001)',
                               'adadelta': f'{_keras_optimizers_std_api_root}.Adadelta(learning_rate=0.001)',
                               'adamw': f'{_keras_optimizers_std_api_root}.AdamW(learning_rate=0.001)',
                               'adagrad': f'{_keras_optimizers_std_api_root}.Adagrad(learning_rate=0.001)',
                               'adamax': f'{_keras_optimizers_std_api_root}.Adamax(learning_rate=0.001)',
                               'adafactor': f'{_keras_optimizers_std_api_root}.Adafactor(learning_rate=0.001)',
                               'nadam': f'{_keras_optimizers_std_api_root}.Nadam(learning_rate=0.001)',
                               'ftrl': f'{_keras_optimizers_std_api_root}.Ftrl(learning_rate=0.001)',
                               'lion': f'{_keras_optimizers_std_api_root}.Lion(learning_rate=0.0001)'}
_keras_optimizer_alias2defaultlr = {'adam': 0.001,
                                    'sgd': 0.01,
                                    'rmsprop': 0.001,
                                    'adadelta': 0.001,
                                    'adamw': 0.001,
                                    'adagrad': 0.001,
                                    'adamax': 0.001,
                                    'adafactor': 0.001,
                                    'nadam': 0.001,
                                    'ftrl': 0.001,
                                    'lion': 0.0001}
def _std_keras_optimizer_usage_style(node) -> ast.AST | None:
    if isinstance(node, ast.Call):
        has_learning_rate = False
        if len(node.args) == 1:  # learning_rate
            has_learning_rate = True
            node.keywords.append(ast.keyword(arg='learning_rate', value=node.args.pop()))
        elif len(node.args) > 1:
            has_learning_rate = True
        node.keywords.sort(key=lambda kw: '' if kw.arg in ('lr', 'learning_rate') else kw.arg)
        if node.keywords and node.keywords[0].arg in ('lr', 'learning_rate'):
            has_learning_rate = True
            node.keywords[0].arg = 'learning_rate'
        if not has_learning_rate and (deflr := _keras_optimizer_alias2defaultlr.get(node.func.attr, None)):
            node.keywords.insert(0, ast.keyword(arg='learning_rate', value=ast.Constant(value=deflr)))
    if isinstance(node, ast.Constant) and (call := _keras_optimizer_alias2call.get(node.value, None)):
        return ast.parse(call, mode='eval').body
    return None


# [call/name ->] string -> original name -> snake name
_keras_metric_alias2string = {"bce": 'BinaryCrossentropy', 
                              "BCE": 'BinaryCrossentropy', 
                              "mse": 'MeanSquaredError', 
                              "MSE": 'MeanSquaredError', 
                              "mae": 'MeanAbsoluteError', 
                              "MAE": 'MeanAbsoluteError', 
                              "mape": 'MeanAbsolutePercentageError', 
                              "MAPE": 'MeanAbsolutePercentageError', 
                              "msle": 'MeanSquaredLogarithmicError', 
                              "MSLE": 'MeanSquaredLogarithmicError'}
def _std_keras_metric_usage_style(node) -> ast.AST | None:
    def to_original_name(name: str):
        return _keras_metric_alias2string.get(name, name)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if (name := node.func.attr) in _keras_metric_names:
            return ast.Constant(value=_to_snake_case(to_original_name(name)))
    elif _is_name_node(node) and isinstance(node, ast.Attribute):
        if (name := node.attr) in _keras_metric_names:
            return ast.Constant(value=_to_snake_case(to_original_name(name)))
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        return ast.Constant(value=_to_snake_case(to_original_name(node.value)))
    return None


_std_keras_usage_style_trans_fn = {'K_initializer': _std_keras_initializer_usage_style,  # call/name -> string -> snake name
                                   'K_constraint': _std_keras_constraint_usage_style,    # call/name -> string -> snake name
                                   'K_activation': _std_keras_activation_usage_style,    # name -> string
                                   'K_loss': _std_keras_loss_usage_style,                # call/name -> string -> original name -> snake name
                                   'K_optimizer': _std_keras_optimizer_usage_style,      # string -> call; call -> std call
                                   'K_metric': _std_keras_metric_usage_style}            # call/name -> string -> original name -> snake name
def _std_keras_usage_style(codeast):
    replace_node_map = {}  # node => new node
    for node in ast.walk(codeast):
        if fn := _std_keras_usage_style_trans_fn.get(_get_possible_ast_hint(node), None):
            if new_node := fn(node):
                replace_node_map[node] = new_node
    codeast = ReplaceXWithY(x_cls_names=[e.__class__.__name__ for e in replace_node_map.keys()],
                            is_x=lambda x: x in replace_node_map,
                            make_y=lambda x: replace_node_map.pop(x)).visit(codeast)
    codeast = _set_parent(ast.fix_missing_locations(codeast))
    return codeast


def _make_module_table(roots: str):
    roots = [os.path.abspath(root) for root in roots]
    return ModuleTable.make(roots)


def _make_func_table(codeast):
    return FuncTable.make(codeast)


def _inline_call_gen_argvarinits(call: ast.Call, funcdef: ast.FunctionDef):
    assert isinstance(call, ast.Call)
    assert isinstance(funcdef, ast.FunctionDef)
    funcdef_paramlist = funcdef.args
    pos_params = funcdef_paramlist.posonlyargs + funcdef_paramlist.args  # list of ast.arg
    vararg_param = funcdef_paramlist.vararg  # ast.arg; *args
    kw_params = funcdef_paramlist.args + funcdef_paramlist.kwonlyargs  # list of ast.arg
    kwarg_param = funcdef_paramlist.kwarg  # ast.arg; **kwargs
    posargs = call.args
    kwargs = call.keywords
    arg_vars, arg_var_init_stmts = set(), []
    assert len(funcdef_paramlist.defaults) <= len(funcdef_paramlist.args)
    for i in range(-1, -(len(funcdef_paramlist.defaults) + 1), -1):
        ass = ast.Assign(
            targets=[ast.Name(id=funcdef_paramlist.args[i].arg, ctx=ast.Store())],
            value=funcdef_paramlist.defaults[i],
            type_comment=None)
        arg_vars.add(ass.targets[0].id)
        arg_var_init_stmts.append(ass)
    assert len(funcdef_paramlist.kwonlyargs) == len(funcdef_paramlist.kw_defaults)
    for i in range(len(funcdef_paramlist.kwonlyargs)):
        if funcdef_paramlist.kw_defaults[i] is not None:
            ass = ast.Assign(
                targets=[ast.Name(id=funcdef_paramlist.kwonlyargs[i].arg, ctx=ast.Store())],
                value=funcdef_paramlist.kw_defaults[i],
                type_comment=None)
            arg_vars.add(ass.targets[0].id)
            arg_var_init_stmts.append(ass)
    for i in range(min(len(posargs), len(pos_params))):
        ass = ast.Assign(
            targets=[ast.Name(id=pos_params[i].arg, ctx=ast.Store())],
            value=posargs[i],
            type_comment=None)
        arg_vars.add(ass.targets[0].id)
        arg_var_init_stmts.append(ass)
    if len(posargs) <= len(pos_params):
        if vararg_param is not None:
            ass = ast.Assign(
                targets=[ast.Name(id=vararg_param.arg, ctx=ast.Store())],
                value=ast.List(elts=[], ctx=ast.Load()),
                type_comment=None)
            arg_vars.add(ass.targets[0].id)
            arg_var_init_stmts.append(ass)
    else:  # len(posargs) > len(pos_params)
        if vararg_param is None: return None, None
        rem_posargs = posargs[len(pos_params):]
        ass = ast.Assign(
            targets=[ast.Name(id=vararg_param.arg, ctx=ast.Store())],
            value=ast.List(elts=rem_posargs, ctx=ast.Load()),
            type_comment=None)
        arg_vars.add(ass.targets[0].id)
        arg_var_init_stmts.append(ass)
    kw_param_names = {e.arg for e in kw_params}
    rem_kwargs = []
    for kw in kwargs:
        if kw.arg in kw_param_names:
            ass = ast.Assign(
                targets=[ast.Name(id=kw.arg, ctx=ast.Store())],
                value=kw.value,
                type_comment=None)
            arg_vars.add(ass.targets[0].id)
            arg_var_init_stmts.append(ass)
        else:
            rem_kwargs.append(kw)
    if rem_kwargs:
        if kwarg_param is None: return None, None
        rem_kwargs_map = {ast.Constant(value=e.arg): e.value for e in rem_kwargs}
        ass = ast.Assign(
            targets=[ast.Name(id=kwarg_param.arg, ctx=ast.Store())],
            value=ast.Dict(
                keys=list(rem_kwargs_map.keys()),
                values=list(rem_kwargs_map.values())),
            type_comment=None)
        arg_vars.add(ass.targets[0].id)
        arg_var_init_stmts.append(ass)
    elif kwarg_param is not None:
        ass = ast.Assign(
            targets=[ast.Name(id=kwarg_param.arg, ctx=ast.Store())],
            value=ast.Dict(
                keys=[],
                values=[]),
            type_comment=None)
        arg_vars.add(ass.targets[0].id)
        arg_var_init_stmts.append(ass)

    return arg_vars, arg_var_init_stmts


def _inline_import(
        import_, 
        id_suffix_maker,
        module_table: ModuleTable, 
        recursive=True, 
        max_tries=512, 
        print_progress=False,
        rename_with_more_info=False):
    assert callable(id_suffix_maker)
    mod: ImportedModule = module_table.lookup(import_)

    if mod is None:
        return None, None, None

    if mod.symbol_map:
        # The module has been inlined
        return [ast.Pass()], mod.symbol_map, mod

    if recursive:
        # _set_parent -> remove ifEqMs -> _inline_all_imports
        mod.module = _set_parent(mod.module)
        ifNEqMs = _find_if_name_eq_main(mod.module)
        mod.module = ReplaceXWithY(
            x_cls_names=['If'],
            is_x=lambda x: x in ifNEqMs,
            make_y=lambda n: n.orelse).visit(mod.module)
        mod.module, _, _ = _inline_all_imports(
            codeast=mod.module,
            module_table=module_table,
            idx_maker=id_suffix_maker,
            recursive=recursive,
            max_tries=max_tries - 1,
            print_progress=print_progress,
            rename_with_more_info=rename_with_more_info)

    all_stmts = _clone_ast(mod.module.body)
    symbol_map = {}  # origin => renamed

    # Rename local_ids with '..._suffix'
    id_suffix = id_suffix_maker()
    mag_name = lambda n: f'{n}_T{id_suffix}'
    local_id2nodes: dict = _get_local_id2nodes(all_stmts, ignores=module_table.all_module_glbs)  # id => (node => seq)
    for i, p in local_id2nodes.items():  # Change nodes in place
        i_l = i.split('.')
        if len(i_l) > 1: # Only rename the toplevel locals
            continue
        origin, renamed = i, mag_name(i)
        symbol_map[origin] = renamed
        for node in p.keys():
            if isinstance(node, ast.Name):
                assert node.id == i_l[-1]
                node.id = mag_name(node.id)
            elif isinstance(node, ast.FunctionDef):
                assert node.name == i_l[-1]
                node.name = mag_name(node.name)
            elif isinstance(node, ast.ClassDef):
                assert node.name == i_l[-1]
                node.name = mag_name(node.name)
            elif isinstance(node, ast.alias) and i_l[-1] != '*':
                assert (node.asname or node.name) == i_l[-1]
                node.asname = mag_name(node.asname or node.name)
            elif isinstance(node, ast.arg):
                assert node.arg == i_l[-1]
                node.arg = mag_name(node.arg)

    mod.symbol_map = symbol_map  # Mark the module as imported&inlined and Set its symbol_map
    return all_stmts or [ast.Pass()], symbol_map, mod


def _inline_call(call: ast.Call, id_suffix_maker, func_table: FuncTable):
    assert callable(id_suffix_maker)
    funcdef = func_table.lookup(call.func)
    if funcdef is None or \
       _find_x(funcdef, is_x=lambda n: isinstance(n, (ast.Yield, ast.YieldFrom)), ignore_sub_scope=True):
        # DON'T INLINE: no funcdef or `yield` in the funcdef
        return None, None, None  # stmts, arg_var_ids, ret_var_id

    # print('call:\n', ast.unparse(call))
    # print('funcdef: ', ast.unparse(funcdef))

    # Construct arg vars
    arg_var_ids, arg_var_init_stmts = _inline_call_gen_argvarinits(call, funcdef)
    if arg_var_ids is None: return None, None, None  # stmts, arg_var_ids, ret_var_id
    # Construct ret var
    ret_var_id = f'r2ms_icall_ret'
    ret_var_init_stmt = ast.Assign(
        targets=[ast.Name(id=ret_var_id, ctx=ast.Store())],
        value=ast.Constant(value=None),
        type_comment=None)

    # Extract body
    body = _clone_ast(funcdef.body)
    def make_y(ret: ast.Return):
        return ast.Assign(
            targets=[ast.Name(id=ret_var_id, ctx=ast.Store())],
            value=ret.value if ret.value else ast.Constant(value=None),
            type_comment=None)
    rrwx = ReplaceXWithY(x_cls_names=['Return'], is_x=lambda n: True, make_y=make_y, ignore_parents=['FunctionDef'])
    body = [rrwx.visit(e) for e in body]
    all_stmts = [*arg_var_init_stmts, ret_var_init_stmt, *body]

    # Rename local_ids with '..._suffix'
    id_suffix = id_suffix_maker()
    mag_name = lambda n: f'{n}_T{id_suffix}'
    local_id2nodes: dict = _get_local_id2nodes(all_stmts)  # id => (node => seq)
    assert not ({*arg_var_ids, ret_var_id} - local_id2nodes.keys())
    for i, p in local_id2nodes.items():  # Change nodes in place
        i_l = i.split('.')
        if len(i_l) > 1: # Only rename the toplevel locals
            continue
        for node in p.keys():
            if isinstance(node, ast.Name):
                assert node.id == i_l[-1]
                node.id = mag_name(node.id)
            elif isinstance(node, ast.FunctionDef):
                assert node.name == i_l[-1]
                node.name = mag_name(node.name)
            elif isinstance(node, ast.ClassDef):
                assert node.name == i_l[-1]
                node.name = mag_name(node.name)
            elif isinstance(node, ast.alias) and i_l[-1] != '*':
                assert (node.asname or node.name) == i_l[-1]
                node.asname = mag_name(node.asname or node.name)
            elif isinstance(node, ast.arg):
                assert node.arg == i_l[-1]
                node.arg = mag_name(node.arg)

    # for stmt in all_stmts:
    #     print(ast.unparse(ast.fix_missing_locations(stmt)))
    arg_var_ids = {mag_name(i) for i in arg_var_ids}
    ret_var_id = mag_name(ret_var_id)
    return all_stmts, arg_var_ids, ret_var_id


def _find_arg_from_call(node: ast.Call, pos: int | None, key: str | None) -> ast.AST | None:
    assert isinstance(node, ast.Call)
    if pos is not None and pos < len(node.args):
        return node.args[pos]
    elif key is not None:
        for kw in node.keywords:
            if key == kw.arg:
                return kw.value
    return None


def _split_import(import_):
    names = import_.names
    if len(names) == 1: return [import_]
    if isinstance(import_, ast.Import):
        return [ast.Import(names=[n]) for n in names]
    elif isinstance(import_, ast.ImportFrom):
        return [ast.ImportFrom(module=import_.module, names=[n], level=import_.level) for n in names]


def _inline_all_imports(
        codeast, 
        module_table, 
        idx_maker, 
        recursive=True, 
        max_tries=512, 
        print_progress=False, 
        rename_with_more_info=False):
    assert callable(idx_maker)
    if max_tries <= 0: return codeast, 0, []

    if print_progress:
        log = lambda *args, **kwargs: print(*[arg() for arg in args], **{k: arg() for k, arg in kwargs.items()})
    else:
        log = lambda *args, **kwargs: None
    
    gi: GetRWRefIDs = _get_rwref_ids(codeast, get_all=True)
    get_1st_init_seq = lambda name: gi.initid_nodes.get(name, ((gi.current_cnt + 1,),))[0][0]
    init_node2seqft = {}  # node => seq # = {n[1][0]: n[0] for nodes in gi.initid_nodes.values() for n in nodes}
    for init_nodes in gi.initid_nodes.values():
        init_nodes = sorted(init_nodes, key=lambda e: e[0])
        for i, (seq, (node, inode)) in enumerate(init_nodes):
            try:
                init_node2seqft[node] = (seq, init_nodes[i + 1][0])
            except IndexError:
                init_node2seqft[node] = (seq, gi.current_cnt + 1)
    rnode2seq = {node: seq for rid_nodes in gi.rid_nodes.values() for seq, (node, rnode) in rid_nodes}
    wnode2seq = {node: seq for wid_nodes in gi.wid_nodes.values() for seq, (node, wnode) in wid_nodes}
    node2seq = {**rnode2seq, **wnode2seq}

    imports = _find_imports(codeast)
    import2si = {}  # import => [(single-import-0, (lowseq0, highseq0)), ...]
    for i in imports:
        si_list = []
        for si in _split_import(i):
            assert len(si.names) == 1
            alias: ast.alias = si.names[0]
            lseq, hseq = init_node2seqft[alias]
            si_list.append((si, (lseq, hseq)))
        import2si[i] = si_list

    if rename_with_more_info:
        def soft_module_name(n):
            assert len(n.names) == 1
            alias: ast.alias = n.names[0]
            if isinstance(n, ast.Import):
                # import a as b: "b"
                return (alias.asname or alias.name).replace('.', 'S') 
            elif isinstance(n, ast.ImportFrom):
                # from ..a import b: "PPa_b"
                return (f'{"P"*n.level}{n.module or "_"}_{alias.asname or alias.name}').replace('.', 'S')
    else:
        def soft_module_name(n):
            return ''

    replace_node_map = {}  # node => [stmt0, stmt1, ...]
    inlined_imports = []
    imported_symbol_maps = []  # [(single-import, {original => renamed}), ...]
    for original_import, si_list in import2si.items():
        inlined_stmts_of_oi = []
        for single_import, (lseq, hseq) in si_list:
            inlined_stmts, glb_symbol_map, module = _inline_import(
                import_=single_import,
                id_suffix_maker=lambda: f'{soft_module_name(single_import)}{idx_maker()}',
                module_table=module_table,
                recursive=recursive,
                max_tries=max_tries,
                print_progress=print_progress,
                rename_with_more_info=rename_with_more_info)
            log(lambda: f'Inline "{ast.unparse(ast.fix_missing_locations(single_import))}":\n'
                        f'\thas_been_imported/inlined={bool(not inlined_stmts)},\n'
                        f'\tsymbol_map={glb_symbol_map}')
            if inlined_stmts is not None:
                assert len(inlined_stmts) > 0
                inlined_stmts_of_oi.extend(inlined_stmts)
                inlined_imports.append(single_import)
                imported_symbol_maps.append((single_import, (lseq, hseq), glb_symbol_map, module))
            else:
                inlined_stmts_of_oi.append(single_import)
        replace_node_map[original_import] = inlined_stmts_of_oi
    codeast = ReplaceXWithY(
        x_cls_names={x.__class__.__name__ for x in replace_node_map.keys()},
        is_x=lambda n: n in replace_node_map,
        make_y=lambda n: replace_node_map[n]).visit(codeast)

    # Rename imported ids
    replace_node_map = {}  # node => new_node
    for si, (lseq, hseq), symbol_map, imp_mod in imported_symbol_maps:
        imported_symbol, imported_id = _get_imported_symbol_and_id(si)
        if imported_symbol == imp_mod.fullname:  # imported_symbol is module
            def is_x(x):
                # <imported_id>.xxx...
                return isinstance(x, ast.Attribute) and \
                        isinstance(x.value, ast.Name) and \
                        x.value.id == imported_id and \
                        x.attr in symbol_map and \
                        x.value in node2seq and \
                        lseq <= node2seq[x.value] < hseq
            for attr in _find_x(codeast, is_x=is_x):
                replace_node_map[attr] = ast.Name(id=symbol_map[attr.attr], ctx=attr.ctx)
        else:  # imported_symbol is symbol
            if imported_id == '*':
                def is_x(x):
                    # <imported_id>...
                    return isinstance(x, ast.Name) and \
                            x.id in symbol_map and \
                            x in node2seq and \
                            lseq <= node2seq[x] < get_1st_init_seq(x.id)
                for name in _find_x(codeast, is_x=is_x):
                    name.id = symbol_map[name.id]
            else:
                def is_x(x):
                    # <imported_id>...
                    return isinstance(x, ast.Name) and \
                            x.id == imported_id and \
                            x in node2seq and \
                            lseq <= node2seq[x] < hseq
                if (rel_i_sym := _get_relname(imported_symbol, start=imp_mod.fullname)) in symbol_map:
                    for name in _find_x(codeast, is_x=is_x):
                        name.id = symbol_map[rel_i_sym]
    codeast = ReplaceXWithY(
        x_cls_names={x.__class__.__name__ for x in replace_node_map.keys()},
        is_x=lambda n: n in replace_node_map,
        make_y=lambda n: replace_node_map[n]).visit(codeast)
    # Fix codeast (locs & set-parent)
    codeast = _set_parent(ast.fix_missing_locations(codeast))

    return codeast, len(inlined_imports), inlined_imports


# NOTE: Required: After `_trans_comp_to_loop`
def _inline_all_calls(
        codeast, 
        func_table: FuncTable, 
        idx_maker, 
        recursive=True, 
        max_tries=512, 
        rename_with_more_info=False): # -> codeast, int
    assert callable(idx_maker)
    codeast = _set_depth(codeast)
    calls = _find_calls(codeast, recursive=True, ignore_parents=['FunctionDef'])
    calls.sort(key=lambda e: -e._pg_depth)  # sort by -depth; rq:stable-sort
    soft_func_name = (lambda n: n.func.id if isinstance(n.func, ast.Name) else '') \
                        if rename_with_more_info else (lambda n: '')
    replace_node_map = {}  # node => [stmt0, stmt1, ...]
    inlined_calls = []
    for c in calls:
        inlined_stmts, arg_var_ids, ret_var_id = _inline_call(
            call=c,
            id_suffix_maker=lambda: f'{soft_func_name(c)}{idx_maker()}',
            func_table=func_table)
        if inlined_stmts is None: continue
        block, c_pstmt = _get_nearest_block_stmt(c)
        # c_pstmt -> [*inlined_stmts, replace_call_with_retvar(c_pstmt, ...)]
        proced_c_pstmt = ReplaceXWithY(
            x_cls_names=['Call'],
            is_x=lambda n: n == c,
            make_y=lambda n: ast.Name(id=ret_var_id, ctx=ast.Load())).visit(c_pstmt)
        if c_pstmt not in replace_node_map:
            replace_node_map[c_pstmt] = []
        replace_node_map[c_pstmt].extend(inlined_stmts)
        inlined_calls.append(c)
    assert len(inlined_calls) >= len(replace_node_map)
    codeast = ReplaceXWithY(
        x_cls_names={x.__class__.__name__ for x in replace_node_map.keys()},
        is_x=lambda n: n in replace_node_map,
        make_y=lambda n: [*replace_node_map[n], n]).visit(codeast)
    # Fix codeast (locs & set-parent)
    codeast = _set_parent(ast.fix_missing_locations(codeast))
    # Inline calls in the codeast recursively
    ch_cnt = len(inlined_calls)
    if recursive:
        for _ in range(max_tries):
            func_table.update(codeast)
            codeast, ch_cnt, ics = _inline_all_calls(codeast, func_table, idx_maker, False, None, rename_with_more_info)
            if ch_cnt == 0: break
            inlined_calls.extend(ics)

    return codeast, ch_cnt, inlined_calls


def _rename_imported_id_to_fullname(codeast, remove_imports: bool, root_mark: str = None):
    root_mark = f'{root_mark}.' if root_mark else ''
    gi = _get_rwref_ids(codeast, get_all=True)
    node_and_seq = {}  # name => [(node, seq, xnode, r/w/i), ...]
    for name in gi.wids | gi.rids | gi.initids:
        node_and_seq[name] = sorted([
            *[(e[1][0], e[0], e[1][1], 'r') for e in gi.rid_nodes.get(name, [])],
            *[(e[1][0], e[0], e[1][1], 'w') for e in gi.wid_nodes.get(name, [])],
            *[(e[1][0], e[0], e[1][1], 'i') for e in gi.initid_nodes.get(name, [])],
        ], key=lambda e: e[1])
    imported_id_to_fullname = {}  # imported_id => prefix
    replace_node_map = {}  # node => new-node
    imports_canbe_removed = []
    for name, node_and_seq_of_name in node_and_seq.items():
        for node, seq, xnode, t in node_and_seq_of_name:
            if t == 'i':
                if isinstance(xnode, ast.Import):
                    # import a.b.c as d
                    ## d => a.b.c
                    imports_canbe_removed.append(xnode)
                    for n in xnode.names:
                        imported_id = n.asname or n.name.split('.')[0]
                        fullname = n.name
                        imported_id_to_fullname[imported_id] = f'{root_mark}{fullname}'
                elif isinstance(xnode, ast.ImportFrom) and xnode.level == 0:  # ignore `from .xxx import ...`
                    # from a.b.c import d as D
                    ## D => a.b.c.d
                    if len(xnode.names) != 1 or xnode.names[0].name != '*':  # ignore `from a.b.c import *`
                        imports_canbe_removed.append(xnode)
                        for n in xnode.names:
                            imported_id = n.asname or n.name
                            fullname = f'{xnode.module}.{n.name}'
                            imported_id_to_fullname[imported_id] = f'{root_mark}{fullname}'
                else:
                    imported_id_to_fullname.pop(name, None)
            elif isinstance(node, ast.Name):
                if node.id in imported_id_to_fullname:
                    replace_node_map[node] = _make_name_node(name=imported_id_to_fullname[node.id], ctx=node.ctx)
    if remove_imports:
        for imp in imports_canbe_removed:
            replace_node_map[imp] = ast.Pass()  ##TODO: DON'T ALWAYS
    codeast = ReplaceXWithY(
        x_cls_names={x.__class__.__name__ for x in replace_node_map.keys()},
        is_x=lambda n: n in replace_node_map,
        make_y=lambda n: replace_node_map[n]).visit(codeast)
    # Fix codeast (locs & set-parent)
    codeast = _set_parent(ast.fix_missing_locations(codeast))
    return codeast


#+ repo2model_s

def _make_program_fragments(
        codeast: ast.Module,
        glb_ignores,
        cared_ast_checkers: list,
        cared_ast_transformer,
        print_progress: bool = False,
        spec_cared_ast = None) -> list:
    if cared_ast_transformer is None: return []
    assert spec_cared_ast or all([callable(f) for f in cared_ast_checkers])
    assert callable(cared_ast_transformer)
    glb_ignores = glb_ignores or set()

    def is_cared_stmt(x) -> bool:
        for ck in cared_ast_checkers:
            if ck(x): return True
        return False

    nodes = spec_cared_ast or _find_x(codeast, is_x=is_cared_stmt, ignore_sub_scope=True)
    if not nodes:
        raise R2MSError(f'Make program fragment: failed to find cared nodes')

    for node in nodes:
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        import_ids = _get_ids(_find_imports(codeast))
        body: list = codeast.body

        for p in _iter_parent_chain(node):
            try:
                index = body.index(p)
                break
            except ValueError as ex:
                pass

        stmt_list = []
        id_set = set()

        (should_include_cared_stmt,
         t_node,
         ignore_ids_fc,
         fragment_maker,
         fragment_checker,
         next_cared_ast_transformer) = cared_ast_transformer(node)
        ignore_ids = {*glb_ignores, *import_ids, *ignore_ids_fc}

        cared_stmt = _set_node_id(codeast.body[index])
        cared_stmt = ReplaceXWithY(
            x_cls_names=[node.__class__.__name__],
            is_x=lambda n: n._pg_node_id == node._pg_node_id,
            make_y=lambda n: t_node).visit(_set_node_id(_clone_ast(cared_stmt)))
        if should_include_cared_stmt:
            stmt_list.append(cared_stmt)
        id_set.update(_get_ids(cared_stmt, ignores=ignore_ids))
        for stmt in reversed(body[: index]):
            rids, wids, initids, allr_after_i_ids = _get_rwref_ids(stmt, recursive=False, ignores=ignore_ids)
            stmt_src = ast.unparse(ast.fix_missing_locations(stmt))
            if (wids & id_set) or \
                '__mask_0__' in stmt_src or \
                'seed(' in  stmt_src:
            # if (wids & id_set):
                if print_progress:
                    print('CARE: ', ast.unparse(stmt), f'r{rids}, w{wids}, i{initids}\n\t{id_set}')
                stmt_list.append(_clone_ast(stmt))
                # id_set = id_set - initids + (rids - (rids{>}initids)) + (wids - initids)
                id_set -= initids
                id_set.update(rids - allr_after_i_ids)
                id_set.update(wids - initids)
            else:
                if print_progress:
                    print('XXXX: ', ast.unparse(stmt), f'r{rids}, w{wids}, i{initids}\n\t{id_set}')
        stmt_list.reverse()

        possible_fragment = fragment_maker(stmt_list)
        if fragment_checker(possible_fragment):
            return [
                possible_fragment,
                *_make_program_fragments(codeast,
                                         glb_ignores=glb_ignores,
                                         cared_ast_checkers=cared_ast_checkers,
                                         cared_ast_transformer=next_cared_ast_transformer,
                                         print_progress=print_progress,
                                         spec_cared_ast=[node]),
            ]

    raise R2MSError(f'Make program fragment: failed to verify the program')


def _split_model_data_context(codeast: ast.Module) -> list[ast.Module]:
    cared_api_calls = ['keras_fit', 'keras_fit_generator', 'keras_train_on_batch']
    cared_ast_transformer = 'keras_program'
    cared_ast_checkers = [eval(f'_AST_CHECKER_{ck_name}') for ck_name in cared_api_calls]
    cared_ast_transformer = eval(f'_CARED_AST_TRANSFORMER_{cared_ast_transformer}')
    codeast = _set_parent(ast.fix_missing_locations(codeast))
    fragments = _make_program_fragments(codeast,
                                    glb_ignores={'__root__'},
                                    cared_ast_checkers=cared_ast_checkers,
                                    cared_ast_transformer=cared_ast_transformer,
                                    print_progress=False)
    module_body = []
    for frag in fragments:
        if isinstance(frag, ast.Module): module_body.extend(frag.body)
        else: module_body.append(frag)
    assert len(module_body) == 2
    assert isinstance(module_body[0], ast.FunctionDef) and module_body[0].name == 'train_model'
    assert isinstance(module_body[1], ast.FunctionDef) and module_body[1].name == 'get_data'
    return [ast.Module(body=[b], type_ignores=[]) for b in module_body]


def _main(code: str, import_root: str, configs: dict) -> str:
    assert isinstance(configs, dict)
    import_root = os.path.abspath(import_root)

    def _make_idx_maker():
        c = [-1]
        def maker():
            c[0] += 1
            return c[0]
        return maker

    # Parse code as codeast
    codeast = _set_parent(ast.parse(code))
    glb_idx_maker = _make_idx_maker()
    # Flat the code
    ## Remove&Untab `if __name__ == '__main__': ...`
    if configs['ENABLE_force_untab_ifNEqM_block']:
        codeast = _force_untab_block(codeast, _find_if_name_eq_main(codeast))
    ## Inline imports
    if configs['ENABLE_inline_imports']:
        codeast, latest_ch_cnt, inlined_imports = _inline_all_imports(codeast,
                                                                      module_table=_make_module_table(
                                                                          roots=[
                                                                              *configs['CONFIG_import_roots'],
                                                                              import_root
                                                                          ]
                                                                      ),
                                                                      idx_maker=glb_idx_maker,
                                                                      recursive=True,
                                                                      max_tries=configs['CONFIG_inline_imports_max_tries'],
                                                                      print_progress=configs['ENABLE_print_inline_imports_progress'],
                                                                      rename_with_more_info=configs['ENABLE_rename_with_more_info'])
    ## Flat the non-vatomic-stmt(s): _trans_comp_to_loop, _std_calls; NOTE: DO NOT CHANE ORDER OF THE PASSES
    if configs['ENABLE_trans_comp_to_loop']:
        codeast = _trans_comp_to_loop(codeast, idx_maker=glb_idx_maker)
    if configs['ENABLE_std_call']:
        codeast = _std_calls(codeast, idx_maker=glb_idx_maker)
    ## Inline calls
    inlined_calls = []
    if configs['ENABLE_inline_calls']:
        codeast, latest_ch_cnt, inlined_calls = _inline_all_calls(codeast,
                                                                  func_table=_make_func_table(codeast),
                                                                  idx_maker=glb_idx_maker,
                                                                  recursive=True,
                                                                  max_tries=configs['CONFIG_inline_calls_max_tries'],
                                                                  rename_with_more_info=configs['ENABLE_rename_with_more_info'])
    if configs['ENABLE_rename_imported_id_to_fullname']:
        codeast = _rename_imported_id_to_fullname(codeast, 
                                                  remove_imports=configs['ENABLE_remove_imports_after_renaming_imported_id_to_fullname'],
                                                  root_mark=configs['CONFIG_rename_imported_id_root_mark'])
    ## Remove all single-name (ast.Expr(ast.Name()/ast.Constant()/ast.Attribute(value=ast.Name()/ast.Attribute()...)))
    if configs['ENABLE_remove_useless_stmts']:
        codeast = _remove_useless_stmts(codeast, level=0)
    ## Inline vars & Fold constants
    if configs['ENABLE_inline_vars']:
        codeast = _inline_all_vars_l1(codeast)
    if configs['ENABLE_inline_vars_l2']:
        codeast = _inline_all_vars_l2(codeast)
    ## Unroll for loop
    if configs['ENABLE_unroll_for_loop']:
        ...# TODO: codeast = _unroll_for_loop(codeast)
    if configs['ENABLE_print_flatted_code']:
        print(f'Original code:\n"{code}"\n\nFlatted code:\n"{ast.unparse(codeast)}"')
    # Make `get_dataset` and `train_model`
    cared_ast_checkers = [eval(f'_AST_CHECKER_{ck_name}') for ck_name in configs['CONFIG_cared_api_calls']]
    cared_ast_transformer = eval(f'_CARED_AST_TRANSFORMER_{configs["CONFIG_cared_ast_transformer"]}')
    fragments = _make_program_fragments(codeast,
                                        glb_ignores={
                                            configs['CONFIG_rename_imported_id_root_mark'],
                                        # } | {
                                            # c.func.id
                                            # for c in inlined_calls
                                            # if isinstance(c.func, ast.Name)
                                        },
                                        cared_ast_checkers=cared_ast_checkers,
                                        cared_ast_transformer=cared_ast_transformer,
                                        print_progress=configs['ENABLE_print_clip_process'])
    module_body = []
    for frag in fragments:
        if isinstance(frag, ast.Module): module_body.extend(frag.body)
        else: module_body.append(frag)
    codeast = ast.Module(body=module_body, type_ignores=[])
    # Post process
    ## 1._norm_local_ids
    if configs['ENABLE_norm_local_ids']:
        def _make_id_maker():
            mk_idx = _make_idx_maker()
            return lambda: f'var{mk_idx()}'
        funcdef_names = {f.name for f in codeast.body if isinstance(f, ast.FunctionDef)}
        codeast = _norm_local_ids(codeast, id_maker=_make_id_maker(), ignores=funcdef_names)
    ## 2. _std_api_call_kwargs (_sort_call_kwargs)
    if configs['ENABLE_std_api_call_kwargs']:
        codeast = _sort_call_kwargs(codeast)
    ## 2. _rename_model_id
    if configs['ENABLE_rename_model_id']:
        ...# TODO: codeast = _rename_model_id(codeast, name='model')
    ## 3. _std_keras_compile_and_fit_kwargs
    if configs['ENABLE_std_keras_compile_and_fit_kwargs']:
        codeast = _std_keras_compile_and_fit_kwargs(codeast)
    ## 4. _std_keras_api_root
    if configs['ENABLE_std_keras_api_root']:
        codeast = _std_keras_api_root(codeast)
    ## 5. _std_keras_usage_style
    if configs['ENABLE_std_keras_usage_style']:
        codeast = _std_keras_usage_style(codeast)
    # Unparse & Return the changed source code
    changed_code = ast.unparse(ast.fix_missing_locations(codeast))
    if configs['ENABLE_print_changed_code']:
        print(f'Changed code:\n"{changed_code}"')
    return changed_code


def main(entry_file_path: str, out_file_path: str, configs) -> int:
    try:
        entry_file_path = os.path.abspath(entry_file_path)
        out_file_path = os.path.abspath(out_file_path)
        with open(entry_file_path, 'r', encoding='UTF-8') as fp:
            code = fp.read()
        changed_code = _main(code, _get_import_root(entry_file_path), configs=configs)
        with open(out_file_path, 'w', encoding='UTF-8') as fp:
            fp.write(changed_code)
        return 0
    except Exception as ex:
        import traceback
        traceback.print_exc(file=sys.stdout)
        return -1


# AST Checkers (!!! DO NOT DELETE THEM !!!)
_AST_CHECKER_any = lambda n: True
_AST_CHECKER_keras_fit = lambda n: _check_call_member_fn('fit', n)
_AST_CHECKER_keras_fit_generator = lambda n: _check_call_member_fn('fit_generator', n)
_AST_CHECKER_keras_train_on_batch = lambda n: _check_call_member_fn('train_on_batch', n)
_AST_CHECKER_pt_step = lambda n: _check_call_member_fn('step', n, nargs=0, kwargkeys=[])

# Cared AST Transformer (!!! DO NOT DELETE THEM !!!)
## -> should_include_cared_stmt, fixed_node, ignore_ids_fc, fragment_maker, fragment_checker, next_cared_ast_transformer
def _CARED_AST_TRANSFORMER_keras_program(node: ast.Call):
    assert isinstance(node, ast.Call)
    node = _clone_ast(node)

    ##TODO: Consider different version

    # keras.Model.fit(
    #     x=None,
    #     y=None,
    #     batch_size=None,
    #     epochs=1,
    #     verbose="auto",
    #     callbacks=None,
    #     validation_split=0.0,
    #     validation_data=None,
    #     shuffle=True,
    #     class_weight=None,
    #     sample_weight=None,
    #     initial_epoch=0,
    #     steps_per_epoch=None,
    #     validation_steps=None,
    #     validation_batch_size=None,
    #     validation_freq=1,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False,
    # )
    FIT_X_POS_AND_KEY = (0, 'x')
    FIT_Y_POS_AND_KEY = (1, 'y')
    FIT_VDATA_POS_AND_KEY = (7, 'validation_data')
    FIT_CALLBACKS_POS_AND_KEY = (5, 'callbacks')
    FIT_REPLACE_TABLE = [
        (FIT_X_POS_AND_KEY, lambda: None),
        (FIT_Y_POS_AND_KEY, lambda: None),
        (FIT_VDATA_POS_AND_KEY, lambda: ast.Constant(value=None)),
        (FIT_CALLBACKS_POS_AND_KEY, lambda: ast.Constant(value=None)),
    ]
    FIT_RET_DATA_TABLE = [FIT_X_POS_AND_KEY, FIT_Y_POS_AND_KEY]

    # keras.Model.fit_generator(
    #     generator,
    #     steps_per_epoch=None,
    #     epochs=1,
    #     verbose=1,
    #     callbacks=None,
    #     validation_data=None,
    #     validation_steps=None,
    #     class_weight=None,
    #     max_queue_size=10,
    #     workers=1,
    #     use_multiprocessing=False,
    #     shuffle=True,
    #     initial_epoch=0)
    FITG_GENERATOR_POS_AND_KEY = (0, 'generator')
    FITG_VDATA_POS_AND_KEY = (5, 'validation_data')
    FITG_CALLBACKS_POS_AND_KEY = (4, 'callbacks')
    FITG_REPLACE_TABLE = [
        (FITG_GENERATOR_POS_AND_KEY, lambda: None),
        (FITG_VDATA_POS_AND_KEY, lambda: ast.Constant(value=None)),
        (FITG_CALLBACKS_POS_AND_KEY, lambda: ast.Constant(value=None)),
    ]
    FITG_RET_DATA_TABLE = [FITG_GENERATOR_POS_AND_KEY]


    # keras.Model.train_on_batch(
    #     x,
    #     y=None,
    #     sample_weight=None,
    #     class_weight=None,
    #     reset_metrics=True,
    #     return_dict=False,
    # )
    TRAINOB_X_POS_AND_KEY = (0, 'x')
    TRAINOB_Y_POS_AND_KEY = (1, 'y')
    TRAINOB_REPLACE_TABLE = [
        (TRAINOB_X_POS_AND_KEY, lambda: None),
        (TRAINOB_Y_POS_AND_KEY, lambda: None),
    ]
    TRAINOB_RET_DATA_TABLE = [TRAINOB_X_POS_AND_KEY, TRAINOB_Y_POS_AND_KEY]

    def try_find_and_replace_impl(node, arg_pos_and_key, new_node):
        if arg := _find_arg_from_call(node, *arg_pos_and_key):
            if new_node:
                node = ReplaceXWithY(
                    x_cls_names=[arg.__class__.__name__],
                    is_x=lambda n: n == arg,
                    make_y=lambda n: new_node).visit(node)
                return node, {}
            else:
                ign = _get_ids(arg)
                return node, ign
        return node, {}
    
    def try_find_and_replace(node, r_table):
        ignore_ids = []
        for pos_and_key, mk in r_table:
            node, ign = try_find_and_replace_impl(node, pos_and_key, mk())
            ignore_ids.extend([*ign])
        return node, ignore_ids
    
    def make_use_and_ret_data(node, r_table):
        args = []
        for pos_and_key in r_table:
            if arg := _find_arg_from_call(node, *pos_and_key):
                args.append(_clone_ast(arg))
        ret_ids = []
        for a in args:
            ids = [ast.Name(id=i, ctx=ast.Load()) for i in _get_ids(a)]
            ret_ids.extend(ids)
        if len(ret_ids) == 1:
            return ast.Expr(value=ret_ids[0]), ast.Return(value=ret_ids[0]) 
        return ast.Expr(value=ast.Tuple(elts=ret_ids, ctx=ast.Load())), ast.Return(value=ast.Tuple(elts=ret_ids, ctx=ast.Load()))

    def make_fragment_maker(args: list, model_id):
        def fragment_maker(stmt_list: list):
            return ast.fix_missing_locations(
                    ast.FunctionDef(
                        name='train_model',
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[ast.arg(arg=e) for e in args],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[]),
                        body=[
                            *stmt_list,
                            ast.Return(value=model_id)],
                        decorator_list=[]))
        return fragment_maker

    # 1. Replace x, y, validation_data, callbacks in fit* with '_R2M_TRAIN_X', '_R2M_TRAIN_Y', None, None
    if _check_call_member_fn('fit', node):
        model_id = node.func.value
        node, ignore_ids = try_find_and_replace(node, FIT_REPLACE_TABLE)
        train_model_fn_args = ignore_ids
    elif _check_call_member_fn('fit_generator', node):
        model_id = node.func.value
        node, ignore_ids = try_find_and_replace(node, FITG_REPLACE_TABLE)
        train_model_fn_args = ignore_ids
    elif _check_call_member_fn('train_on_batch', node):
        model_id = node.func.value
        node, ignore_ids = try_find_and_replace(node, TRAINOB_REPLACE_TABLE)
        train_model_fn_args = ignore_ids
    else:
        assert False, 'xxx...'

    # 2. Return x, y
    def next_transformer(node):
        node = _clone_ast(node)
        if _check_call_member_fn('fit', node):
            use_expr, ret_stmt = make_use_and_ret_data(node, FIT_RET_DATA_TABLE)
        elif _check_call_member_fn('fit_generator', node):
            use_expr, ret_stmt = make_use_and_ret_data(node, FITG_RET_DATA_TABLE)
        elif _check_call_member_fn('train_on_batch', node):
            use_expr, ret_stmt = make_use_and_ret_data(node, TRAINOB_RET_DATA_TABLE)
        else:
            assert False, 'xxx...'

        def make_fragment_maker(ret_stmt):
            def fragment_maker(stmt_list: list):
                return ast.fix_missing_locations(
                        ast.FunctionDef(
                            name='get_data',
                            args=ast.arguments(
                                posonlyargs=[],
                                args=[],
                                kwonlyargs=[],
                                kw_defaults=[],
                                defaults=[]),
                            body=[*stmt_list, ret_stmt],
                            decorator_list=[]))
            return fragment_maker

        return False, use_expr, {}, make_fragment_maker(ret_stmt), lambda p: True, None

    return True, node, ignore_ids, make_fragment_maker(train_model_fn_args, model_id), _verify_keras_sfmodel, next_transformer


## -> should_include_cared_stmt, fixed_node, ignore_ids_fc, fragment_maker, fragment_checker, next_cared_ast_transformer
def _CARED_AST_TRANSFORMER_keras_program_1f_main(node: ast.Call):
    assert isinstance(node, ast.Call)
    node = _clone_ast(node)

    ##TODO: Consider different version
    FIT_REPLACE_TABLE = []
    if not int(os.getenv('R2MS_ASTT_KERAS_PROGRAM_1F_RETAIN_FIT_ALL_ARGS', '0')):
        FIT_VDATA_POS_AND_KEY = (7, 'validation_data')
        FIT_CALLBACKS_POS_AND_KEY = (5, 'callbacks')
        FIT_REPLACE_TABLE = [
            (FIT_VDATA_POS_AND_KEY, lambda: ast.Constant(value=None)),
            (FIT_CALLBACKS_POS_AND_KEY, lambda: ast.Constant(value=None)),
        ]

    FITG_REPLACE_TABLE = []
    if not int(os.getenv('R2MS_ASTT_KERAS_PROGRAM_1F_RETAIN_FITG_ALL_ARGS', '0')):
        FITG_VDATA_POS_AND_KEY = (5, 'validation_data')
        FITG_CALLBACKS_POS_AND_KEY = (4, 'callbacks')
        FITG_REPLACE_TABLE = [
            (FITG_VDATA_POS_AND_KEY, lambda: ast.Constant(value=None)),
            (FITG_CALLBACKS_POS_AND_KEY, lambda: ast.Constant(value=None)),
        ]


    TRAINOB_REPLACE_TABLE = [
    ]

    def try_find_and_replace_impl(node, arg_pos_and_key, new_node):
        if arg := _find_arg_from_call(node, *arg_pos_and_key):
            if new_node:
                node = ReplaceXWithY(
                    x_cls_names=[arg.__class__.__name__],
                    is_x=lambda n: n == arg,
                    make_y=lambda n: new_node).visit(node)
                return node, {}
            else:
                ign = _get_ids(arg)
                return node, ign
        return node, {}
    
    def try_find_and_replace(node, r_table):
        ignore_ids = []
        for pos_and_key, mk in r_table:
            node, ign = try_find_and_replace_impl(node, pos_and_key, mk())
            ignore_ids.extend([*ign])
        return node, ignore_ids
    
    def make_fragment_maker(args: list, model_id):
        def fragment_maker(stmt_list: list):
            return ast.fix_missing_locations(ast.Module(body=stmt_list, type_ignores=[]))
        return fragment_maker

    # 1. Replace validation_data, callbacks in fit* with '_R2M_TRAIN_X', '_R2M_TRAIN_Y', None, None
    if _check_call_member_fn('fit', node):
        model_id = node.func.value
        node, ignore_ids = try_find_and_replace(node, FIT_REPLACE_TABLE)
        train_model_fn_args = ignore_ids
    elif _check_call_member_fn('fit_generator', node):
        model_id = node.func.value
        node, ignore_ids = try_find_and_replace(node, FITG_REPLACE_TABLE)
        train_model_fn_args = ignore_ids
    elif _check_call_member_fn('train_on_batch', node):
        model_id = node.func.value
        node, ignore_ids = try_find_and_replace(node, TRAINOB_REPLACE_TABLE)
        train_model_fn_args = ignore_ids
    else:
        assert False, 'xxx...'

    return True, node, ignore_ids, make_fragment_maker(train_model_fn_args, model_id), _verify_keras_sfmodel, None


## -> should_include_cared_stmt, fixed_node, ignore_ids_fc, fragment_maker, fragment_checker, next_cared_ast_transformer
def _CARED_AST_TRANSFORMER_pt_program(node: ast.Call):
    assert isinstance(node, ast.Call)
    node = _clone_ast(node)

    #1. Make `train`
    def make_fragment_maker():
        def fragment_maker(stmt_list: list):
            return ast.fix_missing_locations(
                    ast.FunctionDef(
                        name='train',
                        args=ast.arguments(
                            posonlyargs=[],
                            args=[],
                            kwonlyargs=[],
                            kw_defaults=[],
                            defaults=[]),
                        body=stmt_list,
                        decorator_list=[]))
        return fragment_maker

    return True, node, [], make_fragment_maker(), _verify_pt_sfmodel, None


DEFAULT_CONFIGS = {
    'ENABLE_print_flatted_code': False,
    'ENABLE_print_clip_process': False,
    'ENABLE_print_changed_code': False,
    'ENABLE_print_inline_imports_progress': False,
    'ENABLE_rename_with_more_info': False,
    'ENABLE_rename_imported_id_to_fullname': True,
    'ENABLE_remove_imports_after_renaming_imported_id_to_fullname': True,
    'ENABLE_std_call': False,  # enable or not? it's a question
    'ENABLE_force_untab_ifNEqM_block': True,
    'ENABLE_inline_imports': True,
    'ENABLE_trans_comp_to_loop': True,
    'ENABLE_inline_calls': True,
    'ENABLE_remove_useless_stmts': True,
    'ENABLE_inline_vars': True,
    'ENABLE_inline_vars_l2': True,
    'ENABLE_unroll_for_loop': False,  # TODO: To be impl...
    'ENABLE_norm_local_ids': True,
    'ENABLE_std_api_call_kwargs': True and 0,  # enable or not? it's a question
    'ENABLE_rename_model_id': False,  # TODO: To be impl...
    'ENABLE_std_keras_compile_and_fit_kwargs': True,
    'ENABLE_std_keras_api_root': True,
    'ENABLE_std_keras_usage_style': True,
    'CONFIG_inline_imports_max_tries': 16,
    'CONFIG_inline_calls_max_tries': 16,
    'CONFIG_rename_imported_id_root_mark': '__root__',
    'CONFIG_import_roots': [],
    'CONFIG_cared_api_calls': ['keras_fit', 'keras_fit_generator', 'keras_train_on_batch'],
    'CONFIG_cared_ast_transformer': 'keras_program',
}


def get_v6_config():
    configs = {}
    configs['ENABLE_print_flatted_code'] = False
    configs['ENABLE_print_clip_process'] = False
    configs['ENABLE_print_changed_code'] = False
    configs['ENABLE_print_inline_imports_progress'] = False
    configs['ENABLE_rename_with_more_info'] = False
    configs['ENABLE_rename_imported_id_to_fullname'] = True
    configs['ENABLE_remove_imports_after_renaming_imported_id_to_fullname'] = True
    configs['ENABLE_std_call'] = False
    configs['ENABLE_force_untab_ifNEqM_block'] = True
    configs['ENABLE_inline_imports'] = True
    configs['ENABLE_trans_comp_to_loop'] = True
    configs['ENABLE_inline_calls'] = True
    configs['ENABLE_remove_useless_stmts'] = True
    configs['ENABLE_inline_vars'] = False
    configs['ENABLE_inline_vars_l2'] = False
    configs['ENABLE_unroll_for_loop'] = False
    configs['ENABLE_norm_local_ids'] = False
    configs['ENABLE_std_api_call_kwargs'] = False
    configs['ENABLE_rename_model_id'] = False
    configs['ENABLE_std_keras_compile_and_fit_kwargs'] = False
    configs['ENABLE_std_keras_api_root'] = False
    configs['ENABLE_std_keras_usage_style'] = False
    configs['CONFIG_inline_imports_max_tries'] = 16
    configs['CONFIG_inline_calls_max_tries'] = 16
    configs['CONFIG_rename_imported_id_root_mark'] = '__root__'
    configs['CONFIG_import_roots'] = []
    configs['CONFIG_cared_api_calls'] = ['keras_fit', 'keras_fit_generator', 'keras_train_on_batch']
    configs['CONFIG_cared_ast_transformer'] = 'keras_program'
    return configs


if __name__ == '__main__':
    class EvalAndStoreAction(argparse.Action):
        def __init__(self, option_strings, dest, nargs=None, **kwargs):
            if nargs is not None:
                raise ValueError("nargs not allowed")
            super().__init__(option_strings, dest, **kwargs)
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, eval(values, {}))

    for key, default in DEFAULT_CONFIGS.items():
        if (config_from_env := os.getenv(f'R2MS_{key}', None)):
            raise NotImplementedError(f'Config from env is not supported yet')

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--entry-file-path", type=str, required=True)
    parser.add_argument("-o", "--out-file-path", type=str, required=True)
    parser.add_argument("--no-output", action='store_true', default=False)
    parser.add_argument("--v6", action='store_true', default=False, help="Use v6 configs")
    parser.add_argument("--std-keras-usage", action='store_true', default=False)

    for key, default in DEFAULT_CONFIGS.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", type=int, default=None)
        elif isinstance(default, int):
            parser.add_argument(f"--{key}", type=int, default=None)
        elif isinstance(default, str):
            parser.add_argument(f"--{key}", type=str, default=None)
        else:
            parser.add_argument(f"--{key}", action=EvalAndStoreAction, default=None)

    args = parser.parse_args()

    if args.no_output:
        global print
        print = lambda *args, **kwargs: ...

    if args.v6:
        print('+>>> Use v6 configs\n')
        DEFAULT_CONFIGS = get_v6_config()

    if args.std_keras_usage:
        print('+>>> Use std keras usage configs\n')
        DEFAULT_CONFIGS['ENABLE_std_keras_usage_style'] = True
        DEFAULT_CONFIGS['ENABLE_std_keras_api_root'] = True
        DEFAULT_CONFIGS['ENABLE_std_keras_compile_and_fit_kwargs'] = True

    configs = {}
    for key, d in DEFAULT_CONFIGS.items():
        arg = getattr(args, key)
        configs[key] = type(d)(arg) if arg is not None else d

    print(f'+>>> CONFIG:')
    for k, v in configs.items():
        print(f'+        {k}={repr(v)}')

    ret_code = main(args.entry_file_path,
                    args.out_file_path,
                    configs)
    print(f'+>>> Done, exit with: {ret_code} ({"succeeded" if ret_code == 0 else "failed"})')
    sys.exit(ret_code)
