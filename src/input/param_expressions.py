import ast
import operator


_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_NAMES = {
    "pi": 3.141592653589793,
    "tau": 6.283185307179586,
    "e": 2.718281828459045,
}

_ALLOWED_MODULE_ATTRS = {
    "np": _ALLOWED_NAMES,
    "numpy": _ALLOWED_NAMES,
    "math": _ALLOWED_NAMES,
}


class _ExpressionResolutionError(ValueError):
    pass


def _eval_expr_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise _ExpressionResolutionError(f"Unsupported literal {node.value!r}")

    if isinstance(node, ast.Num):  # pragma: no cover - legacy Python AST node
        return node.n

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise _ExpressionResolutionError(f"Unsupported operator {op_type.__name__}")
        return _BINARY_OPS[op_type](_eval_expr_node(node.left), _eval_expr_node(node.right))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise _ExpressionResolutionError(f"Unsupported operator {op_type.__name__}")
        return _UNARY_OPS[op_type](_eval_expr_node(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise _ExpressionResolutionError(f"Unknown name {node.id!r}")

    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        module_name = node.value.id
        attrs = _ALLOWED_MODULE_ATTRS.get(module_name)
        if attrs is None or node.attr not in attrs:
            raise _ExpressionResolutionError(f"Unknown attribute {module_name}.{node.attr}")
        return attrs[node.attr]

    raise _ExpressionResolutionError(f"Unsupported expression node {type(node).__name__}")


def try_resolve_expression(value):
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    try:
        node = ast.parse(text, mode="eval")
        return _eval_expr_node(node.body)
    except (SyntaxError, _ExpressionResolutionError, ZeroDivisionError, OverflowError):
        return value


def resolve_parameter_expressions(value):
    if isinstance(value, dict):
        return {key: resolve_parameter_expressions(item) for key, item in value.items()}

    if isinstance(value, list):
        return [resolve_parameter_expressions(item) for item in value]

    return try_resolve_expression(value)
