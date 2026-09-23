"""Validate a small code task with withheld cases and a restricted syntax."""
import ast
import json
import subprocess
import sys

VALIDATOR_VERSION = "invoice-restricted-v2-none-identity"

PROMPT = '''Implement invoice_total(rows) in Python. Each row is a dict with quantity, unit_price_cents, and optional discount_percent (default 0). Return the sum of each row's integer cents after discount. Round each discounted row down with integer division. Reject any negative quantity or price with ValueError. Reject discounts outside 0 through 100 with ValueError. An empty invoice returns 0. Use one function, assignments, a for loop, if statements, arithmetic, dict subscripts, dict.get, and raise ValueError. Do not use helper functions, comprehensions, generators, decorators, imports, or other calls. Return only the function code, with no Markdown fences.'''

ALLOWED = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign,
           ast.For, ast.If, ast.Raise, ast.Call, ast.Name, ast.Load, ast.Store, ast.Constant,
           ast.Subscript, ast.Attribute, ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp,
           ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.USub, ast.Lt, ast.LtE,
           ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Is, ast.IsNot, ast.Or, ast.And, ast.Not, ast.Expr, ast.Pass)
CASES = [([], 0), ([{"quantity": 3, "unit_price_cents": 101, "discount_percent": 17}], 251),
         ([{"quantity": 2, "unit_price_cents": 19}, {"quantity": 1, "unit_price_cents": 7, "discount_percent": 50}], 41),
         ([{"quantity": -1, "unit_price_cents": 10}], "ValueError"),
         ([{"quantity": 1, "unit_price_cents": -1}], "ValueError"),
         ([{"quantity": 1, "unit_price_cents": 4, "discount_percent": 101}], "ValueError"),
         ([{"quantity": 1, "unit_price_cents": 4, "discount_percent": -1}], "ValueError"),
         ([{"quantity": 1, "unit_price_cents": 999, "discount_percent": 100}], 0)]


def validate(source):
    try:
        tree = ast.parse(source)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "invoice_total":
            return {"passed": False, "reason": "function_contract"}
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED):
                return {"passed": None, "evaluated": False, "reason": "restricted_syntax"}
            if isinstance(node, ast.Compare) and any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
                operands = [node.left] + node.comparators
                if len(node.ops) != 1 or not any(isinstance(value, ast.Constant) and value.value is None for value in operands):
                    return {"passed": None, "evaluated": False, "reason": "restricted_identity"}
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                return {"passed": None, "evaluated": False, "reason": "private_name"}
            if isinstance(node, ast.Attribute) and node.attr != "get":
                return {"passed": None, "evaluated": False, "reason": "restricted_attribute"}
            if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in {"ValueError", "int"} or isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
                return {"passed": None, "evaluated": False, "reason": "restricted_call"}
        child = '''import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU,(1,1))
resource.setrlimit(resource.RLIMIT_AS,(128*1024*1024,128*1024*1024))
d=json.load(sys.stdin)
namespace={"__builtins__":{"ValueError":ValueError,"int":int}}
exec(compile(d["source"],"fixture","exec"),namespace)
results=[]
for rows, expected in d["cases"]:
 try: actual=namespace["invoice_total"](rows)
 except ValueError: actual="ValueError"
 results.append(actual==expected)
print(json.dumps({"passed":all(results),"evaluated":True,"cases":results}))
'''
        result = subprocess.run([sys.executable, "-I", "-c", child], input=json.dumps({"source": source, "cases": CASES}), text=True, capture_output=True, timeout=3)
        return json.loads(result.stdout) if result.returncode == 0 else {"passed": False, "reason": "execution_error"}
    except (SyntaxError, ValueError, subprocess.TimeoutExpired):
        return {"passed": False, "reason": "invalid_or_timeout"}
