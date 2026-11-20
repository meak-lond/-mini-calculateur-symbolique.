#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-calculateur symbolique autonome.

Fonctionnalités :
- analyse lexicale et syntaxique d'expressions mathématiques ;
- AST (arbre de syntaxe abstraite) ;
- évaluation numérique ;
- simplification algébrique de base ;
- dérivation symbolique ;
- REPL interactive ;
- tests unitaires.

Syntaxe supportée :
  nombres, variables, + - * / ^, parenthèses,
  fonctions : sin, cos, tan, exp, log, sqrt, abs.

Exemples :
  :eval 2 + 3 * 4
  :simplify x + 0 + 2 * 3
  :diff x^3 + 2*x x
"""

from __future__ import annotations

import math
import re
import sys
import unittest
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class UnaryOp:
    op: str
    expr: "Expr"


@dataclass(frozen=True)
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Func:
    name: str
    arg: "Expr"


Expr = Union[Num, Var, UnaryOp, BinOp, Func]


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    kind: str
    value: object


TOKEN_RE = re.compile(
    r"""
    (?P<SPACE>\s+)
  | (?P<NUMBER>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<IDENT>[A-Za-z_]\w*)
  | (?P<OP>[+\-*/^(),])
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0

    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise SyntaxError(
                f"Caractère invalide à la position {pos}: {text[pos]!r}"
            )

        kind = m.lastgroup
        raw = m.group()
        pos = m.end()

        if kind == "SPACE":
            continue

        if kind == "NUMBER":
            tokens.append(Token("NUMBER", float(raw)))
        elif kind == "IDENT":
            tokens.append(Token("IDENT", raw))
        elif kind == "OP":
            tokens.append(Token("OP", raw))
        else:
            raise AssertionError(f"Type de token inattendu: {kind}")

    tokens.append(Token("EOF", None))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        tok = self.peek()
        self.pos += 1
        return tok

    def match_op(self, op: str) -> bool:
        tok = self.peek()
        if tok.kind == "OP" and tok.value == op:
            self.pos += 1
            return True
        return False

    def expect_op(self, op: str) -> None:
        if not self.match_op(op):
            raise SyntaxError(f"Attendu {op!r}, obtenu {self.peek()!r}")

    def parse(self) -> Expr:
        expr = self.parse_expression()
        if self.peek().kind != "EOF":
            raise SyntaxError(f"Token inattendu: {self.peek()!r}")
        return expr

    def parse_expression(self) -> Expr:
        left = self.parse_term()

        while True:
            if self.match_op("+"):
                left = BinOp("+", left, self.parse_term())
            elif self.match_op("-"):
                left = BinOp("-", left, self.parse_term())
            else:
                break

        return left

    def parse_term(self) -> Expr:
        left = self.parse_unary()

        while True:
            if self.match_op("*"):
                left = BinOp("*", left, self.parse_unary())
            elif self.match_op("/"):
                left = BinOp("/", left, self.parse_unary())
            else:
                break

        return left

    def parse_unary(self) -> Expr:
        if self.match_op("+"):
            return self.parse_unary()

        if self.match_op("-"):
            return UnaryOp("-", self.parse_unary())

        return self.parse_power()

    def parse_power(self) -> Expr:
        left = self.parse_primary()

        if self.match_op("^"):
            # ^ est associatif à droite : 2^3^2 = 2^(3^2)
            right = self.parse_unary()
            return BinOp("^", left, right)

        return left

    def parse_primary(self) -> Expr:
        tok = self.peek()

        if tok.kind == "NUMBER":
            self.advance()
            return Num(float(tok.value))

        if tok.kind == "IDENT":
            self.advance()
            name = str(tok.value)

            if self.match_op("("):
                arg = self.parse_expression()
                self.expect_op(")")
                return Func(name, arg)

            return Var(name)

        if self.match_op("("):
            expr = self.parse_expression()
            self.expect_op(")")
            return expr

        raise SyntaxError(f"Expression attendue, obtenu {tok!r}")


def parse_expression(text: str) -> Expr:
    return Parser(tokenize(text)).parse()


# ---------------------------------------------------------------------------
# Évaluateur
# ---------------------------------------------------------------------------

FUNCTIONS: Dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
}


def evaluate(node: Expr, env: Optional[Dict[str, float]] = None) -> float:
    if env is None:
        env = {}

    if isinstance(node, Num):
        return node.value

    if isinstance(node, Var):
        if node.name not in env:
            raise NameError(f"Variable non définie: {node.name}")
        return env[node.name]

    if isinstance(node, UnaryOp):
        if node.op == "-":
            return -evaluate(node.expr, env)
        raise ValueError(f"Opérateur unaire inconnu: {node.op}")

    if isinstance(node, BinOp):
        a = evaluate(node.left, env)
        b = evaluate(node.right, env)

        if node.op == "+":
            return a + b
        if node.op == "-":
            return a - b
        if node.op == "*":
            return a * b
        if node.op == "/":
            return a / b
        if node.op == "^":
            return a ** b

        raise ValueError(f"Opérateur binaire inconnu: {node.op}")

    if isinstance(node, Func):
        if node.name not in FUNCTIONS:
            raise NameError(f"Fonction inconnue: {node.name}")
        return FUNCTIONS[node.name](evaluate(node.arg, env))

    raise TypeError(f"Nœud AST inconnu: {node!r}")


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def to_string(node: Expr) -> str:
    if isinstance(node, Num):
        v = node.value
        if v.is_integer():
            return str(int(v))
        return repr(v)

    if isinstance(node, Var):
        return node.name

    if isinstance(node, UnaryOp):
        return f"-{to_string(node.expr)}"

    if isinstance(node, BinOp):
        return f"({to_string(node.left)} {node.op} {to_string(node.right)})"

    if isinstance(node, Func):
        return f"{node.name}({to_string(node.arg)})"

    raise TypeError(f"Nœud AST inconnu: {node!r}")


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------

def _is_num(node: Expr, value: float) -> bool:
    return isinstance(node, Num) and abs(node.value - value) < 1e-12


def simplify(node: Expr) -> Expr:
    if isinstance(node, (Num, Var)):
        return node

    if isinstance(node, UnaryOp):
        inner = simplify(node.expr)

        if isinstance(inner, Num):
            return Num(-inner.value)

        if isinstance(inner, UnaryOp) and inner.op == "-":
            return inner.expr

        return UnaryOp("-", inner)

    if isinstance(node, Func):
        arg = simplify(node.arg)

        if isinstance(arg, Num):
            try:
                return Num(evaluate(Func(node.name, arg)))
            except (ValueError, ZeroDivisionError, OverflowError):
                pass

        return Func(node.name, arg)

    if isinstance(node, BinOp):
        left = simplify(node.left)
        right = simplify(node.right)
        op = node.op

        # Évaluation constante
        if isinstance(left, Num) and isinstance(right, Num):
            try:
                return Num(evaluate(BinOp(op, left, right)))
            except (ValueError, ZeroDivisionError, OverflowError):
                pass

        if op == "+":
            if _is_num(right, 0):
                return left
            if _is_num(left, 0):
                return right

        elif op == "-":
            if _is_num(right, 0):
                return left
            if _is_num(left, 0):
                return UnaryOp("-", right)
            if left == right:
                return Num(0)

        elif op == "*":
            if _is_num(left, 0) or _is_num(right, 0):
                return Num(0)
            if _is_num(left, 1):
                return right
            if _is_num(right, 1):
                return left
            if _is_num(left, -1):
                return UnaryOp("-", right)
            if _is_num(right, -1):
                return UnaryOp("-", left)

        elif op == "/":
            if _is_num(left, 0):
                return Num(0)
            if _is_num(right, 1):
                return left
            if left == right:
                return Num(1)

        elif op == "^":
            if _is_num(right, 1):
                return left
            if _is_num(right, 0):
                return Num(1)
            if _is_num(left, 1):
                return Num(1)
            if _is_num(left, 0) and isinstance(right, Num) and right.value > 0:
                return Num(0)

        return BinOp(op, left, right)

    raise TypeError(f"Nœud AST inconnu: {node!r}")


# ---------------------------------------------------------------------------
# Dérivation symbolique
# ---------------------------------------------------------------------------

def diff(node: Expr, var: str) -> Expr:
    if isinstance(node, Num):
        return Num(0)

    if isinstance(node, Var):
        return Num(1 if node.name == var else 0)

    if isinstance(node, UnaryOp):
        return UnaryOp("-", diff(node.expr, var))

    if isinstance(node, Func):
        u = node.arg
        du = diff(u, var)
        name = node.name

        if name == "sin":
            return BinOp("*", Func("cos", u), du)

        if name == "cos":
            return UnaryOp("-", BinOp("*", Func("sin", u), du))

        if name == "tan":
            return BinOp("/", du, BinOp("^", Func("cos", u), Num(2)))

        if name == "exp":
            return BinOp("*", Func("exp", u), du)

        if name == "log":
            return BinOp("/", du, u)

        if name == "sqrt":
            return BinOp("/", du, BinOp("*", Num(2), Func("sqrt", u)))

        if name == "abs":
            # d|u|/du = u / |u| * du, sauf en 0 où la dérivée n'existe pas.
            return BinOp("*", BinOp("/", u, Func("abs", u)), du)

        raise ValueError(f"Fonction non dérivable: {name}")

    if isinstance(node, BinOp):
        u = node.left
        v = node.right
        du = diff(u, var)
        dv = diff(v, var)
        op = node.op

        if op == "+":
            return BinOp("+", du, dv)

        if op == "-":
            return BinOp("-", du, dv)

        if op == "*":
            return BinOp(
                "+",
                BinOp("*", du, v),
                BinOp("*", u, dv),
            )

        if op == "/":
            return BinOp(
                "/",
                BinOp("-", BinOp("*", du, v), BinOp("*", u, dv)),
                BinOp("^", v, Num(2)),
            )

        if op == "^":
            if isinstance(v, Num):
                # d(u^c)/dx = c * u^(c-1) * u'
                return BinOp(
                    "*",
                    BinOp("*", v, BinOp("^", u, Num(v.value - 1))),
                    du,
                )

            # Forme générale : d(u^v) = u^v * (v' ln(u) + v * u'/u)
            return BinOp(
                "*",
                BinOp("^", u, v),
                BinOp(
                    "+",
                    BinOp("*", dv, Func("log", u)),
                    BinOp("*", v, BinOp("/", du, u)),
                ),
            )

        raise ValueError(f"Opérateur non dérivable: {op}")

    raise TypeError(f"Nœud AST inconnu: {node!r}")


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def repl() -> None:
    print("Mini-calculateur symbolique")
    print("Commandes :")
    print("  :eval EXPR        évalue numériquement")
    print("  :simplify EXPR    simplifie")
    print("  :diff EXPR VAR    dérive par rapport à VAR")
    print("  :quit             quitter")
    print()

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line in (":quit", ":q", "quit", "exit"):
            break

        try:
            if line.startswith(":eval "):
                expr = parse_expression(line[6:].strip())
                print(evaluate(expr))

            elif line.startswith(":simplify "):
                expr = parse_expression(line[10:].strip())
                print(to_string(simplify(expr)))

            elif line.startswith(":diff "):
                rest = line[6:].strip()
                parts = rest.rsplit(maxsplit=1)

                if len(parts) != 2:
                    raise SyntaxError("Usage : :diff EXPR VAR")

                expr = parse_expression(parts[0])
                var = parts[1]
                print(to_string(simplify(diff(expr, var))))

            else:
                expr = parse_expression(line)
                print(to_string(simplify(expr)))

        except Exception as exc:
            print(f"Erreur: {exc}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSymbolicCalculator(unittest.TestCase):
    def test_parse_and_eval(self) -> None:
        cases = {
            "2+3*4": 14,
            "2^3^2": 512,
            "-2^2": -4,
            "2*-3": -6,
            "sin(0)+cos(0)": 1,
        }

        for src, expected in cases.items():
            with self.subTest(src=src):
                self.assertAlmostEqual(evaluate(parse_expression(src)), expected)

    def test_simplify(self) -> None:
        expr = parse_expression("x + 0 + 2*3")
        self.assertEqual(to_string(simplify(expr)), "(x + 6)")

    def test_diff_polynomial(self) -> None:
        expr = parse_expression("x^3 + 2*x")
        derivative = simplify(diff(expr, "x"))
        self.assertAlmostEqual(evaluate(derivative, {"x": 2}), 14)

    def test_diff_sin(self) -> None:
        expr = parse_expression("sin(x)")
        derivative = simplify(diff(expr, "x"))
        self.assertAlmostEqual(evaluate(derivative, {"x": 0}), 1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        unittest.main(argv=[sys.argv[0]])
    else:
        repl()
