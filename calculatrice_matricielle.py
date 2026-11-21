#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculatrice matricielle détaillée.

Fonctionnalités :
- Déterminant par développement de Laplace (cofacteurs) avec détail
- Déterminant par élimination de Gauss avec détail
- Inverse par cofacteurs
- Résolution de systèmes linéaires par la règle de Cramer
- Tests unitaires

Arithmétique : fractions exactes (fractions.Fraction).
"""

from __future__ import annotations

import re
import sys
import unittest
from fractions import Fraction
from typing import List, Optional

Matrix = List[List[Fraction]]


# ---------------------------------------------------------------------------
# Parsing et formatage
# ---------------------------------------------------------------------------

def parse_matrix(text: str) -> Matrix:
    """Formats acceptés :
    - '1 2 3; 4 5 6; 7 8 9'
    - '1 2 3\n4 5 6\n7 8 9'
    - '[[1,2,3],[4,5,6],[7,8,9]]'
    """
    s = text.strip()
    if s.startswith("[["):
        rows = re.findall(r"\[([^\[\]]*)\]", s)
        if not rows:
            raise ValueError("Format [[...],[...]] non reconnu")
        matrix = []
        for r in rows:
            vals = [x for x in re.split(r"[,\s]+", r.strip()) if x]
            matrix.append([Fraction(v) for v in vals])
    else:
        rows = re.split(r"[;\n]", s)
        matrix = []
        for r in rows:
            r = r.strip()
            if not r:
                continue
            vals = [x for x in re.split(r"[,\s]+", r) if x]
            matrix.append([Fraction(v) for v in vals])

    if not matrix:
        raise ValueError("Matrice vide")
    cols = len(matrix[0])
    for i, row in enumerate(matrix):
        if len(row) != cols:
            raise ValueError(f"Ligne {i+1} a {len(row)} valeurs, attendu {cols}")
    return matrix


def fmt(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def format_matrix(M: Matrix, prefix: str = "") -> str:
    if not M:
        return prefix + "[]"
    strs = [[fmt(v) for v in row] for row in M]
    w = max(len(s) for row in strs for s in row)
    return "\n".join(
        prefix + "[ " + "  ".join(s.rjust(w) for s in row) + " ]"
        for row in strs
    )


def minor(M: Matrix, i: int, j: int) -> Matrix:
    return [
        [M[r][c] for c in range(len(M[r])) if c != j]
        for r in range(len(M)) if r != i
    ]


# ---------------------------------------------------------------------------
# Déterminant — Laplace (avec détail)
# ---------------------------------------------------------------------------

def det_laplace(M: Matrix, trace: Optional[List[str]] = None, indent: int = 0) -> Fraction:
    n = len(M)
    if n == 0 or len(M[0]) != n:
        raise ValueError("Matrice carrée requise")

    pad = "  " * indent
    if trace is not None:
        trace.append(f"{pad}det {n}x{n} de :")
        trace.extend(format_matrix(M, prefix=pad + "  ").splitlines())

    if n == 1:
        v = M[0][0]
        if trace is not None:
            trace.append(f"{pad}  1x1 -> det = {fmt(v)}")
        return v

    if n == 2:
        a, b, c, d = M[0][0], M[0][1], M[1][0], M[1][1]
        v = a * d - b * c
        if trace is not None:
            trace.append(
                f"{pad}  2x2 -> ({fmt(a)})*({fmt(d)}) - ({fmt(b)})*({fmt(c)})"
                f" = {fmt(a*d)} - {fmt(b*c)} = {fmt(v)}"
            )
        return v

    if trace is not None:
        terms = " + ".join(f"({fmt(M[0][j])})*C1,{j+1}" for j in range(n))
        trace.append(f"{pad}  Développement selon ligne 1 : det = {terms}")

    total = Fraction(0)
    for j in range(n):
        a = M[0][j]
        sub = minor(M, 0, j)
        if trace is not None:
            trace.append(f"{pad}  -- Cofacteur C1,{j+1} :")
            trace.extend(format_matrix(sub, prefix=pad + "     mineur ").splitlines())
        sub_det = det_laplace(sub, trace, indent + 2)
        cofactor = sub_det if j % 2 == 0 else -sub_det
        contrib = a * cofactor
        if trace is not None:
            sign = "+" if j % 2 == 0 else "-"
            trace.append(
                f"{pad}     C1,{j+1} = {sign}({fmt(sub_det)}) = {fmt(cofactor)}"
                f" ; contribution = {fmt(contrib)}"
            )
        total += contrib

    if trace is not None:
        trace.append(f"{pad}  => det = {fmt(total)}")
    return total


# ---------------------------------------------------------------------------
# Déterminant — Gauss (avec détail)
# ---------------------------------------------------------------------------

def det_gauss(M: Matrix, trace: Optional[List[str]] = None) -> Fraction:
    n = len(M)
    if n == 0 or len(M[0]) != n:
        raise ValueError("Matrice carrée requise")

    A = [row[:] for row in M]
    det = Fraction(1)
    swaps = 0

    if trace is not None:
        trace.append(f"Élimination de Gauss ({n}x{n}) :")
        trace.extend(format_matrix(A, prefix="  ").splitlines())

    for k in range(n):
        pivot_row = k
        while pivot_row < n and A[pivot_row][k] == 0:
            pivot_row += 1
        if pivot_row == n:
            if trace is not None:
                trace.append(f"  Colonne {k+1} sans pivot -> det = 0")
            return Fraction(0)

        if pivot_row != k:
            A[k], A[pivot_row] = A[pivot_row], A[k]
            swaps += 1
            if trace is not None:
                trace.append(f"  Échange L{k+1} <-> L{pivot_row+1}")

        pivot = A[k][k]
        if trace is not None:
            trace.append(f"  Pivot ({k+1},{k+1}) = {fmt(pivot)}")

        for i in range(k + 1, n):
            if A[i][k] == 0:
                continue
            factor = A[i][k] / pivot
            if trace is not None:
                trace.append(f"    L{i+1} <- L{i+1} - ({fmt(factor)})*L{k+1}")
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]

    for k in range(n):
        det *= A[k][k]
    if swaps % 2 == 1:
        det = -det

    if trace is not None:
        trace.append("  Matrice triangulaire :")
        trace.extend(format_matrix(A, prefix="  ").splitlines())
        diag = " * ".join(fmt(A[k][k]) for k in range(n))
        sign = " * (-1)" if swaps % 2 == 1 else ""
        trace.append(f"  det = {diag}{sign} = {fmt(det)}")

    return det


# ---------------------------------------------------------------------------
# Inverse par cofacteurs
# ---------------------------------------------------------------------------

def inverse_cofactors(M: Matrix, trace: Optional[List[str]] = None) -> Optional[Matrix]:
    n = len(M)
    if n == 0 or len(M[0]) != n:
        raise ValueError("Matrice carrée requise")

    det = det_gauss(M)
    if det == 0:
        if trace is not None:
            trace.append("det = 0 -> non inversible.")
        return None

    if trace is not None:
        trace.append(f"det = {fmt(det)} != 0 -> inversible.")
        trace.append("Cofacteurs C(i,j) :")

    cof = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            sign = 1 if (i + j) % 2 == 0 else -1
            cof[i][j] = sign * det_gauss(minor(M, i, j))
            if trace is not None:
                trace.append(
                    f"  C({i+1},{j+1}) = {'+' if sign == 1 else '-'}"
                    f" * det(M{i+1},{j+1}) = {fmt(cof[i][j])}"
                )

    adj = [[cof[j][i] for j in range(n)] for i in range(n)]
    inv = [[adj[i][j] / det for j in range(n)] for i in range(n)]

    if trace is not None:
        trace.append("Comatrice transposée :")
        trace.extend(format_matrix(adj, prefix="  ").splitlines())
        trace.append(f"Division par det = {fmt(det)} :")
        trace.extend(format_matrix(inv, prefix="  ").splitlines())

    return inv


# ---------------------------------------------------------------------------
# Cramer
# ---------------------------------------------------------------------------

def solve_cramer(A: Matrix, b: List[Fraction], trace: Optional[List[str]] = None) -> Optional[List[Fraction]]:
    n = len(A)
    if len(b) != n or any(len(r) != n for r in A):
        raise ValueError("Dimensions incompatibles")

    detA = det_gauss(A)
    if trace is not None:
        trace.append(f"det(A) = {fmt(detA)}")
    if detA == 0:
        return None

    x: List[Fraction] = []
    for k in range(n):
        Ak = [row[:] for row in A]
        for i in range(n):
            Ak[i][k] = b[i]
        detAk = det_gauss(Ak)
        xi = detAk / detA
        x.append(xi)
        if trace is not None:
            trace.append(f"det(A{k+1}) = {fmt(detAk)} -> x{k+1} = {fmt(xi)}")
    return x


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

def run_det(M: Matrix) -> None:
    print("=== Laplace ===")
    t: List[str] = []
    d1 = det_laplace(M, t)
    for l in t:
        print(l)
    print("\n=== Gauss ===")
    t = []
    d2 = det_gauss(M, t)
    for l in t:
        print(l)
    print(f"\nDéterminant : {fmt(d1)} (Laplace) | {fmt(d2)} (Gauss)")


def repl() -> None:
    print("Calculatrice matricielle détaillée")
    print("  det <matrice>          déterminant (Laplace + Gauss)")
    print("  inv <matrice>          inverse par cofacteurs")
    print("  solve <A> | <b>        résout Ax=b par Cramer")
    print("  quit")
    print("Format : '1 2 3; 4 5 6; 7 8 9' ou '[[1,2,3],[4,5,6],[7,8,9]]'")
    print()

    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", ":q"):
            break
        try:
            if line.lower().startswith("det "):
                run_det(parse_matrix(line[4:].strip()))
            elif line.lower().startswith("inv "):
                t: List[str] = []
                inv = inverse_cofactors(parse_matrix(line[4:].strip()), t)
                for l in t:
                    print(l)
                if inv is not None:
                    print("A^-1 =")
                    print(format_matrix(inv))
            elif line.lower().startswith("solve "):
                parts = line[6:].split("|")
                if len(parts) != 2:
                    raise ValueError("Format : solve <A> | <b>")
                A = parse_matrix(parts[0].strip())
                b = [Fraction(x) for x in re.split(r"[,\s]+", parts[1].strip()) if x]
                t = []
                x = solve_cramer(A, b, t)
                for l in t:
                    print(l)
                if x is not None:
                    print("Solution : " + ", ".join(f"x{i+1}={fmt(v)}" for i, v in enumerate(x)))
            else:
                print("Commande inconnue.")
        except Exception as exc:
            print(f"Erreur: {exc}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMatrix(unittest.TestCase):
    def test_parse(self) -> None:
        M = parse_matrix("1 2; 3 4")
        self.assertEqual(M, [[1, 2], [3, 4]])

    def test_det_2x2(self) -> None:
        M = parse_matrix("1 2; 3 4")
        self.assertEqual(det_laplace(M), -2)
        self.assertEqual(det_gauss(M), -2)

    def test_det_3x3_singular(self) -> None:
        M = parse_matrix("1 2 3; 4 5 6; 7 8 9")
        self.assertEqual(det_laplace(M), 0)
        self.assertEqual(det_gauss(M), 0)

    def test_det_3x3_regulier(self) -> None:
        M = parse_matrix("3 1 0; 1 2 1; 0 1 1")
        d = det_laplace(M)
        self.assertEqual(d, det_gauss(M))
        self.assertEqual(d, 2)

    def test_inverse(self) -> None:
        M = parse_matrix("4 7; 2 6")
        inv = inverse_cofactors(M)
        self.assertIsNotNone(inv)
        I = [[sum(M[i][k] * inv[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        for i in range(2):
            for j in range(2):
                self.assertEqual(I[i][j], 1 if i == j else 0)

    def test_cramer(self) -> None:
        A = parse_matrix("2 1; 1 3")
        b = [Fraction(5), Fraction(10)]
        x = solve_cramer(A, b)
        self.assertEqual(x, [Fraction(1), Fraction(3)])

    def test_cramer_singular(self) -> None:
        A = parse_matrix("1 2; 2 4")
        b = [Fraction(1), Fraction(2)]
        self.assertIsNone(solve_cramer(A, b))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        unittest.main(argv=[sys.argv[0]])
    else:
        repl()
