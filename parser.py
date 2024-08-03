from copy import deepcopy
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple


class ArithmeticSyntaxError(Exception):
    """Raise this error if an operation does not abide by legal arithmetic syntax."""

    def __init__(self, msg: str) -> None:
        """Pass error message."""
        super().__init__(msg)


def compute_arithmetic(operation: str) -> float:
    """Compute the value of the given operation.

    For example, compute_arithmetic("15*(-10)") returns -150

    Returns
    -------
    float
        The computed value.

    """
    return build_ast(operation).compute()


class Expression:
    """Represents a token or an expression obtained from lexing or parsing an arithmetic operation."""

    def __init__(
        self,
        value: str,
        left: Optional["Expression"] = None,
        right: Optional["Expression"] = None,
        is_num: bool = False,
        is_resolved: bool = True,
        depth: int = 0,
    ) -> None:
        """Initialize an expression. An unresolved expression is considered a token."""
        self.value = value
        self.left = left
        self.right = right
        self.is_num = is_num
        self.is_resolved = is_resolved
        self.depth = depth

    def __eq__(self, other: Any) -> bool:
        """Two Expressions are equal if their values are equal down to the sub tree."""
        if type(other).__name__ == "Expression":
            return (
                self.value == other.value
                and self.left == other.left
                and self.right == other.right
            )
        return False

    def is_add(self) -> bool:
        """Check whether token or expression is an addition.

        Returns
        -------
        bool
            True if the token or expression is an addition.
        """
        return self.value == "+"

    def is_sub(self) -> bool:
        """Check whether token or expression is a subtraction.

        Returns
        -------
        bool
            True if the token or expression is a subtraction.
        """
        return self.value == "-"

    def is_mul(self) -> bool:
        """Check whether token or expression is a multiplication.

        Returns
        -------
        bool
            True if the token or expression is a multiplication.
        """
        return self.value == "*"

    def is_div(self) -> bool:
        """Check whether token or expression is a division.

        Returns
        -------
        bool
            True if the token or expression is a division.
        """
        return self.value == "/"

    def is_open(self) -> bool:
        """Check whether token or expression is an open paranthesis.

        Returns
        -------
        bool
            True if the token or expression is an open paranthesis.
        """
        return self.value == "("

    def is_close(self) -> bool:
        """Check whether token or expression is a closed paranthesis.

        Returns
        -------
        bool
            True if the token or expression is a closed paranthesis.
        """
        return self.value == ")"

    def is_unresolved_operator(self) -> bool:
        """Check whether the expression is an unresolved operator.

        Returns
        -------
        bool
            True if the token or expression is a unresolved operator.
        """
        return not self.is_resolved and (
            self.is_add() or self.is_sub() or self.is_mul() or self.is_div()
        )

    def compute(self) -> float:
        """Recursively compute the result of the arithmetic operation.

        Returns
        -------
        float
            The result.
        """
        if not self.left and not self.right:
            if self.is_num:
                return float(self.value)

        if self.left and self.right:
            if self.is_add():
                return self.left.compute() + self.right.compute()
            if self.is_sub():
                return self.left.compute() - self.right.compute()
            if self.is_mul():
                return self.left.compute() * self.right.compute()
            if self.is_div():
                try:
                    return self.left.compute() / self.right.compute()
                except ZeroDivisionError:
                    return float("nan")

        raise ArithmeticSyntaxError("tried to compute value of invalid ast")


def lex(op: str) -> Tuple[List[Expression], int]:
    """Lex the arithmetic operation.

    Returns
    -------
    Tuple[List[Expression], int]
        List of tokens as well as max paranthesis depth.
    """
    tokens = []
    num = ""
    depth = 0
    max_depth = 0

    for c in op:
        if c in "0123456789":
            num += c
            continue

        if c == ".":
            if num.count(".") == 0:
                num += c
                continue
            raise ArithmeticSyntaxError('float numbers may only contain one "."')

        if num != "":
            tokens.append(Expression(num, is_resolved=False, is_num=True, depth=depth))
            num = ""

        if c == "(":
            depth += 1
            tokens.append(Expression(c, is_resolved=False, depth=depth))
            if depth > max_depth:
                max_depth = depth
            continue

        if c == ")":
            tokens.append(Expression(c, is_resolved=False, depth=depth))
            depth -= 1
            if depth < 0:
                raise ArithmeticSyntaxError(
                    "uneven amount of open and closed parantheses"
                )
            continue

        if c in "+-*/":
            tokens.append(Expression(c, is_resolved=False, depth=depth))
            continue

        raise ArithmeticSyntaxError("illegal character '{s}'".format(s=c))

    if num != "":
        tokens.append(Expression(num, is_resolved=False, is_num=True, depth=depth))

    return tokens, max_depth


def resolve_nums(tokens: List[Expression]) -> List[Expression]:
    """Resolve numbers."""
    resolved = []

    for i in range(len(tokens)):
        exp = match_num(tokens, i)
        if exp:
            resolved += [exp]
            continue

        resolved += [tokens[i]]

    return resolved


def resolve_paras(
    exps: List[Expression], depth: int = 0
) -> Tuple[List[Expression], bool]:
    """Resolve paranthesis."""
    resolved = []
    change_detected = False

    for i in range(len(exps)):
        exp = match_paras(exps, i, depth)
        if exp:
            resolved += [exp] + exps[i + 3 :]
            change_detected = True
            break
        resolved += [exps[i]]

    return resolved, change_detected


def resolve_mul_div(
    exps: List[Expression], depth: int = 0
) -> Tuple[List[Expression], bool]:
    """Resolve multiplication and division."""
    resolved = []
    change_detected = False

    for i in range(len(exps)):
        exp = match_mul_div(exps, i, depth)
        if exp:
            resolved += [exp] + exps[i + 3 :]
            change_detected = True
            break
        resolved += [exps[i]]

    return resolved, change_detected


def resolve_neg(
    exps: List[Expression], depth: int = 0
) -> Tuple[List[Expression], bool]:
    """Resolve negation."""
    resolved = []
    change_detected = False

    for i in range(len(exps)):
        exp = match_neg(exps, i, depth)
        if exp:
            resolved += [exp] + exps[i + 2 :]
            change_detected = True
            break
        resolved += [exps[i]]

    return resolved, change_detected


def resolve_add_sub(
    exps: List[Expression], depth: int = 0
) -> Tuple[List[Expression], bool]:
    """Resolve addition."""
    resolved = []
    change_detected = False

    for i in range(len(exps)):
        exp = match_add_sub(exps, i, depth)
        if exp:
            resolved += [exp] + exps[i + 3 :]
            change_detected = True
            break
        resolved += [exps[i]]

    return resolved, change_detected


def build_ast(op: str) -> Expression:
    """Build the abstract syntax tree.

    The following CFG describes the desired arithmetic sorted by order of precedence:

    S -> num

    S -> (S)

    S -> -S

    S -> S*S | S/S

    S -> S+S | S-S

    The AST is built by iteratively matching sub-expressions with the CFG rules until a single expression
    emerges (the deepest parantheses are evaluated first). Failing to finish with a single expression
    indicates a syntax error in the input.

    Returns
    -------
    Expression
        The AST.
    """
    tokens, depth = lex(op)
    resolved = resolve_nums(tokens)

    while True:
        exps = deepcopy(resolved)

        resolved, change_detected = resolve_paras(exps, depth)
        if change_detected:
            continue

        resolved, change_detected = resolve_neg(exps, depth)
        if change_detected:
            continue

        resolved, change_detected = resolve_mul_div(exps, depth)
        if change_detected:
            continue

        resolved, change_detected = resolve_add_sub(exps, depth)
        if change_detected:
            continue

        if depth > 0:
            for i in range(len(resolved)):
                if resolved[i].depth == depth:
                    resolved[i].depth -= 1
            depth -= 1
            continue

        if len(resolved) == 1 and resolved[0].is_resolved:
            return resolved[0]

        raise ArithmeticSyntaxError(
            "tried to build ast of invalid arithmetic expression"
        )


def match_num(exps: List[Expression], i: int) -> Optional[Expression]:
    """Try to match the current sub-expressions with num.

    Returns
    -------
    Optional[Expression]
        The parsed sub-expression if matching was possible, else None
    """
    exp = exps[i]
    if exp.is_num:
        exp.is_resolved = True
        return exp

    return None


def match_paras(exps: List[Expression], i: int, depth: int = 0) -> Optional[Expression]:
    """Try to match the three current sub-expressions with (S).

    Returns
    -------
    Optional[Expression]
        The parsed sub-expression if matching was possible, else None
    """
    if i + 2 >= len(exps):
        return None

    left = exps[i]
    mid = exps[i + 1]
    right = exps[i + 2]

    if not (left.depth == mid.depth == right.depth == depth):
        return None

    if left.is_open() and mid.is_resolved and right.is_close():
        return mid

    return None


def match_neg(exps: List[Expression], i: int, depth: int = 0) -> Optional[Expression]:
    """Try to match the two current sub-expressions with -S.

    Returns
    -------
    Optional[Expression]
        The parsed sub-expression if matching was possible, else None
    """
    if i + 1 >= len(exps):
        return None

    mid = exps[i]
    right = exps[i + 1]
    zero = Expression("0", is_resolved=True, is_num=True)

    if not (mid.depth == right.depth == depth):
        return None

    if i == 0 and mid.is_sub() and right.is_resolved:
        return Expression("-", left=zero, right=right, depth=depth)

    left = exps[i - 1]
    if not left.depth == depth:
        return None

    if (
        (left.is_open() or left.is_unresolved_operator())
        and mid.is_sub()
        and right.is_resolved
    ):
        return Expression("-", left=zero, right=right, depth=depth)

    return None


def match_mul_div(
    exps: List[Expression], i: int, depth: int = 0
) -> Optional[Expression]:
    """Try to match the three current sub-expressions with S*S or S/S.

    Returns
    -------
    Optional[Expression]
        The parsed sub-expression if matching was possible, else None
    """
    if i + 2 >= len(exps):
        return None

    left = exps[i]
    mid = exps[i + 1]
    right = exps[i + 2]

    if not (left.depth == mid.depth == right.depth == depth):
        return None

    if left.is_resolved and (mid.is_mul() or mid.is_div()) and right.is_resolved:
        return Expression(mid.value, left=left, right=right, depth=depth)

    return None


def match_add_sub(
    exps: List[Expression], i: int, depth: int = 0
) -> Optional[Expression]:
    """Try to match the three current sub-expressions with S+S or S-S.

    Returns
    -------
    Optional[Expression]
        The parsed sub-expression if matching was possible, else None
    """
    if i + 2 >= len(exps):
        return None

    left = exps[i]
    mid = exps[i + 1]
    right = exps[i + 2]

    if not (left.depth == mid.depth == right.depth == depth):
        return None

    if left.is_resolved and (mid.is_add() or mid.is_sub()) and right.is_resolved:
        return Expression(mid.value, left=left, right=right, depth=depth)

    return None

