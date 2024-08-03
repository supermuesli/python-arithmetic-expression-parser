import pytest

from parser import ArithmeticSyntaxError
from parser import Expression
from parser import build_ast
from parser import lex
from parser import match_add_sub
from parser import match_mul_div
from parser import match_neg
from parser import match_num
from parser import match_paras
from parser import resolve_add_sub
from parser import resolve_mul_div
from parser import resolve_neg
from parser import resolve_nums
from parser import resolve_paras
from parser import compute_arithmetic

num = Expression("1337", is_num=True)
num2 = Expression("666", is_resolved=False, is_num=True)
zero = Expression("0", is_num=True)
add = Expression("+", is_resolved=False)
sub = Expression("-", is_resolved=False)
mul = Expression("*", is_resolved=False)
div = Expression("/", is_resolved=False)
open_para = Expression("(", is_resolved=False)
closed_para = Expression(")", is_resolved=False)


class TestCalculatorMatchMulDiv:
    def test_arr_too_short(self):
        exp = match_mul_div([None, None, None], 1)
        assert exp is None

    def test_match_mul(self):
        exp = match_mul_div([num, mul, num], 0)
        assert exp == Expression("*", left=num, right=num)

    def test_match_div(self):
        exp = match_mul_div([num, div, num], 0)
        assert exp == Expression("/", left=num, right=num)

    def test_no_match(self):
        exp = match_mul_div([num, add, num], 0)
        assert exp is None


class TestCalculatorMatchAddSub:
    def test_arr_too_short(self):
        exp = match_add_sub([None, None, None], 1)
        assert exp is None

    def test_match_add(self):
        exp = match_add_sub([num, add, num], 0)
        assert exp == Expression("+", left=num, right=num)

    def test_match_sub(self):
        exp = match_add_sub([num, sub, num], 0)
        assert exp == Expression("-", left=num, right=num)

    def test_no_match(self):
        exp = match_add_sub([num, mul, num], 0)
        assert exp is None


class TestCalculatorMatchNeg:
    def test_arr_too_short(self):
        exp = match_neg([None, None], 1)
        assert exp is None

    def test_match_at_idx_zero(self):
        exp = match_neg([sub, num], 0)
        assert exp == Expression("-", left=zero, right=num)

    def test_no_match_at_idx_zero(self):
        exp = match_neg([add, num], 0)
        assert exp is None

    def test_match_open_para(self):
        exp = match_neg([open_para, sub, num], 1)
        assert exp == Expression("-", left=zero, right=num)

    def test_match_unresolved_operator(self):
        exp = match_neg([add, sub, num], 1)
        assert exp == Expression("-", left=zero, right=num)

    def test_no_match(self):
        exp = match_neg([num, sub, num], 1)
        assert exp is None


class TestCalculatorMatchNum:
    def test_match(self):
        exp = match_num([num2], 0)
        assert exp == num2
        assert exp.is_resolved

    def test_no_match(self):
        exp = match_num([add], 0)
        assert exp is None


class TestCalculatorMatchParas:
    def test_arr_too_short(self):
        exp = match_paras([None, None, None], 1)
        assert exp is None

    def test_match(self):
        exp = match_paras([open_para, num, closed_para], 0)
        assert exp == num

    def test_no_match(self):
        exp = match_paras([open_para, open_para, num], 0)
        assert exp is None


class TestCalculatorLex:
    def test_int(self):
        op = "1337"
        tokens, _ = lex(op)
        assert tokens == [Expression("1337", is_num=True)]

    def test_float(self):
        op = "1337.1337"
        tokens, _ = lex(op)
        assert tokens == [Expression("1337.1337", is_num=True)]

    def test_float_syntax_error(self):
        op = "1337.1337.1337"
        with pytest.raises(ArithmeticSyntaxError):
            lex(op)

    def test_depth(self):
        op = "1+(2+(3-4))"
        tokens, max_depth = lex(op)
        assert max_depth == 2
        assert len(tokens) == 11
        assert tokens[0].depth == 0  # 1
        assert tokens[1].depth == 0  # +
        assert tokens[2].depth == 1  # (
        assert tokens[3].depth == 1  # 2
        assert tokens[4].depth == 1  # +
        assert tokens[5].depth == 2  # (
        assert tokens[6].depth == 2  # 3
        assert tokens[7].depth == 2  # -
        assert tokens[8].depth == 2  # 4
        assert tokens[9].depth == 2  # )
        assert tokens[10].depth == 1  # )

    def test_operators_paras(self):
        op = "+-*/()"
        tokens, _ = lex(op)
        assert tokens == [Expression(x, is_resolved=False) for x in op]

    def test_illegal_characters(self):
        op = "do_evil_stuff()"
        with pytest.raises(ArithmeticSyntaxError):
            lex(op)


class TestCalculatorResolveX:
    def test_resolve_nums(self):
        resolved = resolve_nums([num, add, num2, sub])
        for exp in resolved:
            if exp.is_num:
                assert exp.is_resolved
            else:
                assert not exp.is_resolved

    def test_resolve_paras_change_detected(self):
        resolved, change_detected = resolve_paras([open_para, num2, closed_para])
        assert len(resolved) == 1
        assert change_detected

    def test_resolve_paras_no_change_detected(self):
        resolved, change_detected = resolve_paras([num2, closed_para])
        assert len(resolved) == 2
        assert not change_detected

    def test_resolve_mul_div_change_detected(self):
        resolved, change_detected = resolve_mul_div([num, mul, num2])
        assert len(resolved) == 1
        assert change_detected

    def test_resolve_mul_div_no_change_detected(self):
        resolved, change_detected = resolve_mul_div([num2, closed_para])
        assert len(resolved) == 2
        assert not change_detected

    def test_resolve_neg_change_detected(self):
        resolved, change_detected = resolve_neg([sub, num])
        assert len(resolved) == 1
        assert change_detected

    def test_resolve_neg_no_change_detected(self):
        resolved, change_detected = resolve_neg([num2, closed_para])
        assert len(resolved) == 2
        assert not change_detected

    def test_resolve_add_sub_change_detected(self):
        resolved, change_detected = resolve_add_sub([num, add, num])
        assert len(resolved) == 1
        assert change_detected

    def test_resolve_add_sub_no_change_detected(self):
        resolved, change_detected = resolve_add_sub([num2, closed_para])
        assert len(resolved) == 2
        assert not change_detected


class TestCalculatorBuildAst:
    def test_syntax_error(self):
        for op in ["(", "()", "23+", "+23", ""]:
            with pytest.raises(ArithmeticSyntaxError):
                build_ast(op)

    def test_legal(self):
        op = "(-(1337)+1337)"
        ast = build_ast(op)
        tree = Expression("+", Expression("-", zero, num), num)
        assert ast == tree
        assert ast.compute() == 0


class TestCalculatorCompute:
    @pytest.mark.parametrize(
        "operation",
        [
            "1337",
            "(1337)",
            "((1337))",
            "((((1337))))",
        ],
    )
    def test_compute_paranthesis(self, operation):
        ast = build_ast(operation)
        assert ast.compute() == eval(operation)

    @pytest.mark.parametrize(
        "operation",
        [
            "-1337",
            "-(1337)",
            "--1337",
            "--(1337)",
            "-(-1337)",
        ],
    )
    def test_compute_negate(self, operation):
        ast = build_ast(operation)
        assert ast.compute() == eval(operation)

    def test_compute_add_commutative(self):
        ast1 = build_ast("5+1")
        ast2 = build_ast("1+5")
        assert ast1.compute() == ast2.compute() == 6

    def test_compute_add_associative(self):
        ast1 = build_ast("(5+1)+1")
        ast2 = build_ast("5+(1+1)")
        assert ast1.compute() == ast2.compute() == 7

    def test_compute_mul_commutative(self):
        ast1 = build_ast("2*5")
        ast2 = build_ast("5*2")
        assert ast1.compute() == ast2.compute() == 10

    def test_compute_mul_associative(self):
        ast1 = build_ast("(2*5)*3")
        ast2 = build_ast("2*(5*3)")
        assert ast1.compute() == ast2.compute() == 30

    def test_compute_mul_distributive(self):
        ast1 = build_ast("10*(4+5)")
        ast2 = build_ast("(10*4)+(10*5)")
        ast3 = build_ast("(4+5)*10")
        ast4 = build_ast("(4*10)+(5*10)")
        assert (
            ast1.compute() == ast2.compute() == ast3.compute() == ast4.compute() == 90
        )

    def test_compute_sub_not_commutative(self):
        ast1 = build_ast("8-2")
        ast2 = build_ast("2-8")
        assert ast1.compute() == 6
        assert ast2.compute() == -6

    def test_compute_sub_not_associative(self):
        ast1 = build_ast("(8-2)-1")
        ast2 = build_ast("8-(2-1)")
        assert ast1.compute() == 5
        assert ast2.compute() == 7

    def test_compute_div_not_commutative(self):
        ast1 = build_ast("8/2")
        ast2 = build_ast("2/8")
        assert ast1.compute() == 4
        assert ast2.compute() == 0.25

    def test_compute_div_not_associative(self):
        ast1 = build_ast("(8/2)/4")
        ast2 = build_ast("8/(2/4)")
        assert ast1.compute() == 1
        assert ast2.compute() == 16

    def test_compute_div_right_distributive(self):
        ast1 = build_ast("10/(2+4)")
        ast2 = build_ast("(10/2)+(10/4)")
        ast3 = build_ast("(2+4)/10")
        ast4 = build_ast("(2/10)+(4/10)")
        assert ast1.compute() == 10 / 6
        assert ast2.compute() == 7.5
        assert ast3.compute() == pytest.approx(ast4.compute(), 0.00000000001) == 6 / 10

    def test_compute_div_by_zero(self):
        from math import isnan

        ast = build_ast("5/0")
        assert isnan(ast.compute())

    @pytest.mark.parametrize(
        "operation",
        [
            "-(1+2*3/4-(-1+2*3/4+(-1+2*3/4*(-1+2*3/4/(-1+2*3/4)))))",
            "(1+2*3/4)+(1+2*3/4)-(1+2*3/4)*(1+2*3/4)/(1+2*3/4)",
            "150/-150*-----150+150-200/300*(1000*20)-(-24-24-24)",
            "-1-1-(-5*---1*-92/93--2-2-(25/--25))+25+24.24+143-0--1*(0+1-1/1*1)",
            "1324/12/13/14/-15/--16/---17/(16*190/-23-25-11-0)*1000000",
            "100/10/-10*10*10*-1",
            "100+4*(5+6)",
            "1+2*(3+4*(5))",
            "-(-24-24)",
            "-(-0--0)",
            "2+2*3+2",
            "-2+--2",
            "-0-5",
            "-(-5)",
        ],
    )
    def test_compute(self, operation):
        assert compute_arithmetic(operation) == pytest.approx(eval(operation), 0.000000000000001)
