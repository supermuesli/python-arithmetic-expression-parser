# Python arithmetic expression parser without `eval`

This parser can handle **addition**, **subtraction**, **division**, **multiplication**, **parentheses**, **floating point numbers** and **negation**.

It builds its own AST and **does not rely on using `eval` for computations**, making it completely safe for use in production.

## Usage
```
>>> from parser import compute_arithmetic as compute
>>>
>>> compute("-24")
-24
>>> compute("1+20+3.1415")
24.1415
>>> compute("-1-1-(-5*---1*-92/93--2-2-(25/--25))+25+24.24+143-0--1*(0+1-1/1*1)")
196.1862365591398
>>> compute("1/0")
nan
```

