import os
import google.generativeai as genai
from invgen import *
from pyparsing import *
from functools import reduce
import sys
import random
from fastapi import FastAPI, Request


def mk_numeral(toks):
    return [ExprNumeral(int(toks[0]))]
def mk_var(toks):
    return [ExprVar(str(toks[0]))]
def mk_plus_minus(toks):
    curr = toks[0][0]
    for i in range(1, len(toks[0]), 2):
        if toks[0][i] == "+":
            curr = ExprPlus(curr, toks[0][i+1])
        else:
            curr = ExprPlus(curr, ExprNeg(toks[0][i+1]))
    return [curr]
def mk_mul(toks):
    return [reduce(ExprMul, toks[0][0::2])]
def mk_neg(toks):
    return [ExprNeg(toks[0][1])]

def mk_not(toks):
    return [FormNot(toks[0][1])]
def mk_and(toks):
    return [reduce(FormAnd, toks[0][0::2])]
def mk_or(toks):
    return [reduce(FormOr, toks[0][0::2])]
def mk_atom(toks):
    if toks[1] == "=":
        return FormEq(toks[0], toks[2])
    elif toks[1] == "<":
        return FormLt(toks[0], toks[2])
    elif toks[1] == ">":
        return FormNot(FormOr(FormLt(toks[0], toks[2]),
                      FormEq(toks[0], toks[2])))
    elif toks[1] == ">=":
        return FormNot(FormLt(toks[0], toks[2]))
    else:
        return FormOr(FormLt(toks[0], toks[2]),
                      FormEq(toks[0], toks[2]))

def mk_assign(toks):
    return StmtAssign(str(toks[0]), toks[2])
def mk_block(toks):
    if len(toks) == 1:
        return toks[0]
    else:
        return StmtBlock(toks)
def mk_while(toks):
    return StmtWhile(toks[1], toks[2])
def mk_if(toks):
    return StmtIf(toks[1], toks[2], toks[3])
def mk_print(toks):
    return StmtPrint(toks[1])

integer = pyparsing_common.signed_integer
varname = pyparsing_common.identifier
var = pyparsing_common.identifier
integer.setParseAction(mk_numeral)
var.setParseAction(mk_var)

expr = infixNotation(integer | var,
                     [('-', 1, opAssoc.RIGHT,mk_neg),
                      ('*', 2, opAssoc.LEFT,mk_mul),
                      (oneOf('+ -'), 2, opAssoc.LEFT,mk_plus_minus)])

atom = expr + (Literal("<=") | Literal("<") | Literal("=") | Literal(">=") | Literal(">")) + expr
atom.setParseAction(mk_atom)

formula = infixNotation(atom,
                        [("not", 1, opAssoc.RIGHT, mk_not),
                         ("and", 2, opAssoc.LEFT, mk_and),
                         ("or", 2, opAssoc.LEFT, mk_or)])


block = Forward()
assign_stmt = varname + ":=" + expr
if_stmt = Keyword("if") + formula + block + Keyword("else").suppress() + block
while_stmt = Keyword("while") + formula + block
print_stmt = Literal("print") + Literal('(').suppress() + expr + Literal(')').suppress()
stmt = if_stmt ^ while_stmt ^ print_stmt ^ assign_stmt
block << Literal('{').suppress() + delimitedList(stmt, delim=';') + Literal('}').suppress()
program = delimitedList(stmt, delim=';')
assign_stmt.setParseAction(mk_assign)
if_stmt.setParseAction(mk_if)
while_stmt.setParseAction(mk_while)
block.setParseAction(mk_block)
print_stmt.setParseAction(mk_print)
program.setParseAction(mk_block)

def analyze_and_print(domain, stmt,P, precondition, postcondition):
    cfa = ControlFlowAutomaton()
    loc_entry = cfa.fresh_vertex()
    loc_exit = cfa.fresh_vertex()
    stmt.to_cfa(cfa, loc_entry, loc_exit)
    annotation = analyze_houdini(domain, cfa,P, precondition)
    stmt.print_annotation(annotation, 0)
    print("{" + str(annotation[loc_exit]) + "}")
    s = Solver()
    for form in annotation[loc_exit].formulas:
        s.add(form.to_formula())
    s.add(Not(postcondition.to_formula()))
    res = s.check()
    if (res == unsat):
        print("program verified")
        return True
    else:
        print("not able to verify program")
        return False

app = FastAPI()

@app.post("/check")
async def backend(req: Request):
    data = await req.json()
    print(data)
    pre = data["pre"]
    if (pre == "True"):
        pre = "1 = 1"
    elif pre == "False":
        pre = "0 = 1"
    prog = data["program"]
    post = data["post"]
    if (post == "True"):
        post = "1 = 1"
    elif post == "False":
        post = "0 = 1"
    progr = program.parseString(prog, parseAll=True)[0]
    # program = program.parseFile(sys.argv[1],parseAll=True)[0]
    # with open(sys.argv[1], "r") as f:
    #     s = f.read()
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    grammar = """expr = infixNotation(integer | var,
                    [('-', 1, opAssoc.RIGHT,mk_neg),
                    ('*', 2, opAssoc.LEFT,mk_mul),
                    (oneOf('+ -'), 2, opAssoc.LEFT,mk_plus_minus)])

atom = expr + (Literal("<=") | Literal("<") | Literal("=") | Literal(">=") | Literal(">")) + expr
atom.setParseAction(mk_atom)

formula = infixNotation(atom,
                    [("not", 1, opAssoc.RIGHT, mk_not),
                        ("and", 2, opAssoc.LEFT, mk_and),
                        ("or", 2, opAssoc.LEFT, mk_or)])"""
    prompt = "Give me precisely all the formulas (state predicates) the Houdini algorithm would need to proof the correctness of this program for precondition " +pre+ " and postcondition "+post+ "strictly adhering to this grammar " + grammar+". Really think about it and make sure that you give all invariants so that it can prove the desired result! Don't write any additional text and put every invariant on a new line. Really think about it and give it your best! Try to keep each individual formula short: " + prog
    prompt2 = """You are a helpful AI software assistant that reasons about how code behaves. Given a program,
you can find ALL formulas (or state predicates) for Houdini, which can then be used to verify some property in the program.
Houdini is a software verification algorithm for programs. The input to Houdini is a set of candidate predicates. For the given program, find
the necessary predicates/state predicates to help Houdini verify the post-condition.
Instructions:
• Make a note of the pre-conditions or variable assignments in the program.
• Analyze the loop body and make a note of the loop condition.
• Output loop invariants that are true
(i) before the loop execution,
(ii) in every iteration of the loop and
(iii) after the loop termination,
such that the loop invariants imply the post condition.
• If a loop invariant is a conjunction, split it into its parts.
We have precondition: """ + pre + " postcondition: " + post + " and program " + prog + """. 
For all variables, add conjunctions that bound the maximum and minimum values that they
can take, if such bounds exist.
If a variable is always equal to or smaller or larger than another variable, add a conjunction
for their relation.
If the assertion is guarded by a condition, use the guard condition in an implication.
If certain variables are non-deterministic at the beginning or end of the loop, use an implication
to make the invariant trivially true at that location.
Output the state formulas needed for the program above. Lets think step by step. Really think about it and give it your best!
Don't write any additional text and put every predicate on a new line and strictly adhere to this grammar
""" + grammar
    
    inequalityprompt = "Try to capture all of the relations of the variables using different strict and not strict inequalities"
    response = model.generate_content(prompt).text

    # 4️⃣ Print the result
    print("Response: ", response, "end")
    lines = response.splitlines()
    # print("lines: ",lines)
    P = set([])
    for line in lines:
        # print(line)
        if (line == "True" or line == "true"):
            line = "1 = 1"
        elif (line == "False" or line == "false"):
            line = "0 = 1"
        form = formula.parseString(line)[0]
        # print(type(form))
        # print(form)
        # print("form: ",form.to_formula())
        P.add(form)
    ok = analyze_and_print(Houdini, progr,P, formula.parseString(pre)[0], formula.parseString(post)[0])
    return {"success": ok, "invariants": response}

