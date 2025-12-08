import copy
from z3 import *

# Maintain a mapping from strings to Z3 constant symbols
constants = {}
def const_symbol(x):
    if x not in constants:
        constants[x] = Int(x)
    return constants[x]

# An arith-structure gives meaning to numerals, addition, negation,
# and multiplcation.

class StdInt:
    """The standard structure over the integers"""
    @staticmethod
    def of_numeral(num):
        return num

    @staticmethod
    def add(left, right):
        return left + right

    @staticmethod
    def negate(expr):
        return -expr

    @staticmethod
    def mul(left, right):
        return left * right

############################################################################
# Expressions are represented as a tree, where there is a different class for
# each type of node.  For example, an expression (x + 3*4) corresponds to a
# tree:
#     +                                    ExprPlus
#   /   \                                   /   \
#  x    *     -- corresponding to --  ExprVar ExprMul
#      / \                                     /     \
#     3   4                           ExprNumeral  ExprNumeral
# Each expression class is equipped with the methods:
#    eval: takes in an arith-structure, a state (valuation over that structure), and
#          produces a value in the arith-structure
#    vars: the set of variables used in that expression
# to_term: encode the expression as a Z3 term
# Formulas are similar.

class ExprVar:
    """Variables"""
    def __init__(self, name):
        self.name = name
    def eval(self, struct, state):
        return state[self.name]
    def __str__(self):
        return self.name
    def vars(self):
        return set([self.name])
    def to_term(self):
        return const_symbol(self.name)

class ExprNumeral:
    """Numerals"""
    def __init__(self, value):
        self.value = value
    def eval(self, struct, state):
        return struct.of_numeral(self.value)
    def __str__(self):
        return str(self.value)
    def vars(self):
        return set([])
    def to_term(self):
        return IntVal(self.value)

class BinaryExpr:
    """Abstract class representing binary expressions"""
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def eval(self, struct, state):
        pass
    def vars(self):
        return self.left.vars() | self.right.vars()

class ExprPlus(BinaryExpr):
    """Addition"""
    def eval(self, struct, state):
        return struct.add(self.left.eval(struct, state), self.right.eval(struct, state))
    def __str__(self):
        return "(" + str(self.left) + " + " + str(self.right) + ")"
    def to_term(self):
        return self.left.to_term() + self.right.to_term()
    
class ExprNeg:
    """Negation"""
    def __init__(self, expr):
        self.expr = expr
    def eval(self, struct, state):
        return struct.negate(self.expr.eval(struct, state))
    def __str__(self):
        return "-(" + str(self.expr) + ")"
    def to_term(self):
        return - self.expr.to_term()
    def vars(self):
        return self.expr.vars()

class ExprMul(BinaryExpr):
    """Multiplication"""
    def eval(self, struct, state):
        return struct.mul(self.left.eval(struct, state), self.right.eval(struct, state))
    def __str__(self):
        return "(" + str(self.left) + " * " + str(self.right) + ")"
    def to_term(self):
        return self.left.to_term() * self.right.to_term()

class FormLt(BinaryExpr):
    """Strictly less-than"""
    def eval(self, state):
        return self.left.eval(StdInt, state) < self.right.eval(StdInt, state)
    def __str__(self):
        return str(self.left) + " < " + str(self.right)
    def to_formula(self):
        return self.left.to_term() < self.right.to_term()

class FormEq(BinaryExpr):
    """Equal to"""
    def eval(self, state):
        return self.left.eval(StdInt, state) == self.right.eval(StdInt, state)
    def __str__(self):
        return str(self.left) + " == " + str(self.right)
    def to_formula(self):
        return self.left.to_term() == self.right.to_term()

class FormAnd(BinaryExpr):
    """And"""
    def eval(self, state):
        return self.left.eval(state) and self.right.eval(state)
    def __str__(self):
        return "(" + str(self.left) + " and " + str(self.right) + ")"
    def to_formula(self):
        return And(self.left.to_formula(), self.right.to_formula())
               
class FormOr(BinaryExpr):
    """Or"""
    def eval(self, state):
        return self.left.eval(state) or self.right.eval(state)
    def __str__(self):
        return "(" + str(self.left) + " or " + str(self.right) + ")"
    def to_formula(self):
        return Or(self.left.to_formula(), self.right.to_formula())

class FormNot:
    """Not"""
    def __init__(self, phi):
        self.phi = phi
    def eval(self, state):
        return not (self.phi.eval(state))
    def __str__(self):
        return "not(" + str(self.phi) + ")"
    def vars(self):
        return self.phi.vars()
    def to_formula(self):
        return Not(self.phi.to_formula())

############################################################################
# Control flow

class Command:
    def post(self, struct, state):
        pass
    def vars(self):
        pass
    def str(self):
        pass

class CmdAssign(Command):
    """Variable assignment"""
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def post(self, struct, state):
        if state.is_bottom():
            return state
        else:
            post_state = copy.deepcopy(state)
            post_state[self.lhs] = self.rhs.eval(struct, state)
            return post_state
    def vars(self):
        return set([self.lhs]) | self.rhs.vars()
    def __str__(self):
        return "%s := %s" % (self.lhs, str(self.rhs))

class CmdAssume(Command):
    """Guard"""
    def __init__(self, condition):
        self.condition = condition
    def post(self, struct, state):
        return state
    def vars(self):
        return self.condition.vars()
    def __str__(self):
        return "[%s]" % str(self.condition)


class CmdPrint(Command):
    """Print to stdout"""
    def __init__(self, expr):
        self.expr = expr
    def post(self, struct, state):
        return state
    def vars(self):
        return self.expr.vars()
    def __str__(self):
        return "print(%s)" % str(self.expr)

# Control flow automaton: directed graph with a command labelling each edge
class ControlFlowAutomaton:
    def __init__(self):
        self.max_loc = 0
        self.succs = {}
        self.labels = {}
        self.entry = 0
    def fresh_vertex(self):
        v = self.max_loc
        self.max_loc = v + 1
        self.succs[v] = set([])
        return v
    def add_edge(self, u, cmd, v):
        self.succs[u].add(v)
        self.labels[(u,v)] = cmd
    def successors(self, v):
        """Set of all successors of a given vertex"""
        return self.succs[v]
    def command(self, u, v):
        """The command associated with a given edge"""
        return self.labels[(u,v)]
    def vars(self):
        """The set of variables that appear in the CFA"""
        vars = set([])
        for command in self.labels.values():
            vars = vars | command.vars()
        return vars
    def locations(self):
        """The set of locations (vertices) in the CFA"""
        return set(range(self.max_loc))
############################################################################

############################################################################
# Statements are program phrases that can change the state of the program.
# Again, each statement is equipped with three methods:
#  execute: takes in a state, and produces nothing (but may change the state
#           or print something!)
#       pp: take in an indentation level and pretty-print a string
#           representation of the statement
#   to_cfa: takes in a control flow automaton, a source vertex, and a target
#           vertex, and adds the statement to the control flow automaton
#           connecting source and target
# print_annotation: like pp, but additionally print an annotation

class Stmt:
    """Statements"""
    def __init__(self):
        self.entry = None
    def pp(self, indent):
        pass
    def print_annotation(self, annotation, indent):
        pass
    def to_cfa(self, cfa, u, v):
        pass
    def execute(self, state):
        pass

class StmtAssign(Stmt):
    """Variable assignment"""
    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
    def execute(self, state):
        state[self.lhs] = self.rhs.eval(StdInt, state)
    def pp(self, indent):
        return ("    " * indent) + self.lhs + " = " + str(self.rhs) + "\n"
    def print_annotation(self, annotation, indent):
        print(("    " * indent) + "{" + str(annotation[self.entry]) + "}")
        print(("    " * indent) + self.lhs + " = " + str(self.rhs))
    def to_cfa(self, cfa, u, v):
        self.entry = u
        cfa.add_edge(u, CmdAssign(self.lhs, self.rhs), v)

class StmtIf(Stmt):
    """Conditional statement"""
    def __init__(self, cond, bthen, belse):
        self.cond = cond
        self.bthen = bthen
        self.belse = belse
    def execute(self, state):
        if (self.cond.eval(state)):
            self.bthen.execute(state)
        else:
            self.belse.execute(state)
    def pp(self, indent):
        program = ("    " * indent) + "if " + str(self.cond) + ":\n"
        program += self.bthen.pp(indent + 1)
        program += ("    " * indent) + "else:\n"
        program += self.belse.pp(indent + 1)
        return program

    def print_annotation(self, annotation, indent):
        print(("    " * indent) + "{" + str(annotation[self.entry]) + "}")
        print(("    " * indent) + "if " + str(self.cond) + ":")
        self.bthen.print_annotation(annotation, indent + 1)
        print(("    " * indent) + "else:")
        self.belse.print_annotation(annotation, indent + 1)

    def to_cfa(self, cfa, u, v):
        self.entry = u
        then_target = cfa.fresh_vertex()
        else_target = cfa.fresh_vertex()
        cfa.add_edge(u, CmdAssume(self.cond), then_target)
        cfa.add_edge(u, CmdAssume(FormNot(self.cond)), else_target)
        self.bthen.to_cfa(cfa, then_target, v)
        self.belse.to_cfa(cfa, else_target, v)

class StmtBlock(Stmt):
    """Sequence of statements"""
    def __init__(self, block):
        self.block = block
    def execute(self, state):
        for stmt in self.block:
            stmt.execute(state)
    def pp(self, indent):
        return "".join(map(lambda x: x.pp(indent), self.block))

    def print_annotation(self, annotation, indent):
        for stmt in self.block:
            stmt.print_annotation(annotation, indent)

    def to_cfa(self, cfa, u, v):
        self.entry = u
        last = u
        for i in range(len(self.block)-1):
            nxt = cfa.fresh_vertex()
            self.block[i].to_cfa(cfa, last, nxt)
            last = nxt
        self.block[len(self.block)-1].to_cfa(cfa, last, v)

class StmtWhile(Stmt):
    """While loop"""
    def __init__(self, cond, body):
        self.cond = cond
        self.body = body
    def execute(self, state):
        while self.cond.eval(state):
            self.body.execute(state)
    def pp(self, indent):
        program = ("    " * indent) + "while " + str(self.cond) + ":\n"
        program += self.body.pp(indent + 1)
        return program
    def print_annotation(self, annotation, indent):
        print(("    " * indent) + "{" + str(annotation[self.entry]) + "}")
        print(("    " * indent) + "while " + str(self.cond) + ":")
        self.body.print_annotation(annotation, indent + 1)

    def to_cfa(self, cfa, u, v):
        self.entry = u
        w = cfa.fresh_vertex()
        cfa.add_edge(u, CmdAssume(self.cond), w)
        cfa.add_edge(u, CmdAssume(FormNot(self.cond)), v)
        self.body.to_cfa(cfa, w, u)

class StmtPrint(Stmt):
    """Print to stdout"""
    def __init__(self, expr):
        self.expr = expr
    def execute(self, state):
        print(self.expr.eval(StdInt, state))
    def pp(self, indent):
        return ("    " * indent) + "print(" + str(self.expr) + ")\n"
    def to_cfa(self, cfa, u, v):
        self.entry = u
        cfa.add_edge(u, CmdPrint(self.expr), v)
    def print_annotation(self, annotation, indent):
        print(("    " * indent) + "{" + str(annotation[self.entry]) + "}")
        print(("    " * indent) + "print(" + str(self.expr) + ")")



class Houdini:
    def __init__(self, P):
        self.formulas = set([])
        self.P = P

    def __setformula__(self, f):
        self.formulas = f

    def __getformula__(self):
        return self.formulas

    def __str__(self):
        return ",".join(str(x) for x in self.formulas)
    
    @staticmethod
    def pre(precon, P):
        valuation = Houdini(P)
        valuation.formulas = set([precon])
        return valuation


    @staticmethod
    def bottom(P):
        valuation = Houdini(P)
        valuation.formulas = P
        return valuation

    @staticmethod
    def helper(var, substitution, form):

        if isinstance(form, FormAnd):
            return (FormAnd(Houdini.helper(var,substitution,form.left), Houdini.helper(var,substitution,form.right)))
        elif isinstance(form, FormOr):
            return (FormOr(Houdini.helper(var,substitution,form.left), Houdini.helper(var,substitution,form.right)))
        elif isinstance(form, FormNot):
            return (FormNot(Houdini.helper(var,substitution,form.phi)))
        elif isinstance(form, FormEq):
            return (FormEq(Houdini.helper(var,substitution,form.left), Houdini.helper(var,substitution,form.right)))
        elif isinstance(form, FormLt):
            return (FormLt(Houdini.helper(var,substitution,form.left), Houdini.helper(var,substitution,form.right)))
        elif isinstance(form, ExprMul):
            return (ExprMul(Houdini.helper(var,substitution,form.left), Houdini.helper(var,substitution,form.right)))
        elif isinstance(form, ExprPlus):
            return (ExprPlus(Houdini.helper(var,substitution,form.left), Houdini.helper(var,substitution,form.right)))
        elif isinstance(form, ExprNeg):
            return (ExprNeg(Houdini.helper(var,substitution,form.expr)))
        elif isinstance(form, ExprNumeral):
            return form
        elif isinstance(form, ExprVar):
            if (form.name == var):
                return substitution
            else:
                return form
            

    @staticmethod
    def wp(command, annotation):
        if isinstance(command, CmdAssign):
            lhs = command.lhs
            rhs = command.rhs
            erg = Houdini.helper(lhs, rhs, annotation)
            return erg
        else:
            return (FormOr(FormNot(command.condition), annotation))

    @staticmethod
    def consequence(form1, form2):
        s = Solver()
        for a in form1.formulas:
            s.add(a.to_formula())

        s.add(Not(form2.to_formula()))
        return s.check() == unsat


def analyze_houdini(domain, cfa, P, precondition):
    """Given a domain and a control flow automaton, compute the least
    inductive annotation (i.e., for all edge (u,v), we have
    domain.leq(annotation[u].post(cfa.command(u,v)), annotation[v])) such that
    annotation[cfa.entry] is domain.top
    """
    
    annotation = {}
    for v in cfa.locations():
        annotation[v] = domain.bottom(P.copy())
    annotation[cfa.entry] = domain.pre(precondition, P.copy())

    work = []
    for v in cfa.locations():
        work.append(v)

    while (len(work) > 0):
        u = work.pop(0)
        for v in cfa.successors(u):
            tmp = set([])
            inwork = False
            for F in annotation[v].formulas:
                if (not domain.consequence(annotation[u],domain.wp(cfa.command(u,v),F))):
                    tmp.add(F)
                    if (not inwork):
                        work.append(v)
                        inwork = True
            for a in tmp:
                annotation[v].formulas.remove(a)
    return annotation