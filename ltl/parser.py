import ply.lex as lex
import ply.yacc as yacc
from functools import reduce


class LTLParser:
    tokens = (
        'TRUE', 'FALSE', 'PROPOSITION',
        'NOT', 'AND', 'OR', 'IMPLIES', 'EQUIV',
        'NEXT', 'EVENTUALLY', 'ALWAYS', 'UNTIL', 'RELEASE',
        'LIMSUPAVG', 'LIMINFAVG',
        'LPAREN', 'RPAREN'
    )

    # Lexer rules
    t_TRUE = r'True'
    t_FALSE = r'False'
    t_NOT = r'!'
    t_AND = r'&&'
    t_OR = r'\|\|'
    t_IMPLIES = r'->'
    t_EQUIV = r'<->'
    t_NEXT = r'X'
    t_EVENTUALLY = r'F'
    t_ALWAYS = r'G'
    t_UNTIL = r'U'
    t_RELEASE = r'R'
    t_LIMSUPAVG = r'LimSupAvg'
    t_LIMINFAVG = r'LimInfAvg'
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_ignore = ' \t\n'

    def t_PROPOSITION(self, t):
        r"""[a-z][a-z0-9_]*"""
        return t

    def t_error(self, t):
        raise SyntaxError(f"Illegal character '{t.value[0]}'")

    precedence = (
        ('left', 'EQUIV'),
        ('left', 'IMPLIES'),
        ('left', 'OR'),
        ('left', 'AND'),
        ('right', 'NOT', 'NEXT', 'EVENTUALLY', 'ALWAYS'),
        ('left', 'UNTIL', 'RELEASE'),
        ('nonassoc', 'LIMSUPAVG', 'LIMINFAVG'),
    )

    def p_formula(self, p):
        """
        formula : formula EQUIV formula
                | formula IMPLIES formula
                | formula OR formula
                | formula AND formula
                | NOT formula
                | NEXT formula
                | EVENTUALLY formula
                | ALWAYS formula
                | formula UNTIL formula
                | formula RELEASE formula
                | limavg_formula
                | atomic
                | LPAREN formula RPAREN
        """
        if len(p) == 4 and p[1] == '(':
            p[0] = p[2]
        elif len(p) == 3:
            p[0] = (p[1], p[2])
        elif len(p) == 4:
            p[0] = (p[2], p[1], p[3])
        else:
            p[0] = p[1]

    def p_limavg_formula(self, p):
        """
        limavg_formula : LIMSUPAVG LPAREN formula RPAREN
                      | LIMINFAVG LPAREN formula RPAREN
        """
        p[0] = (p[1], p[3])

    def p_atomic(self, p):
        """
        atomic : TRUE
               | FALSE
               | PROPOSITION
        """
        p[0] = ('atomic', p[1])

    def p_error(self, p):
        raise SyntaxError(f"Syntax error at '{p.value}'")

    def __init__(self):
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self, debug=False, write_tables=False)

    def parse(self, formula):
        return self.parser.parse(formula)

    def negate(self, tree):
        """Negate parse tree with LimAvg support"""
        if tree[0] == 'atomic':
            return ('!', tree)
        elif tree[0] == '!':
            return tree[1]
        elif tree[0] == 'LimSupAvg':
            return ('LimInfAvg', self.negate(tree[1]))
        elif tree[0] == 'LimInfAvg':
            return ('LimSupAvg', self.negate(tree[1]))
        elif tree[0] in ('&&', 'AND'):
            return ('||', self.negate(tree[1]), self.negate(tree[2]))
        elif tree[0] in ('||', 'OR'):
            return ('&&', self.negate(tree[1]), self.negate(tree[2]))
        elif tree[0] == 'X':
            return ('X', self.negate(tree[1]))
        elif tree[0] == 'F':
            return ('G', self.negate(tree[1]))
        elif tree[0] == 'G':
            return ('F', self.negate(tree[1]))
        elif tree[0] == 'U':
            return ('R', self.negate(tree[1]), self.negate(tree[2]))
        elif tree[0] == 'R':
            return ('U', self.negate(tree[1]), self.negate(tree[2]))
        return ('!', tree)

    def find_limit_avg_assertions(self, formula):
        """Find all unique LimAvg assertions in formula"""
        if not isinstance(formula, tuple):
            return []

        if formula[0] in ('LimSupAvg', 'LimInfAvg'):
            return [formula]

        assertions = []
        for child in formula[1:]:
            assertions.extend(self.find_limit_avg_assertions(child))

        seen = set()
        unique = []
        for a in assertions:
            if str(a) not in seen:
                seen.add(str(a))
                unique.append(a)
        return unique

    def substitute_limit_avg(self, formula, substitutions):
        """Replace LimAvg assertions with values"""
        if not isinstance(formula, tuple):
            return formula

        if formula[0] in ('LimSupAvg', 'LimInfAvg'):
            return substitutions.get(formula, formula)

        return (formula[0],) + tuple(
            self.substitute_limit_avg(child, substitutions)
            for child in formula[1:]
        )