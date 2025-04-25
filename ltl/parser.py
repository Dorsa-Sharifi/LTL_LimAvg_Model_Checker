import ply.lex as lex
import ply.yacc as yacc

class LTLParser:
    # Tokens
    tokens = (
        'TRUE', 'FALSE', 'PROPOSITION',
        'NOT', 'AND', 'OR', 'IMPLIES', 'EQUIV',
        'NEXT', 'EVENTUALLY', 'ALWAYS', 'UNTIL', 'RELEASE',
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
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_ignore = ' \t\n'

    def t_PROPOSITION(self, t):
        r'[a-z][a-z0-9_]*'
        return t

    def t_error(self, t):
        raise SyntaxError(f"Illegal character '{t.value[0]}'")

    # Operator precedence table
    precedence = (
        ('left', 'EQUIV'),
        ('left', 'IMPLIES'),
        ('left', 'OR'),
        ('left', 'AND'),
        ('right', 'NOT', 'NEXT', 'EVENTUALLY', 'ALWAYS'),
        ('left', 'UNTIL', 'RELEASE'),
    )

    def p_formula(self, p):
        '''
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
                | atomic
                | LPAREN formula RPAREN
        '''
        if len(p) == 4 and p[1] == '(':
            p[0] = p[2]
        elif len(p) == 3:
            p[0] = (p[1], p[2])
        elif len(p) == 4:
            p[0] = (p[2], p[1], p[3])
        else:
            p[0] = p[1]

    def p_atomic(self, p):
        '''
        atomic : TRUE
               | FALSE
               | PROPOSITION
        '''
        p[0] = ('atomic', p[1])

    def p_error(self, p):
        raise SyntaxError(f"Syntax error at '{p.value}'")

    def __init__(self):
        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self, debug=False)

    def parse(self, formula):
        return self.parser.parse(formula)

    def negate(self, tree):
        """Negate parse tree using temporal dualities"""
        if tree[0] == 'atomic':
            return ('!', tree)
        elif tree[0] == '!':
            return tree[1]
        elif tree[0] == '&&':
            return ('||', self.negate(tree[1]), self.negate(tree[2]))
        elif tree[0] == '||':
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