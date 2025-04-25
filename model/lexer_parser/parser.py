import ply.yacc as yacc
from .lexer import tokens

def p_states_decl(p):
    'states_decl : ID EQUALS LBRACE id_list RBRACE'
    p[0] = ('states', p[4])

def p_initstate_decl(p):
    'initstate_decl : ID EQUALS ID'
    p[0] = ('initstate', p[3])

def p_edges_decl(p):
    'edges_decl : ID EQUALS LBRACE edge_list RBRACE'
    p[0] = ('edges', p[4])

def p_edge_list(p):
    '''edge_list : edge
                | edge COMMA edge_list'''
    p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

def p_edge(p):
    'edge : LPAREN ID COMMA ID RPAREN'
    p[0] = (p[2], p[4])

def p_logicalformulas_decl(p):
    'logicalformulas_decl : ID EQUALS LBRACE formula_list RBRACE'
    p[0] = ('logicalformulas', p[4])

def p_formula_list(p):
    '''formula_list : formula
                   | formula COMMA formula_list'''
    p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

def p_formula(p):
    'formula : ID COLON LBRACE id_list RBRACE'
    p[0] = (p[1], p[4])

def p_values_decl(p):
    'values_decl : value_list'
    p[0] = ('values', p[1])

def p_value_list(p):
    '''value_list : value
                 | value COMMA value_list'''
    p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

def p_value(p):
    'value : ID EQUALS NUMBER'
    p[0] = (p[1], p[3])

def p_id_list(p):
    '''id_list : ID
              | ID COMMA id_list'''
    p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

def p_error(p):
    print(f"Syntax error at '{p.value}'")

parser = yacc.yacc()