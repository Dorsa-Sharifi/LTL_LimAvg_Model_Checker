import ply.yacc as yacc
from .lexer import tokens

def p_kripke(p):
    '''kripke : states_decl init_decl trans_decl label_decl values_decl'''
    p[0] = {
        'states': p[1],
        'initial': p[2],
        'transitions': p[3],
        'labels': p[4],
        'values': p[5]
    }

def p_states_decl(p):
    'states_decl : STATES COLON LBRACE id_list RBRACE SEMICOLON'
    p[0] = p[4]

def p_init_decl(p):
    'init_decl : INIT COLON LBRACE id_list RBRACE SEMICOLON'
    p[0] = p[4]

def p_trans_decl(p):
    'trans_decl : TRANS COLON LBRACE trans_list RBRACE SEMICOLON'
    p[0] = p[4]

def p_trans_list(p):
    '''trans_list : trans_pair
                 | trans_pair COMMA trans_list'''
    p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

def p_trans_pair(p):
    'trans_pair : LPAREN ID COMMA ID RPAREN'
    p[0] = (p[2], p[4])

def p_label_decl(p):
    'label_decl : LABEL COLON LBRACE label_list RBRACE SEMICOLON'
    p[0] = {k: v for k, v in p[4]}

def p_label_list(p):
    '''label_list : label_assignment
                 | label_assignment COMMA label_list'''
    p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

def p_label_assignment(p):
    'label_assignment : ID COLON LBRACE id_list RBRACE'
    p[0] = (p[1], p[4])

def p_values_decl(p):
    'values_decl : VALUES COLON LBRACE value_list RBRACE SEMICOLON'
    p[0] = {k: v for k, v in p[4]}

def p_value_list(p):
    '''value_list : value_assignment
                 | value_assignment COMMA value_list'''
    p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

def p_value_assignment(p):
    'value_assignment : ID COLON NUMBER'
    p[0] = (p[1], p[3])

def p_id_list(p):
    '''id_list : ID
               | ID COMMA id_list'''
    p[0] = [p[1]] if len(p) == 2 else [p[1]] + p[3]

def p_error(p):
    if p:
        print(f"Syntax error at '{p.value}' (type: {p.type}, line: {p.lineno})")
    else:
        print("Syntax error at EOF")

parser = yacc.yacc()