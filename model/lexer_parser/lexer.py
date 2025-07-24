import ply.lex as lex

tokens = (
    'STATES', 'INIT', 'TRANS', 'LABEL', 'VALUES',
    'ID', 'NUMBER', 'LBRACE', 'RBRACE', 'LPAREN', 'RPAREN',
    'COMMA', 'COLON', 'SEMICOLON'
)

reserved = {
    'STATES': 'STATES',
    'INIT': 'INIT',
    'TRANS': 'TRANS',
    'LABEL': 'LABEL',
    'VALUES': 'VALUES'
}

def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    t.type = reserved.get(t.value, 'ID')
    return t

def t_NUMBER(t):
    r'-?\d+'
    t.value = int(t.value)
    return t

t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_COMMA = r','
t_COLON = r':'
t_SEMICOLON = r';'

t_ignore = ' \t\n'

def t_error(t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)

lexer = lex.lex()