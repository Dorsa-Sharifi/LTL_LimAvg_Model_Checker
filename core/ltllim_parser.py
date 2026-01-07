import numpy as np
import ply.lex as lex
import ply.yacc as yacc
from itertools import product as itertools_product
import subprocess
import json
from typing import Set, List, Tuple, Dict, Any
from scipy.spatial import ConvexHull
from fractions import Fraction
from core.QuantitativeKripkeStructure import QuantitativeKripkeStructure

from core.spot_converter import SpotConverter
from core.z3_solver import Z3Solver
from core.config import get_config, MultiOSConfig
from core.platform_detector import PLATFORM, PLATFORM_STR, print_platform_info

WSLSpotConverter = SpotConverter
WSLZ3Solver = Z3Solver


class LTLimProcessor:
    def __init__(self):
        self.tokens = [
            'IDENTIFIER', 'REAL',
            'NEG', 'AND', 'OR', 'IMPLIES', 'IFF',
            'NEXT', 'FINALLY', 'GLOBALLY', 'UNTIL', 'RELEASE',
            'SUM', 'AVG', 'LIM_INF_AVG', 'LIM_SUP_AVG',
            'GEQ', 'LEQ', 'EQ', 'GT', 'LT',
            'PLUS', 'MINUS', 'TIMES', 'DIVIDE',
            'LPAREN', 'RPAREN'
        ]

        self.lexer = lex.lex(module=self)
        self.parser = yacc.yacc(module=self, debug=False)
        self.variables = set()
        self.propositions = set()
        self.limit_avg_assertions = []

    # Lexer (same as before)
    def t_LIM_INF_AVG(self, t):
        r'LimInfAvg'
        return t

    def t_LIM_SUP_AVG(self, t):
        r'LimSupAvg'
        return t

    def t_SUM(self, t):
        r'Sum'
        return t

    def t_AVG(self, t):
        r'Avg'
        return t

    def t_NEXT(self, t):
        r'\bX\b'
        return t

    def t_FINALLY(self, t):
        r'\bF\b'
        return t

    def t_GLOBALLY(self, t):
        r'\bG\b'
        return t

    def t_UNTIL(self, t):
        r'\bU\b'
        return t

    def t_RELEASE(self, t):
        r'\bR\b'
        return t

    def t_NEG(self, t):
        r'¬|!'
        return t

    def t_AND(self, t):
        r'∧|&&'
        return t

    def t_OR(self, t):
        r'∨|\|\|'
        return t

    def t_IMPLIES(self, t):
        r'->|=>|→'
        return t

    def t_IFF(self, t):
        r'<->|<=>|↔'
        return t

    def t_GEQ(self, t):
        r'>='
        return t

    def t_LEQ(self, t):
        r'<='
        return t

    def t_EQ(self, t):
        r'='
        return t

    def t_GT(self, t):
        r'>'
        return t

    def t_LT(self, t):
        r'<'
        return t

    def t_PLUS(self, t):
        r'\+'
        return t

    def t_MINUS(self, t):
        r'-'
        return t

    def t_TIMES(self, t):
        r'\*'
        return t

    def t_DIVIDE(self, t):
        r'/'
        return t

    def t_LPAREN(self, t):
        r'\('
        return t

    def t_RPAREN(self, t):
        r'\)'
        return t

    def t_REAL(self, t):
        r'\d+\.\d+|\d+'
        t.value = float(t.value) if '.' in t.value else int(t.value)
        return t

    def t_IDENTIFIER(self, t):
        r'[a-zA-Z][a-zA-Z0-9_]*'
        return t

    def t_error(self, t):
        print(f"Illegal character '{t.value[0]}'")
        t.lexer.skip(1)

    t_ignore = ' \t\n'

    # Precedence rules
    precedence = (
        ('left', 'OR'),
        ('left', 'AND'),
        ('right', 'IMPLIES'),
        ('right', 'IFF'),
        ('right', 'NEG'),
        ('left', 'UNTIL', 'RELEASE'),
        ('left', 'NEXT', 'FINALLY', 'GLOBALLY'),
    )

    def p_formula(self, p):
        """
        formula : atomic_formula
                | LPAREN formula RPAREN
                | NEG formula
                | NEXT formula
                | FINALLY formula
                | GLOBALLY formula
                | formula AND formula
                | formula OR formula
                | formula IMPLIES formula
                | formula IFF formula
                | formula UNTIL formula
                | formula RELEASE formula
        """
        if len(p) == 2:
            p[0] = p[1]
        elif p[1] == '(':
            p[0] = p[2]
        elif p[1] in ['¬', '!', 'X', 'F', 'G']:
            p[0] = (p[1], p[2])
        else:
            op_map = {
                '->': '→', '=>': '→', '→': '→',
                '<->': '↔', '<=>': '↔', '↔': '↔',
                '∧': '∧', '&&': '∧',
                '∨': '∨', '||': '∨',
                'U': 'U', 'R': 'R'
            }
            canonical_op = op_map.get(p[2], p[2])
            p[0] = (canonical_op, p[1], p[3])

    def p_atomic_formula(self, p):
        """
        atomic_formula : proposition
                      | path_assertion
                      | prefix_assertion
        """
        p[0] = p[1]

    def p_proposition(self, p):
        """proposition : IDENTIFIER"""
        self.propositions.add(p[1])
        p[0] = ('prop', p[1])

    def p_path_assertion(self, p):
        """
        path_assertion : LIM_INF_AVG LPAREN IDENTIFIER RPAREN GEQ REAL
                      | LIM_SUP_AVG LPAREN IDENTIFIER RPAREN GEQ REAL
                      | LIM_INF_AVG LPAREN IDENTIFIER RPAREN LEQ REAL
                      | LIM_SUP_AVG LPAREN IDENTIFIER RPAREN LEQ REAL
                      | LIM_INF_AVG LPAREN IDENTIFIER RPAREN GT REAL
                      | LIM_SUP_AVG LPAREN IDENTIFIER RPAREN GT REAL
                      | LIM_INF_AVG LPAREN IDENTIFIER RPAREN LT REAL
                      | LIM_SUP_AVG LPAREN IDENTIFIER RPAREN LT REAL
        """
        self.variables.add(p[3])
        assertion = (p[1], p[3], p[5], p[6])  # (type, variable, operator, value)
        self.limit_avg_assertions.append(assertion)
        p[0] = assertion

    def p_prefix_assertion(self, p):
        """
        prefix_assertion : arithmetic_expr GEQ arithmetic_expr
                        | arithmetic_expr LEQ arithmetic_expr
                        | arithmetic_expr GT arithmetic_expr
                        | arithmetic_expr LT arithmetic_expr
                        | arithmetic_expr EQ arithmetic_expr
        """
        p[0] = ('assert', p[1], p[2], p[3])

    def p_arithmetic_expr(self, p):
        """
        arithmetic_expr : term
                       | arithmetic_expr PLUS term
                       | arithmetic_expr MINUS term
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = (p[2], p[1], p[3])

    def p_term(self, p):
        """
        term : factor
             | term TIMES factor
             | term DIVIDE factor
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = (p[2], p[1], p[3])

    def p_factor(self, p):
        """
        factor : REAL
               | variable
               | sum_expr
               | avg_expr
               | LPAREN arithmetic_expr RPAREN
        """
        if len(p) == 2:
            p[0] = p[1]
        else:
            p[0] = p[2]

    def p_variable(self, p):
        """variable : IDENTIFIER"""
        self.variables.add(p[1])
        p[0] = ('var', p[1])

    def p_sum_expr(self, p):
        """sum_expr : SUM LPAREN IDENTIFIER RPAREN"""
        self.variables.add(p[3])
        p[0] = ('Sum', p[3])

    def p_avg_expr(self, p):
        """avg_expr : AVG LPAREN IDENTIFIER RPAREN"""
        self.variables.add(p[3])
        p[0] = ('Avg', p[3])

    def p_error(self, p):
        if p:
            raise SyntaxError(f"Syntax error at '{p.value}'")
        else:
            raise SyntaxError("Syntax error at EOF")

    def parse(self, formula):
        """Parse formula and return parse tree"""
        self.variables.clear()
        self.propositions.clear()
        self.limit_avg_assertions.clear()
        return self.parser.parse(formula, lexer=self.lexer)

    def negate_formula(self, tree):
        """Step 1: Negate the parse tree to get ϕ = ¬ψ - FIXED VERSION"""
        if tree is None:
            return None

        if isinstance(tree, tuple):
            op = tree[0]

            if op in ['prop', 'var', 'real', 'Sum', 'Avg']:
                return ('¬', tree)

            elif op in ['LimInfAvg', 'LimSupAvg']:
                old_op = tree[2]
                neg_ops = {'>=': '<', '<=': '>', '>': '<=', '<': '>=', '=': '!='}
                new_op = neg_ops.get(old_op, f'¬{old_op}')

                return (tree[0], tree[1], new_op, tree[3])

            elif op == 'assert':
                old_op = tree[2]
                neg_ops = {'>=': '<', '<=': '>', '>': '<=', '<': '>=', '=': '!='}
                new_op = neg_ops.get(old_op, f'¬{old_op}')
                return ('assert', tree[1], new_op, tree[3])

            elif op == '¬':
                return tree[1]

            elif op == 'X':
                return 'X', self.negate_formula(tree[1])

            elif op == 'F':
                return 'G', self.negate_formula(tree[1])

            elif op == 'G':
                return 'F', self.negate_formula(tree[1])

            elif op == '∧':
                return '∨', self.negate_formula(tree[1]), self.negate_formula(tree[2])

            elif op == '∨':
                return '∧', self.negate_formula(tree[1]), self.negate_formula(tree[2])

            elif op == '→':
                return '∧', tree[1], self.negate_formula(tree[2])

            elif op == '↔':
                return ('∨',
                        ('∧', tree[1], self.negate_formula(tree[2])),
                        ('∧', self.negate_formula(tree[1]), tree[2]))

            elif op == 'U':
                return 'R', self.negate_formula(tree[1]), self.negate_formula(tree[2])

            elif op == 'R':
                return 'U', self.negate_formula(tree[1]), self.negate_formula(tree[2])

            elif op == 'paren':
                return 'paren', self.negate_formula(tree[1])

            else:
                return tree

        return tree

    def tree_to_string(self, tree):
        if tree is None:
            return ""

        if tree in ['T', 'true']:
            return "true"
        elif tree in ['F', 'false']:
            return "false"

        if isinstance(tree, tuple):
            op = tree[0]

            if op == 'prop':
                return tree[1]
            elif op == 'var':
                return tree[1]
            elif op == 'real':
                return str(tree[1])
            elif op in ['LimInfAvg', 'LimSupAvg']:
                operator = tree[2]
                if operator == '<=':
                    operator = '<='
                elif operator == '>=':
                    operator = '>='
                elif operator == '<':
                    operator = '<'
                elif operator == '>':
                    operator = '>'
                elif operator == '=':
                    operator = '='
                return f"{op}({tree[1]}) {operator} {tree[3]}"
            elif op == 'assert':
                return f"{self.tree_to_string(tree[1])} {tree[2]} {self.tree_to_string(tree[3])}"
            elif op in ['Sum', 'Avg']:
                return f"{op}({tree[1]})"
            elif op in ['¬', 'X', 'F', 'G']:
                inner = self.tree_to_string(tree[1])
                return f"{op}({inner})"
            elif op in ['∧', '∨', '→', '↔', 'U', 'R']:
                left = self.tree_to_string(tree[1])
                right = self.tree_to_string(tree[2])
                return f"({left} {op} {right})"
            elif op == 'paren':
                inner = self.tree_to_string(tree[1])
                return f"({inner})"
            else:
                parts = [self.tree_to_string(child) for child in tree[1:]]
                return f"({op} {' '.join(parts)})"

        else:
            return str(tree)

    def _needs_parentheses(self, expression):
        if expression in ['true', 'false']:
            return False

        if expression.isalpha() and expression not in ['true', 'false']:
            return False

        if any(op in expression for op in [' ∧ ', ' ∨ ', ' → ', ' ↔ ', ' U ', ' R ']):
            return True

        if ' ' in expression:
            return True

        return False


    def _needs_parentheses_for_binary(self, expression, parent_op):
        if expression in ['true', 'false']:
            return False

        if not any(char in expression for char in [' ', '∧', '∨', '→', '↔', 'U', 'R']):
            return False

        precedence = {
            'U': 1, 'R': 1,
            '→': 2, '↔': 2,
            '∧': 3, '∨': 3
        }

        main_op = None
        if expression.startswith('(') and expression.endswith(')'):
            inner = expression[1:-1]
            for op in [' U ', ' R ', ' → ', ' ↔ ', ' ∧ ', ' ∨ ']:
                if op in inner:
                    main_op = op.strip()
                    break

        if main_op and main_op in precedence and parent_op in precedence:
            return precedence[main_op] < precedence[parent_op]

        return True


    def extract_limit_avg_assertions(self, tree):
        assertions = []

        def traverse(node):
            if isinstance(node, tuple):
                if node[0] in ['LimInfAvg', 'LimSupAvg']:
                    assertions.append(node)
                else:
                    for child in node[1:]:
                        if isinstance(child, tuple) or isinstance(child, (int, float, str)):
                            traverse(child)

        traverse(tree)
        return assertions


    def build_boolean_combination(self, assertions, truth_values):
        if not assertions:
            return "T"
        terms = []
        for i, (assertion, truth_value) in enumerate(zip(assertions, truth_values)):
            assertion_str = self.tree_to_string(assertion)
            if truth_value:
                terms.append(assertion_str)
            else:
                terms.append(f"¬{assertion_str}")

        if len(terms) == 1:
            return terms[0]
        else:
            return " ∧ ".join(f"({term})" for term in terms)

    def process_formula_negation(self, formula_psi):
        print("=" * 80)
        print(f"PROCESSING FORMULA ψ: {formula_psi}")
        print("=" * 80)

        print("Step 1: Parsing original formula ψ")
        tree_psi = self.parse(formula_psi)
        if tree_psi is None:
            raise ValueError("Failed to parse formula ψ")

        parsed_psi = self.tree_to_string(tree_psi)
        print(f"Parsed ψ: {parsed_psi}")

        print("\nStep 2: Negating formula to get ϕ = ¬ψ")
        tree_phi = self.negate_formula(tree_psi)
        formula_phi = self.tree_to_string(tree_phi)
        print(f"Negated formula ϕ: {formula_phi}")

        assertions = self.extract_limit_avg_assertions(tree_phi)
        print(f"\nStep 3: Found {len(assertions)} limit-average assertions in ϕ:")
        for i, assertion in enumerate(assertions):
            print(f"  θ{i + 1}: {self.tree_to_string(assertion)}")

        if not assertions:
            print("No limit-average assertions found in ϕ. Formula is already standard LTL.")
            return [("true", formula_phi)], formula_phi

        n = len(assertions)
        truth_assignments_list = list(itertools_product([True, False], repeat=n))
        print(f"\nStep 4: Generating {len(truth_assignments_list)} truth assignments")

        disjuncts = []
        print("Step 5: Building disjuncts for ϕ:")

        for truth_values in truth_assignments_list:
            ltl_formula_tree = self.replace_assertions_with_truth_values(tree_phi,
                                                                         list(zip(assertions, truth_values)))
            ltl_formula = self.tree_to_string(ltl_formula_tree)

            limit_avg_formula = self.build_boolean_combination(assertions, truth_values)

            disjunct = (limit_avg_formula, ltl_formula)
            disjuncts.append(disjunct)

            print(f"  Disjunct: {limit_avg_formula} ∧ {ltl_formula}")

        print(f"\nStep 6: Final disjunction has {len(disjuncts)} disjuncts")
        return disjuncts, formula_phi  # Always return both values

    def complete_pipeline_with_nbw(self, formula_psi):
        print("COMPLETE PIPELINE WITH NBW CONVERSION")
        print("=" * 80)

        try:
            print("DEBUG: Calling process_formula_negation...")
            disjuncts, negated_formula = self.process_formula_negation(formula_psi)
            print(f"DEBUG: Got {len(disjuncts)} disjuncts")

        except ValueError as e:
            print(f"Error in processing formula: {e}")
            return None

        if not disjuncts:
            print("No disjuncts generated")
            return None

        print("\n" + "=" * 80)
        print("NBW CONVERSION STEP")
        print("=" * 80)

        nbw_results = []
        for i, (chi, xi) in enumerate(disjuncts):
            print(f"\n--- Disjunct {i + 1} ---")
            print(f"χ (limit-average): {chi}")
            print(f"ξ (LTL): {xi}")

            print(f"DEBUG: Converting LTL to NBW for: {xi}")
            result = self.wsl_converter.ltl_to_nbw(xi)
            nbw_results.append((chi, xi, result))

            self.wsl_converter.print_automaton_details(result, xi)

        print(f"DEBUG: Returning {len(nbw_results)} NBW results")
        return nbw_results

    def replace_assertions_with_truth_values(self, tree, truth_assignments):
        if not isinstance(tree, tuple):
            if tree == 'T':
                return 'true'
            elif tree == 'F':
                return 'false'
            return tree

        op = tree[0]

        if op in ['LimInfAvg', 'LimSupAvg']:
            for assertion, truth_value in truth_assignments:
                if tree == assertion:
                    return 'true' if truth_value else 'false'
            return tree

        new_children = []
        for child in tree[1:]:
            if isinstance(child, tuple):
                new_child = self.replace_assertions_with_truth_values(child, truth_assignments)
                new_children.append(new_child)
            elif child == 'T':
                new_children.append('true')
            elif child == 'F':
                new_children.append('false')
            else:
                new_children.append(child)

        simplified_tree = (op,) + tuple(new_children)
        return self.simplify_boolean_expression(simplified_tree)

    def simplify_boolean_expression(self, tree):
        if not isinstance(tree, tuple):
            return tree

        op = tree[0]

        if op == '¬':
            child = tree[1]
            if child == 'true':
                return 'false'
            elif child == 'false':
                return 'true'
            elif isinstance(child, tuple) and child[0] == '¬':
                return self.simplify_boolean_expression(child[1])
            return tree

        if op in ['∧', '∨', '→', '↔']:
            left = tree[1]
            right = tree[2]

            left_simple = self.simplify_boolean_expression(left) if isinstance(left, tuple) else left
            right_simple = self.simplify_boolean_expression(right) if isinstance(right, tuple) else right

            if op == '∧':
                if left_simple == 'true':
                    return right_simple
                if right_simple == 'true':
                    return left_simple
                if left_simple == 'false' or right_simple == 'false':
                    return 'false'

            elif op == '∨':
                if left_simple == 'true' or right_simple == 'true':
                    return 'true'
                if left_simple == 'false':
                    return right_simple
                if right_simple == 'false':
                    return left_simple

            elif op == '→':
                if left_simple == 'true':
                    return right_simple
                if left_simple == 'false':
                    return 'true'
                if right_simple == 'true':
                    return 'true'
                if right_simple == 'false':
                    return ('¬', left_simple) if not isinstance(left_simple, str) or left_simple not in ['true',
                                                                                                         'false'] else 'true'

            elif op == '↔':
                if left_simple == 'true':
                    return right_simple
                if right_simple == 'true':
                    return left_simple
                if left_simple == 'false':
                    return ('¬', right_simple) if not isinstance(right_simple, str) or right_simple not in ['true',
                                                                                                            'false'] else 'true'
                if right_simple == 'false':
                    return ('¬', left_simple) if not isinstance(left_simple, str) or left_simple not in ['true',
                                                                                                         'false'] else 'true'

            if left_simple != left or right_simple != right:
                return (op, left_simple, right_simple)

        if op in ['X', 'F', 'G', 'U', 'R']:
            simplified_children = [self.simplify_boolean_expression(child) if isinstance(child, tuple) else child
                                   for child in tree[1:]]
            return (op,) + tuple(simplified_children)

        return tree

class WSLSpotLTLimProcessor(LTLimProcessor):
    def __init__(self, script_path_or_config=None):
        super().__init__()

        if script_path_or_config is None:
            self.config = get_config()
            self.wsl_converter = SpotConverter(self.config)

        elif isinstance(script_path_or_config, str) and not script_path_or_config.endswith('.json'):
            from config import MultiOSConfig
            self.config = MultiOSConfig()
            self.config.set_path('spot_script', script_path_or_config)
            self.wsl_converter = SpotConverter(self.config)
            print(f"Note: Using script path: {script_path_or_config}")
            print(f"      Consider using config file instead for full multi-OS support")

        elif isinstance(script_path_or_config, str) and script_path_or_config.endswith('.json'):
            self.config = get_config(script_path_or_config)
            self.wsl_converter = SpotConverter(self.config)

        else:
            self.config = script_path_or_config
            self.wsl_converter = SpotConverter(self.config)

        from core.platform_detector import PLATFORM_STR
        print(f"✓ LTL Processor initialized on {PLATFORM_STR}")

    def complete_pipeline_with_nbw(self, formula_psi):
        print("COMPLETE PIPELINE WITH NBW CONVERSION")
        print("=" * 80)

        disjuncts, negated_formula = self.process_formula_negation(formula_psi)

        if not disjuncts:
            return

        print("\n" + "=" * 80)
        print("NBW CONVERSION STEP")
        print("=" * 80)

        nbw_results = []
        for i, (chi, xi) in enumerate(disjuncts):
            print(f"\n--- Disjunct {i + 1} ---")
            print(f"χ (limit-average): {chi}")
            print(f"ξ (LTL): {xi}")

            result = self.wsl_converter.ltl_to_nbw(xi)
            nbw_results.append((chi, xi, result))

            self.wsl_converter.print_automaton_details(result, xi)

        return nbw_results

class NBWState:

    def __init__(self, name, labels=None, is_initial=False, is_accepting=False):
        self.name = name
        self.labels = labels if labels else set()
        self.is_initial = is_initial
        self.is_accepting = is_accepting

    def __repr__(self):
        return f"NBWState({self.name}, initial={self.is_initial}, accepting={self.is_accepting})"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, NBWState) and self.name == other.name


class NBW:

    def __init__(self, hoa_string=None):
        self.states = set()
        self.initial_states = set()
        self.transitions = {}
        self.accepting_states = set()

        if hoa_string:
            self._from_hoa(hoa_string)

    def _from_hoa(self, hoa_string):
        lines = hoa_string.strip().split('\n')

        states_dict = {}
        current_state = None
        aps = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('--'):
                continue

            if line.startswith('States:'):
                num_states = int(line.split(':')[1].strip())
                for i in range(num_states):
                    state_name = f"b{i}"
                    states_dict[state_name] = NBWState(state_name)
                self.states = set(states_dict.values())

            elif line.startswith('Start:'):
                init_id = line.split(':')[1].strip()
                state_name = f"b{init_id}"
                if state_name in states_dict:
                    states_dict[state_name].is_initial = True
                    self.initial_states.add(states_dict[state_name])

            elif line.startswith('AP:'):
                parts = line.split()
                num_aps = int(parts[1])
                aps = [ap.strip('"') for ap in parts[2:2 + num_aps]]

            elif line.startswith('Acceptance:'):
                pass

            elif line.startswith('State:'):
                parts = line.split()
                state_id = parts[1]
                state_name = f"b{state_id}"
                current_state = states_dict.get(state_name, NBWState(state_name))

                if '{' in line and '}' in line:
                    acc_part = line[line.find('{'):line.find('}') + 1]
                    if '0' in acc_part:
                        current_state.is_accepting = True
                        self.accepting_states.add(current_state)

            elif line.startswith('[') and ']' in line:
                if current_state is None:
                    continue

                label_part, target_part = line.split(']')
                label_str = label_part[1:].strip()
                target_id = target_part.strip()
                target_state = states_dict.get(f"b{target_id}", NBWState(f"b{target_id}"))

                symbol = self._parse_hoa_label(label_str, aps)

                key = (current_state, frozenset(symbol))
                if key not in self.transitions:
                    self.transitions[key] = set()
                self.transitions[key].add(target_state)

                self.states.add(target_state)

    def _parse_hoa_label(self, label_str, aps):
        if label_str == 't':
            return set()
        else:
            symbols = set()
            for i, char in enumerate(label_str):
                if i < len(aps) and char == '1':
                    symbols.add(aps[i])
            return symbols

    def add_state(self, state):
        self.states.add(state)
        if state.is_initial:
            self.initial_states.add(state)
        if state.is_accepting:
            self.accepting_states.add(state)

    def __repr__(self):
        return f"NBW(states={len(self.states)}, initial={len(self.initial_states)}, accepting={len(self.accepting_states)})"


class ProductAutomaton:
    def __init__(self, qks: QuantitativeKripkeStructure, buchi_automaton: NBW, propositions: Set[str]):
        self.qks = qks
        self.buchi = buchi_automaton
        self.propositions = propositions
        self.states = set()
        self.initial_states = set()
        self.transitions = {}
        self.accepting_states = set()
        self.variables = qks.V

        self._build_product()

    def _build_product(self):
        print("Building product automaton K × Aξ according to formal definition...")

        for k_state in self.qks.states:
            for b_state in self.buchi.states:
                product_state = (k_state, b_state)
                self.states.add(product_state)

                if (k_state == self.qks.init_state and
                        b_state in self.buchi.initial_states):
                    self.initial_states.add(product_state)

                if b_state.is_accepting:
                    self.accepting_states.add(product_state)

        for (k_state, b_state) in self.states:
            bool_valuation = self.qks.get_boolean_valuation(k_state)

            true_propositions = {prop for prop in self.propositions if bool_valuation.get(prop, False)}

            valid_b_transitions = []
            for (from_b, symbol), to_states in self.buchi.transitions.items():
                if from_b == b_state:
                    if symbol.issubset(true_propositions):
                        valid_b_transitions.extend(to_states)

            for (src, dst) in self.qks.edges:
                if src == k_state:
                    for b_next in valid_b_transitions:
                        product_next = (dst, b_next)
                        key = (k_state, b_state)
                        if key not in self.transitions:
                            self.transitions[key] = set()
                        self.transitions[key].add(product_next)

        print(f"Product built according to formal definition:")
        print(f"  States (S×Q): {len(self.states)}")
        print(f"  Initial states: {len(self.initial_states)}")
        print(self.initial_states)
        print(f"  Accepting states (S×α): {len(self.accepting_states)}")
        print(f"  Transitions: {sum(len(t) for t in self.transitions.values())}")

    def get_numeric_valuation(self, product_state):
        qks_state, _ = product_state
        return self.qks.get_numeric_valuation(qks_state)

    def get_boolean_valuation(self, product_state):
        qks_state, _ = product_state
        return self.qks.get_boolean_valuation(qks_state)

    def __repr__(self):
        return f"ProductAutomaton(S×Q: {len(self.states)}, initial: {len(self.initial_states)}, S×α: {len(self.accepting_states)})"

    def find_msccs(self) -> List[Set[Any]]:
        index = 0
        indices = {}
        lowlinks = {}
        stack = []
        on_stack = set()
        msccs = []

        def strongconnect(state):
            nonlocal index
            indices[state] = index
            lowlinks[state] = index
            index += 1
            stack.append(state)
            on_stack.add(state)

            if state in self.transitions:
                for successor in self.transitions[state]:
                    if successor not in indices:
                        strongconnect(successor)
                        lowlinks[state] = min(lowlinks[state], lowlinks[successor])
                    elif successor in on_stack:
                        lowlinks[state] = min(lowlinks[state], indices[successor])

            if lowlinks[state] == indices[state]:
                scc = set()
                while True:
                    successor = stack.pop()
                    on_stack.remove(successor)
                    scc.add(successor)
                    if successor == state:
                        break
                msccs.append(scc)

        visited = set()

        def dfs(state):
            if state in visited:
                return
            visited.add(state)
            if state not in indices:
                strongconnect(state)
            if state in self.transitions:
                for successor in self.transitions[state]:
                    dfs(successor)

        for init_state in self.initial_states:
            dfs(init_state)

        return msccs

    def check_fairness(self, mscc: Set[Any]) -> bool:
        return any(state in self.accepting_states for state in mscc)

    def get_reachable_fair_msccs(self) -> List[Set[Any]]:
        all_msccs = self.find_msccs()
        fair_msccs = [mscc for mscc in all_msccs if self.check_fairness(mscc)]
        return fair_msccs

    def print_mscc_analysis(self):
        print("\n" + "=" * 60)
        print("MSCC ANALYSIS")
        print("=" * 60)

        all_msccs = self.find_msccs()
        fair_msccs = self.get_reachable_fair_msccs()

        print(f"Total MSCCs found: {len(all_msccs)}")
        print(f"Fair MSCCs (M ∩ α ≠ ∅): {len(fair_msccs)}")

        for i, mscc in enumerate(all_msccs):
            is_fair = self.check_fairness(mscc)
            status = "-FAIR" if is_fair else "-NOT FAIR"
            print(f"\nMSCC {i + 1}: {status}")
            print(f"  Size: {len(mscc)} states")

            states_list = list(mscc)
            state_preview = states_list[:3]
            preview_str = ", ".join(str(s) for s in state_preview)
            if len(mscc) > 3:
                preview_str += f", ... (+{len(mscc) - 3} more)"
            print(f"  States: [{preview_str}]")

            if is_fair:
                accepting_states = mscc & self.accepting_states
                print(f"  Accepting states in MSCC: {len(accepting_states)}")

    def find_simple_cycles_in_mscc(self, mscc: Set[Any]) -> List[List[Any]]:
        cycles = []
        visited = set()

        def dfs(path, current):
            if len(path) > 1 and current == path[0]:
                normalized_cycle = self._normalize_cycle(path.copy())
                if normalized_cycle not in cycles:
                    cycles.append(normalized_cycle)
                return

            if current in visited:
                return

            visited.add(current)
            path.append(current)

            if current in self.transitions:
                for neighbor in self.transitions[current]:
                    if neighbor in mscc:
                        dfs(path, neighbor)

            path.pop()
            visited.remove(current)

        for start_node in mscc:
            dfs([], start_node)

        return cycles

    def _normalize_cycle(self, cycle: List[Any]) -> List[Any]:
        if not cycle:
            return cycle

        min_index = min(range(len(cycle)), key=lambda i: str(cycle[i]))
        return cycle[min_index:] + cycle[:min_index]

    def compute_cycle_values(self, cycle: List[Any]) -> Dict[str, float]:
        if not cycle:
            return {}

        sums = {var: Fraction(0) for var in self.variables}
        count = len(cycle)

        for state in cycle:
            kripke_state, _ = state
            numeric_vals = self.qks.get_numeric_valuation(kripke_state)
            for var in self.variables:
                sums[var] += Fraction(str(numeric_vals.get(var, 0.0)))  # Convert to exact fraction

        averages = {var: float(sums[var] / count) for var in self.variables}
        return averages

    def deduplicate_cycle_vectors(self, cycle_values: List[Dict[str, float]], tolerance: float = 1e-9) -> List[
        Dict[str, float]]:
        unique_vectors = []

        for vec in cycle_values:
            is_duplicate = False
            for existing in unique_vectors:
                if all(abs(vec[var] - existing[var]) < tolerance for var in self.variables):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_vectors.append(vec)

        print(f"    Deduplicated: {len(cycle_values)} → {len(unique_vectors)} unique vectors")
        return unique_vectors

    def compute_mscc_convex_hull(self, mscc: Set[Any]) -> Tuple[List[Dict[str, float]], List[List[float]]]:
        print(f"  Computing convex hull for MSCC with {len(mscc)} states...")

        cycles = self.find_simple_cycles_in_mscc(mscc)
        print(f"  Found {len(cycles)} simple cycles")

        if not cycles:
            return [], []

        cycle_values = []
        for i, cycle in enumerate(cycles):
            values = self.compute_cycle_values(cycle)
            cycle_values.append(values)

        unique_vectors = self.deduplicate_cycle_vectors(cycle_values)

        variable_list = sorted(self.variables)
        points = []
        for vec in unique_vectors:
            point = [vec[var] for var in variable_list]
            points.append(point)

        print(f"    Unique cycle vectors:")
        for i, vec in enumerate(unique_vectors):
            vec_str = ", ".join(f"{var}:{vec[var]:.3f}" for var in variable_list)
            print(f"      Vector {i + 1}: {{{vec_str}}}")

        full_dim_hull = None
        if len(unique_vectors) >= len(variable_list) + 1:
            try:
                points_array = np.array(points)
                full_dim_hull = ConvexHull(points_array)

                print(f"Full-Dimensional Convex Hull Analysis:")
                print(f"    Dimension: {len(variable_list)} ({', '.join(variable_list)})")
                print(f"    Vertices: {len(full_dim_hull.vertices)}")
                print(f"    Volume: {full_dim_hull.volume:.6f}")
                print(f"    Vertex points (full dimension):")
                for vertex_idx in full_dim_hull.vertices:
                    vertex_point = points_array[vertex_idx]
                    point_str = ", ".join(f"{val:.3f}" for val in vertex_point)
                    print(f"      [{point_str}]")

            except Exception as e:
                print(f"Full-dimensional convex hull failed: {e}")
                full_dim_hull = None

        if len(variable_list) >= 2:
            self._plot_2d_convex_hull(unique_vectors, variable_list[:2])

        return unique_vectors, points

    def _plot_2d_convex_hull(self, cycle_values: List[Dict[str, float]], vars_to_plot: List[str]):
        try:
            if len(cycle_values) < 3:
                return

            points = []
            for values in cycle_values:
                point = [values.get(vars_to_plot[0], 0), values.get(vars_to_plot[1], 0)]
                points.append(point)

            points_array = np.array(points)
            hull = ConvexHull(points_array)

            print(f"2D Visualization ({vars_to_plot[0]} vs {vars_to_plot[1]}):")
            print(f"    Vertices: {len(hull.vertices)}")
            print(f"    Area: {hull.volume:.6f}")

        except Exception as e:
            print(f"2D convex hull visualization skipped: {e}")

    def analyze_mscc_value_region(self, mscc: Set[Any]) -> Dict[str, Any]:
        print(f"\nVALUE REGION ANALYSIS FOR MSCC")
        print(f"   MSCC size: {len(mscc)} states")

        unique_vectors, hull_points = self.compute_mscc_convex_hull(mscc)

        if not unique_vectors:
            return {
                'cycles_found': 0,
                'unique_vectors': [],
                'convex_hull_points': [],
                'variables': sorted(self.variables),
                'dimension': 0
            }

        variable_list = sorted(self.variables)
        value_ranges = {}
        for var in variable_list:
            values = [v[var] for v in unique_vectors]
            value_ranges[var] = {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values)
            }

        result = {
            'cycles_found': len(unique_vectors),
            'unique_vectors': unique_vectors,
            'convex_hull_points': hull_points,
            'value_ranges': value_ranges,
            'variables': variable_list,
            'dimension': len(variable_list),
            'full_dimensional_hull': len(hull_points) >= len(variable_list) + 1
        }

        print(f"   Value ranges (full dimension):")
        for var in variable_list:
            ranges = value_ranges[var]
            print(f"     {var}: [{ranges['min']:.6f}, {ranges['max']:.6f}] (avg: {ranges['avg']:.6f})")

        print(f"   Convex hull computed in {len(variable_list)}D: [{', '.join(variable_list)}]")

        return result


class EnhancedLTLimProcessor(WSLSpotLTLimProcessor):

    def __init__(self, config_path: str = None, qks: QuantitativeKripkeStructure = None):
        from core.config import get_config
        from core.z3_solver import Z3Solver

        config = get_config(config_path)

        super().__init__(config)

        self.qks = qks or self._create_example_qks()

        self.z3_solver = Z3Solver(config)

        from core.platform_detector import PLATFORM_STR, TOOL_STRATEGY
        print(f"✓ Enhanced processor initialized on {PLATFORM_STR}")
        print(f"✓ Tool strategy: {TOOL_STRATEGY}")

    def _create_example_qks(self):
        states = {'s0', 's1', 's2', 's3'}
        init_state = 's0'
        edges = {
            ('s0', 's1'), ('s0', 's2'),
            ('s1', 's0'), ('s1', 's2'), ('s1', 's3'),
            ('s2', 's1'), ('s2', 's3'),
            ('s3', 's0'), ('s3', 's2')
        }
        boolean_vars = {'p', 'q', 'r'}
        logical_formulas = {
            's0': {'p'},
            's1': {'q'},
            's2': {'p', 'r'},
            's3': {'q', 'r'}
        }
        numeric_values = {
            's0': {'x': 1.0, 'y': 2.0, 'z': 0.5},
            's1': {'x': 3.0, 'y': 1.0, 'z': 1.5},
            's2': {'x': 2.0, 'y': 3.0, 'z': 0.8},
            's3': {'x': 4.0, 'y': 0.5, 'z': 2.0}
        }

        return QuantitativeKripkeStructure(
            states, init_state, edges, boolean_vars, logical_formulas, numeric_values
        )

    def build_product_for_disjunct(self, chi, xi, nbw_result):
        print(f"  DEBUG: Starting product build for ξ='{xi}'")

        if not nbw_result or not nbw_result.get('success', False):
            print(f"- Cannot build product - NBW conversion failed for: {xi}")
            return None

        try:
            hoa_string = nbw_result['hoa_format']
            print(f"  DEBUG: HOA string length: {len(hoa_string)}")

            buchi_automaton = NBW(hoa_string)
            print(f"  DEBUG: NBW created: {buchi_automaton}")

            print(f"  DEBUG: About to create ProductAutomaton...")
            print(f"  DEBUG: ProductAutomaton type: {type(ProductAutomaton)}")

            product = ProductAutomaton(
                qks=self.qks,
                buchi_automaton=buchi_automaton,
                propositions=self.qks.boolean_vars
            )

            print(f"- Product K × Aξ: {product}")
            return product

        except Exception as e:
            print(f"- Error building product: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _check_fair_computations(self, product: ProductAutomaton, chi: str):
        print(f"-Checking for fair computations satisfying: {chi}")

        product.print_mscc_analysis()
        fair_msccs = product.get_reachable_fair_msccs()

        if not fair_msccs:
            print(f"-Phase 1-2-3 Result: No fair MSCCs found")
            return {
                'fair_msccs_exist': False,
                'fair_mscc_count': 0,
                'limit_avg_satisfiable': False,
                'value_regions': [],
                'chi': chi,
                'phase3_result': 'NO_FAIR_MSCCS'
            }

        print(f"-Phase 1: Found {len(fair_msccs)} fair MSCC(s)")

        value_regions = []
        for i, mscc in enumerate(fair_msccs):
            print(f"\n-Phase 2: Analyzing value region for fair MSCC {i + 1}")
            value_region = product.analyze_mscc_value_region(mscc)
            value_regions.append(value_region)

        print(f"-Phase 2 Result: Value regions computed for {len(fair_msccs)} fair MSCCs")
        print(f"\n-Phase 3: Z3 LP Checking for χ='{chi}'")

        phase3_results = []
        limit_avg_satisfiable = False

        for i, value_region in enumerate(value_regions):
            if not value_region['unique_vectors']:
                print(f"-Phase 3 MSCC {i + 1}: No cycle vectors - skipping")
                phase3_results.append({'feasible': False, 'error': 'No cycle vectors'})
                continue

            print(
                f"-Phase 3 MSCC {i + 1}: Checking A(χ) ∩ conv(M) with {len(value_region['unique_vectors'])} cycle vectors")

            lp_result = self.z3_solver.check_feasibility(
                cycle_vectors=value_region['unique_vectors'],
                variables=value_region['variables'],
                limit_avg_formula=chi
            )

            if lp_result.get('success', False):
                if lp_result.get('feasible', False):
                    print(f"-Phase 3 MSCC {i + 1}: A(χ) ∩ conv(M) ≠ ∅ - SATISFIABLE!")
                    print(f"-Achievable point: {lp_result.get('result_point', {})}")
                    limit_avg_satisfiable = True
                    phase3_results.append({
                        'feasible': True,
                        'result_point': lp_result.get('result_point', {}),
                        'weights': lp_result.get('weights', [])
                    })
                else:
                    print(f"-Phase 3 MSCC {i + 1}: A(χ) ∩ conv(M) = ∅ - NOT SATISFIABLE")
                    phase3_results.append({'feasible': False, 'reason': 'LP infeasible'})
            else:
                print(f"-Phase 3 MSCC {i + 1}: Z3 solver error - {lp_result.get('error', 'Unknown error')}")
                phase3_results.append({'feasible': False, 'error': lp_result.get('error', 'Unknown error')})

        if limit_avg_satisfiable:
            print(f"-FINAL RESULT: Fair computation satisfying χ EXISTS!")
        else:
            print(f"-FINAL RESULT: No fair computation satisfying χ found")

        return {
            'fair_msccs_exist': True,
            'fair_mscc_count': len(fair_msccs),
            'limit_avg_satisfiable': limit_avg_satisfiable,
            'value_regions': value_regions,
            'phase3_results': phase3_results,
            'chi': chi
        }

    def complete_pipeline_with_product(self, formula_psi):
        print("COMPLETE PIPELINE WITH FORMAL PRODUCT CONSTRUCTION + SCC ANALYSIS")
        print("=" * 80)
        print(f"K = (P,V,S,sin,R,L) where:")
        print(f"  P (propositions) = {self.qks.boolean_vars}")
        print(f"  V (variables) = {self.qks.V}")
        print(f"  S (states) = {self.qks.states}")
        print(f"  sin (initial) = {self.qks.init_state}")
        print("=" * 80)

        try:
            print("DEBUG: Calling complete_pipeline_with_nbw from EnhancedLTLimProcessor...")
            nbw_results = self.complete_pipeline_with_nbw(formula_psi)

            if not nbw_results:
                print("No NBW results to process")
                return None

            print("\n" + "=" * 80)
            print("FORMAL PRODUCT CONSTRUCTION + SCC ANALYSIS")
            print("=" * 80)

            phase1_results = []
            for i, (chi, xi, nbw_result) in enumerate(nbw_results):
                print(f"\n--- Processing Disjunct {i + 1} ---")
                print(f"  χ (limit-average): {chi}")
                print(f"  ξ (LTL): {xi}")

                product = self.build_product_for_disjunct(chi, xi, nbw_result)
                if product:
                    print(f"-Formal product K × Aξ built successfully")

                    phase1_result = self._check_fair_computations(product, chi)
                    phase1_results.append((chi, xi, product, phase1_result))

                    if phase1_result['fair_msccs_exist']:
                        print(f"-Phase 1: Fair computations POSSIBLE for this disjunct!")
                    else:
                        print(f"-Phase 1: No fair computations possible for this disjunct")
                else:
                    print(f"-Product construction failed")
                    phase1_results.append((chi, xi, None, {
                        'fair_msccs_exist': False,
                        'fair_mscc_count': 0,
                        'limit_avg_check_pending': False,
                        'chi': chi
                    }))

            return phase1_results

        except Exception as e:
            print(f"-Error in complete pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_reachable_accepting_states(self, product: ProductAutomaton):
        reachable_accepting = set()
        visited = set()

        def dfs(state):
            if state in visited:
                return
            visited.add(state)

            if state in product.accepting_states:
                reachable_accepting.add(state)
            if state in product.transitions:
                for target in product.transitions[state]:
                    dfs(target)

        for init_state in product.initial_states:
            dfs(init_state)

        return reachable_accepting
