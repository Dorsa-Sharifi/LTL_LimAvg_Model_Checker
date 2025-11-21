#!/usr/bin/env python3

import ply.lex as lex
import ply.yacc as yacc
from itertools import product
import subprocess
import json
from typing import Set, List, Tuple, Dict, Optional

from core.QuantitativeKripkeStructure import QuantitativeKripkeStructure


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
            # Instead of creating a paren node, just return the inner formula
            # This avoids unnecessary parentheses in the tree
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
                # Negate path assertion: ¬(LimXAvg(v) ≥ c) ≡ LimXAvg(v) < c
                # But we need to return the same structure as the parser expects
                old_op = tree[2]  # This is the comparison operator token
                neg_ops = {'>=': '<', '<=': '>', '>': '<=', '<': '>=', '=': '!='}
                new_op = neg_ops.get(old_op, f'¬{old_op}')

                # Return in the same format: (type, variable, operator, value)
                return (tree[0], tree[1], new_op, tree[3])

            elif op == 'assert':
                # Negate comparison operators
                old_op = tree[2]
                neg_ops = {'>=': '<', '<=': '>', '>': '<=', '<': '>=', '=': '!='}
                new_op = neg_ops.get(old_op, f'¬{old_op}')
                return ('assert', tree[1], new_op, tree[3])

            elif op == '¬':
                return tree[1]  # Double negation

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
                # ¬(φ → ψ) ≡ φ ∧ ¬ψ
                return '∧', tree[1], self.negate_formula(tree[2])

            elif op == '↔':
                # ¬(φ ↔ ψ) ≡ (φ ∧ ¬ψ) ∨ (¬φ ∧ ψ)
                return ('∨',
                        ('∧', tree[1], self.negate_formula(tree[2])),
                        ('∧', self.negate_formula(tree[1]), tree[2]))

            elif op == 'U':
                return 'R', self.negate_formula(tree[1]), self.negate_formula(tree[2])

            elif op == 'R':
                return 'U', self.negate_formula(tree[1]), self.negate_formula(tree[2])

            elif op == 'paren':
                return 'paren', self.negate_formula(tree[1])

            else:  # Arithmetic operators
                return tree

        return tree

    def tree_to_string(self, tree):
        """Convert parse tree to string - always safe parentheses - FIXED"""
        if tree is None:
            return ""

        # Handle truth values
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
                # FIX: Handle the negated operators properly
                operator = tree[2]
                # Convert internal representation back to proper syntax
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
        """Determine if an expression needs parentheses"""
        # Never put parentheses around truth values
        if expression in ['true', 'false']:
            return False

        # Never put parentheses around simple propositions
        if expression.isalpha() and expression not in ['true', 'false']:
            return False

        # Put parentheses around binary operations
        if any(op in expression for op in [' ∧ ', ' ∨ ', ' → ', ' ↔ ', ' U ', ' R ']):
            return True

        # Put parentheses around expressions with spaces (complex expressions)
        if ' ' in expression:
            return True

        return False


    def _needs_parentheses_for_binary(self, expression, parent_op):
        """Check if a binary expression needs parentheses given the parent operator"""
        if expression in ['true', 'false']:
            return False

        # If it's a simple atom, no parentheses needed
        if not any(char in expression for char in [' ', '∧', '∨', '→', '↔', 'U', 'R']):
            return False

        # Operator precedence rules
        precedence = {
            'U': 1, 'R': 1,
            '→': 2, '↔': 2,
            '∧': 3, '∨': 3
        }

        # Extract the main operator from the expression if it's binary
        main_op = None
        if expression.startswith('(') and expression.endswith(')'):
            inner = expression[1:-1]
            # Try to find the main operator
            for op in [' U ', ' R ', ' → ', ' ↔ ', ' ∧ ', ' ∨ ']:
                if op in inner:
                    main_op = op.strip()
                    break

        if main_op and main_op in precedence and parent_op in precedence:
            return precedence[main_op] < precedence[parent_op]

        return True


    def extract_limit_avg_assertions(self, tree):
        """Extract all limit-average assertions from the parse tree"""
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
        """Build the Boolean combination of limit-average assertions"""
        if not assertions:
            return "T"  # No assertions, always true

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
        """Complete pipeline: Given formula ψ, process ¬ψ and detach limit-average assertions"""

        print("=" * 80)
        print(f"PROCESSING FORMULA ψ: {formula_psi}")
        print("=" * 80)

        # Step 1: Parse the original formula ψ
        print("Step 1: Parsing original formula ψ")
        tree_psi = self.parse(formula_psi)
        if tree_psi is None:
            raise ValueError("Failed to parse formula ψ")

        parsed_psi = self.tree_to_string(tree_psi)
        print(f"Parsed ψ: {parsed_psi}")

        # Step 2: Negate to get ϕ = ¬ψ
        print("\nStep 2: Negating formula to get ϕ = ¬ψ")
        tree_phi = self.negate_formula(tree_psi)
        formula_phi = self.tree_to_string(tree_phi)
        print(f"Negated formula ϕ: {formula_phi}")

        # Step 3: Extract limit-average assertions from ϕ
        assertions = self.extract_limit_avg_assertions(tree_phi)
        print(f"\nStep 3: Found {len(assertions)} limit-average assertions in ϕ:")
        for i, assertion in enumerate(assertions):
            print(f"  θ{i + 1}: {self.tree_to_string(assertion)}")

        if not assertions:
            print("No limit-average assertions found in ϕ. Formula is already standard LTL.")
            # Return both disjuncts AND the negated formula
            return [("true", formula_phi)], formula_phi  # Single disjunct with no limit-average part

        # Step 4: Generate all possible truth assignments (2^n combinations)
        n = len(assertions)
        truth_assignments_list = list(product([True, False], repeat=n))
        print(f"\nStep 4: Generating {len(truth_assignments_list)} truth assignments")

        # Step 5: Build the disjunction for ϕ
        disjuncts = []
        print("Step 5: Building disjuncts for ϕ:")

        for truth_values in truth_assignments_list:
            # Build the LTL formula with assertions replaced by truth values
            ltl_formula_tree = self.replace_assertions_with_truth_values(tree_phi,
                                                                         list(zip(assertions, truth_values)))
            ltl_formula = self.tree_to_string(ltl_formula_tree)

            # Build the Boolean combination of limit-average assertions
            limit_avg_formula = self.build_boolean_combination(assertions, truth_values)

            disjunct = (limit_avg_formula, ltl_formula)
            disjuncts.append(disjunct)

            print(f"  Disjunct: {limit_avg_formula} ∧ {ltl_formula}")

        # Step 6: Return the final disjunction
        print(f"\nStep 6: Final disjunction has {len(disjuncts)} disjuncts")
        return disjuncts, formula_phi  # Always return both values

    def complete_pipeline_with_nbw(self, formula_psi):
        """Complete pipeline with NBW conversion - DEBUG VERSION"""
        print("COMPLETE PIPELINE WITH NBW CONVERSION")
        print("=" * 80)

        # Steps 1-6: Process formula and detach limit-average assertions
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

        # Convert each LTL formula ξ to NBW using WSL Spot
        nbw_results = []
        for i, (chi, xi) in enumerate(disjuncts):
            print(f"\n--- Disjunct {i + 1} ---")
            print(f"χ (limit-average): {chi}")
            print(f"ξ (LTL): {xi}")

            # Convert LTL to NBW
            print(f"DEBUG: Converting LTL to NBW for: {xi}")
            result = self.wsl_converter.ltl_to_nbw(xi)
            nbw_results.append((chi, xi, result))

            self.wsl_converter.print_automaton_details(result, xi)

        print(f"DEBUG: Returning {len(nbw_results)} NBW results")
        return nbw_results

    def replace_assertions_with_truth_values(self, tree, truth_assignments):
        """Replace limit-average assertions with truth values based on assignments WITH SIMPLIFICATION"""
        if not isinstance(tree, tuple):
            # Convert old format to new format
            if tree == 'T':
                return 'true'
            elif tree == 'F':
                return 'false'
            return tree

        op = tree[0]

        # If this is a limit-average assertion, replace with truth value
        if op in ['LimInfAvg', 'LimSupAvg']:
            for assertion, truth_value in truth_assignments:
                if tree == assertion:
                    return 'true' if truth_value else 'false'
            return tree  # Should not happen

        # Recursively process children FIRST
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

        # ✅ NEW: Simplify Boolean expressions after replacement
        simplified_tree = (op,) + tuple(new_children)
        return self.simplify_boolean_expression(simplified_tree)

    def simplify_boolean_expression(self, tree):
        """Simplify Boolean expressions after truth value substitution"""
        if not isinstance(tree, tuple):
            return tree

        op = tree[0]

        # Handle unary operators
        if op == '¬':
            child = tree[1]
            if child == 'true':
                return 'false'
            elif child == 'false':
                return 'true'
            elif isinstance(child, tuple) and child[0] == '¬':
                return self.simplify_boolean_expression(child[1])  # Double negation
            return tree

        # Handle binary operators
        if op in ['∧', '∨', '→', '↔']:
            left = tree[1]
            right = tree[2]

            # Simplify children first
            left_simple = self.simplify_boolean_expression(left) if isinstance(left, tuple) else left
            right_simple = self.simplify_boolean_expression(right) if isinstance(right, tuple) else right

            # Boolean simplification rules
            if op == '∧':
                # true ∧ φ ≡ φ
                if left_simple == 'true':
                    return right_simple
                # φ ∧ true ≡ φ
                if right_simple == 'true':
                    return left_simple
                # false ∧ φ ≡ false
                if left_simple == 'false' or right_simple == 'false':
                    return 'false'

            elif op == '∨':
                # true ∨ φ ≡ true
                if left_simple == 'true' or right_simple == 'true':
                    return 'true'
                # false ∨ φ ≡ φ
                if left_simple == 'false':
                    return right_simple
                # φ ∨ false ≡ φ
                if right_simple == 'false':
                    return left_simple

            elif op == '→':
                # true → φ ≡ φ
                if left_simple == 'true':
                    return right_simple
                # false → φ ≡ true
                if left_simple == 'false':
                    return 'true'
                # φ → true ≡ true
                if right_simple == 'true':
                    return 'true'
                # φ → false ≡ ¬φ
                if right_simple == 'false':
                    return ('¬', left_simple) if not isinstance(left_simple, str) or left_simple not in ['true',
                                                                                                         'false'] else 'true'

            elif op == '↔':
                # true ↔ φ ≡ φ
                if left_simple == 'true':
                    return right_simple
                # φ ↔ true ≡ φ
                if right_simple == 'true':
                    return left_simple
                # false ↔ φ ≡ ¬φ
                if left_simple == 'false':
                    return ('¬', right_simple) if not isinstance(right_simple, str) or right_simple not in ['true',
                                                                                                            'false'] else 'true'
                # φ ↔ false ≡ ¬φ
                if right_simple == 'false':
                    return ('¬', left_simple) if not isinstance(left_simple, str) or left_simple not in ['true',
                                                                                                         'false'] else 'true'

            # If no simplification applied, return the simplified children
            if left_simple != left or right_simple != right:
                return (op, left_simple, right_simple)

        # For temporal operators, just simplify children but keep structure
        if op in ['X', 'F', 'G', 'U', 'R']:
            simplified_children = [self.simplify_boolean_expression(child) if isinstance(child, tuple) else child
                                   for child in tree[1:]]
            return (op,) + tuple(simplified_children)

        return tree
class WSLSpotConverter:
    """Uses Spot installed on WSL from Windows"""

    def __init__(self, wsl_script_path="/home/otebook/ltl_to_nbw.py"):
        # Replace 'username' with your actual WSL username
        self.wsl_script_path = wsl_script_path
        self._test_wsl_connection()

    def _test_wsl_connection(self):
        """Test if WSL is accessible"""
        try:
            result = subprocess.run('wsl echo "WSL connected"',
                                    shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ WSL connection successful")
            else:
                print("✗ WSL connection failed")
        except Exception as e:
            print(f"✗ WSL test failed: {e}")

    def ltl_to_nbw(self, ltl_formula):
        """Call WSL Spot script to convert LTL to NBW"""
        try:
            # Escape the formula for command line
            escaped_formula = ltl_formula.replace('"', '\\"')

            # Build WSL command
            cmd = f'wsl python3 {self.wsl_script_path} "{escaped_formula}"'
            print(f"Executing: {cmd}")

            # Execute command
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {'success': False, 'error': result.stderr}

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Conversion timeout'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def print_automaton_details(self, result, formula):
        """Print automaton details"""
        if not result.get('success', False):
            print(f"❌ Failed to convert: {formula}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return

        print(f"\n{'=' * 60}")
        print(f"✅ LTL Formula: {formula}")
        if 'formula_used' in result:
            print(f"   (Converted to: {result['formula_used']})")
        print(f"{'=' * 60}")
        print(f"📊 States: {result['states']}")
        print(f"📊 Edges: {result['edges']}")
        print(f"✅ Acceptance: {result['acceptance']}")
        print(f"🔍 Deterministic: {result['is_deterministic']}")

        # Save HOA to file
        if 'hoa_format' in result:
            safe_name = formula.replace(' ', '_').replace('&', 'and').replace('|', 'or')
            filename = f"automaton_{safe_name}.hoa"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(result['hoa_format'])
            print(f"💾 HOA format saved to: {filename}")


class WSLSpotLTLimProcessor(LTLimProcessor):
    """Enhanced processor using WSL Spot for NBW conversion"""

    def __init__(self, wsl_script_path="/home/otebook/ltl_to_nbw.py"):
        super().__init__()
        self.wsl_converter = WSLSpotConverter(wsl_script_path)

    def complete_pipeline_with_nbw(self, formula_psi):
        """Complete pipeline with NBW conversion"""
        print("COMPLETE PIPELINE WITH NBW CONVERSION")
        print("=" * 80)

        # Your existing steps 1-6
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

            # Convert LTL to NBW
            result = self.wsl_converter.ltl_to_nbw(xi)
            nbw_results.append((chi, xi, result))

            self.wsl_converter.print_automaton_details(result, xi)

        return nbw_results


# =============================================================================
# PRODUCT CONSTRUCTION CLASSES - ADD THESE TO YOUR EXISTING FILE
# =============================================================================

class NBWState:
    """Represents a state in the NBW"""

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
    """Nondeterministic Büchi Automaton"""

    def __init__(self, hoa_string=None):
        self.states = set()
        self.initial_states = set()
        self.transitions = {}  # dict: (state, symbol) -> set of next states
        self.accepting_states = set()

        if hoa_string:
            self._from_hoa(hoa_string)

    def _from_hoa(self, hoa_string):
        """Initialize from HOA format string"""
        lines = hoa_string.strip().split('\n')

        states_dict = {}
        current_state = None
        aps = []  # Atomic propositions

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
                # Parse atomic propositions: AP: 2 "p" "q"
                parts = line.split()
                num_aps = int(parts[1])
                aps = [ap.strip('"') for ap in parts[2:2 + num_aps]]

            elif line.startswith('Acceptance:'):
                # We assume Büchi acceptance
                pass

            elif line.startswith('State:'):
                parts = line.split()
                state_id = parts[1]
                state_name = f"b{state_id}"
                current_state = states_dict.get(state_name, NBWState(state_name))

                # Check if accepting
                if '{' in line and '}' in line:
                    acc_part = line[line.find('{'):line.find('}') + 1]
                    if '0' in acc_part:  # Büchi acceptance condition
                        current_state.is_accepting = True
                        self.accepting_states.add(current_state)

            elif line.startswith('[') and ']' in line:
                if current_state is None:
                    continue

                # Parse label and target
                label_part, target_part = line.split(']')
                label_str = label_part[1:].strip()
                target_id = target_part.strip()
                target_state = states_dict.get(f"b{target_id}", NBWState(f"b{target_id}"))

                # Convert label to set of propositions
                symbol = self._parse_hoa_label(label_str, aps)

                # Add transition
                key = (current_state, frozenset(symbol))
                if key not in self.transitions:
                    self.transitions[key] = set()
                self.transitions[key].add(target_state)

                # Ensure target state is in states
                self.states.add(target_state)

    def _parse_hoa_label(self, label_str, aps):
        """Parse HOA label string into set of propositions"""
        if label_str == 't':  # true - all propositions can be anything
            return set()
        else:
            # Simple case: [0], [1], [01], etc.
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
    """Product K × Aξ = (∅, V, S×Q, (sin,qin), R, L, S×α) as defined in the paper"""

    def __init__(self, qks: QuantitativeKripkeStructure, buchi_automaton: NBW, propositions: Set[str]):
        self.qks = qks
        self.buchi = buchi_automaton
        self.propositions = propositions  # Set P from the definition
        self.states = set()  # S × Q
        self.initial_states = set()  # (sin, qin)
        self.transitions = {}  # R: (s,q) -> set of (s',q')
        self.accepting_states = set()  # S × α
        self.variables = qks.V  # V from QKS

        self._build_product()

    def _build_product(self):
        """Build the product automaton K × Aξ according to the formal definition"""
        print("Building product automaton K × Aξ according to formal definition...")

        # Create product states: S × Q
        for k_state in self.qks.states:
            for b_state in self.buchi.states:
                product_state = (k_state, b_state)
                self.states.add(product_state)

                # Check if this is an initial state: (sin, qin)
                if (k_state == self.qks.init_state and
                        b_state in self.buchi.initial_states):
                    self.initial_states.add(product_state)

                # Check if this is an accepting state: S × α
                if b_state.is_accepting:
                    self.accepting_states.add(product_state)

        # Build transitions R according to: R(s,q, s',q') iff R(s,s') and q' ∈ δ(q, [[P]]s)
        for (k_state, b_state) in self.states:
            # Get the Boolean valuation for state s: [[P]]s
            bool_valuation = self.qks.get_boolean_valuation(k_state)

            # Extract the set of true propositions in state s: {p ∈ P | [[p]]s = true}
            true_propositions = {prop for prop in self.propositions if bool_valuation.get(prop, False)}

            # Find all valid buchi transitions: q' ∈ δ(q, [[P]]s)
            valid_b_transitions = []
            for (from_b, symbol), to_states in self.buchi.transitions.items():
                if from_b == b_state:
                    # Check if the buchi transition symbol matches the true propositions
                    if symbol.issubset(true_propositions):
                        valid_b_transitions.extend(to_states)

            # For each Kripke transition R(s, s'), combine with valid buchi transitions
            for (src, dst) in self.qks.edges:
                if src == k_state:  # This is R(s, s')
                    for b_next in valid_b_transitions:
                        product_next = (dst, b_next)
                        key = (k_state, b_state)
                        if key not in self.transitions:
                            self.transitions[key] = set()
                        self.transitions[key].add(product_next)

        print(f"Product built according to formal definition:")
        print(f"  States (S×Q): {len(self.states)}")
        print(f"  Initial states: {len(self.initial_states)}")
        # for boo in range(len(self.initial_states)):
        print(self.initial_states)
        print(f"  Accepting states (S×α): {len(self.accepting_states)}")
        print(f"  Transitions: {sum(len(t) for t in self.transitions.values())}")

    def get_numeric_valuation(self, product_state):
        """Get numeric valuation: [[v]]_(s,q) = [[v]]_s for every v ∈ V"""
        qks_state, _ = product_state
        return self.qks.get_numeric_valuation(qks_state)

    def get_boolean_valuation(self, product_state):
        """Get boolean valuation for the product state"""
        qks_state, _ = product_state
        return self.qks.get_boolean_valuation(qks_state)

    def __repr__(self):
        return f"ProductAutomaton(S×Q: {len(self.states)}, initial: {len(self.initial_states)}, S×α: {len(self.accepting_states)})"


class EnhancedLTLimProcessor(WSLSpotLTLimProcessor):
    """Enhanced processor with product construction - COMPLETELY FIXED"""

    def __init__(self, wsl_script_path, qks: QuantitativeKripkeStructure = None):
        super().__init__(wsl_script_path)
        self.qks = qks or self._create_example_qks()

    def _create_example_qks(self):
        """Create an example Quantitative Kripke structure for testing"""
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
        """Build product K × Aξ for a single disjunct - DEBUG VERSION"""
        print(f"  DEBUG: Starting product build for ξ='{xi}'")

        if not nbw_result or not nbw_result.get('success', False):
            print(f"  ❌ Cannot build product - NBW conversion failed for: {xi}")
            return None

        try:
            # Create NBW from HOA format
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

            print(f"  🔗 Product K × Aξ: {product}")
            return product

        except Exception as e:
            print(f"  💥 Error building product: {e}")
            import traceback
            traceback.print_exc()
            return None

    def complete_pipeline_with_product(self, formula_psi):
        """Complete pipeline including correct product construction - FIXED VERSION"""
        print("COMPLETE PIPELINE WITH FORMAL PRODUCT CONSTRUCTION")
        print("=" * 80)
        print(f"K = (P,V,S,sin,R,L) where:")
        print(f"  P (propositions) = {self.qks.boolean_vars}")
        print(f"  V (variables) = {self.qks.V}")
        print(f"  S (states) = {self.qks.states}")
        print(f"  sin (initial) = {self.qks.init_state}")
        print("=" * 80)

        try:
            # FIX: Use the CORRECT method from the CURRENT class
            print("DEBUG: Calling complete_pipeline_with_nbw from EnhancedLTLimProcessor...")
            nbw_results = self.complete_pipeline_with_nbw(formula_psi)

            if not nbw_results:
                print("No NBW results to process")
                return None

            print("\n" + "=" * 80)
            print("FORMAL PRODUCT CONSTRUCTION STEP: K × Aξ")
            print("=" * 80)

            product_results = []
            for i, (chi, xi, nbw_result) in enumerate(nbw_results):
                print(f"\n--- Building Formal Product for Disjunct {i + 1} ---")
                print(f"  χ (limit-average): {chi}")
                print(f"  ξ (LTL): {xi}")

                product = self.build_product_for_disjunct(chi, xi, nbw_result)
                product_results.append((chi, xi, product))

                if product:
                    print(f"  ✅ Formal product K × Aξ built successfully")
                    # Check for fair computations in the product
                    has_fair_computations = self._check_fair_computations(product, chi)
                    if has_fair_computations:
                        print(f"  🎯 Fair computations FOUND for this disjunct!")
                    else:
                        print(f"  ⚠️ No fair computations found for this disjunct")
                else:
                    print(f"  ❌ Product construction failed")

            return product_results

        except Exception as e:
            print(f"💥 Error in complete pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _check_fair_computations(self, product: ProductAutomaton, chi: str):
        """Complete implementation of Lemma 4.2 - FINAL VERSION"""
        print(f"  🔍 Checking for fair computations satisfying: {chi}")

        if not product.accepting_states:
            print(f"  ⚠️ No accepting states in product - no fair computations")
            return False

        # Step 1: Find all fair MSCCs
        fair_msccs = self.find_fair_msccs(product)

        if not fair_msccs:
            print(f"  ⚠️ No fair MSCCs found")
            return False

        print(f"  📍 Found {len(fair_msccs)} fair MSCCs")

        # Step 2: Check each fair MSCC with ComponentCheck
        for i, mscc in enumerate(fair_msccs):
            print(f"  🔎 Checking MSCC {i + 1} with {len(mscc)} states")
            if self.component_check(mscc, chi, product):
                print(f"  ✅ MSCC {i + 1} satisfies χ")
                return True

        print(f"  ❌ No fair MSCC satisfies χ")
        return False


    def _find_reachable_accepting_states(self, product: ProductAutomaton):
        """Find accepting states reachable from initial states - FIXED"""
        reachable_accepting = set()
        visited = set()

        def dfs(state):
            if state in visited:
                return
            visited.add(state)

            if state in product.accepting_states:
                reachable_accepting.add(state)

            # ✅ CORRECT: Check if state has outgoing transitions
            if state in product.transitions:
                for target in product.transitions[state]:
                    dfs(target)

        # Start DFS from all initial states
        for init_state in product.initial_states:
            dfs(init_state)

        return reachable_accepting

    def _find_all_msccs(self, product: ProductAutomaton):
        """Find all Maximally Strongly Connected Components using Tarjan's algorithm"""
        print(f"  🔍 Finding MSCCs in product with {len(product.states)} states...")

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

            # Consider successors of state
            if state in product.transitions:
                for successor in product.transitions[state]:
                    if successor not in indices:
                        # Successor has not yet been visited; recurse on it
                        strongconnect(successor)
                        lowlinks[state] = min(lowlinks[state], lowlinks[successor])
                    elif successor in on_stack:
                        # Successor is in stack and hence in the current SCC
                        lowlinks[state] = min(lowlinks[state], indices[successor])

            # If state is a root node, pop the stack and generate an SCC
            if lowlinks[state] == indices[state]:
                scc = set()
                while True:
                    successor = stack.pop()
                    on_stack.remove(successor)
                    scc.add(successor)
                    if successor == state:
                        break
                msccs.append(scc)

        # Start DFS from all unvisited states
        for state in product.states:
            if state not in indices:
                strongconnect(state)

        # Filter to non-trivial MSCCs (size > 1 or self-loop)
        non_trivial_msccs = []
        for scc in msccs:
            if len(scc) > 1:
                non_trivial_msccs.append(scc)
            else:
                # Check if single state has self-loop
                state = next(iter(scc))
                if state in product.transitions and state in product.transitions[state]:
                    non_trivial_msccs.append(scc)

        print(f"  📊 Found {len(msccs)} SCCs, {len(non_trivial_msccs)} non-trivial MSCCs")

        # Sort by size (largest first) for debugging
        non_trivial_msccs.sort(key=len, reverse=True)
        for i, mscc in enumerate(non_trivial_msccs[:3]):  # Show top 3
            print(f"    MSCC {i + 1}: {len(mscc)} states")

        return non_trivial_msccs

    def find_fair_msccs(self, product: ProductAutomaton):
        """Find all fair MSCCs (M ∩ α ≠ ∅) in product automaton"""
        all_msccs = self._find_all_msccs(product)

        fair_msccs = []
        for mscc in all_msccs:
            # Check if MSCC contains at least one accepting state
            if any(state in product.accepting_states for state in mscc):
                fair_msccs.append(mscc)

        print(f"  🎯 Found {len(fair_msccs)} fair MSCCs (with accepting states)")

        # Debug: Show accepting states in each fair MSCC
        for i, mscc in enumerate(fair_msccs):
            accepting_count = sum(1 for state in mscc if state in product.accepting_states)
            print(f"    Fair MSCC {i + 1}: {len(mscc)} states, {accepting_count} accepting")

        return fair_msccs

    def _parse_limit_avg_formula(self, chi: str):
        """Parse limit-average formula into linear constraints that define A(χ)"""
        print(f"    📐 Parsing χ to define polyhedron A(χ)...")

        # Remove outer parentheses if present
        chi = chi.strip()
        if chi.startswith('(') and chi.endswith(')'):
            chi = chi[1:-1]

        constraints = []

        # Split by AND (we handle conjunctions first)
        if ' ∧ ' in chi:
            parts = chi.split(' ∧ ')
        else:
            parts = [chi]

        for part in parts:
            part = part.strip()
            constraint = self._parse_single_assertion(part)
            if constraint:
                constraints.append(constraint)
                print(f"    📏 A(χ) constraint: {constraint['variable']} {constraint['operator']} {constraint['value']}")

        print(f"    📊 A(χ) defined by {len(constraints)} half-spaces/equations")
        return constraints

    def _parse_single_assertion(self, assertion: str):
        """Parse a single limit-average assertion into (variable, operator, value)"""
        # Examples:
        # "LimInfAvg(x) >= 2" -> ('x', 'LimInfAvg', '>=', 2.0)
        # "LimSupAvg(y) < 5" -> ('y', 'LimSupAvg', '<', 5.0)

        try:
            # Extract variable name, type, operator, and value
            import re

            # Match patterns like: LimInfAvg(x) >= 2.5
            match = re.match(r'(LimInfAvg|LimSupAvg)\((\w+)\)\s*([><=]=?|!=)\s*([\d.]+)', assertion)
            if match:
                avg_type, var_name, operator, value_str = match.groups()
                value = float(value_str)

                # Convert operators to standard forms
                if operator == '>=':
                    op = '>='
                elif operator == '<=':
                    op = '<='
                elif operator == '>':
                    op = '>'
                elif operator == '<':
                    op = '<'
                elif operator == '=' or operator == '==':
                    op = '='
                else:
                    print(f"    ⚠️ Unsupported operator: {operator}")
                    return None

                return {
                    'variable': var_name,
                    'type': avg_type,  # 'LimInfAvg' or 'LimSupAvg'
                    'operator': op,
                    'value': value
                }

            # Handle negated assertions: ¬(LimInfAvg(x) >= 2)
            match_neg = re.match(r'¬\((LimInfAvg|LimSupAvg)\((\w+)\)\s*([><=]=?|!=)\s*([\d.]+)\)', assertion)
            if match_neg:
                # For negation, flip the operator
                avg_type, var_name, operator, value_str = match_neg.groups()
                value = float(value_str)

                # Flip operators for negation
                neg_ops = {'>=': '<', '<=': '>', '>': '<=', '<': '>=', '=': '!='}
                op = neg_ops.get(operator, f'¬{operator}')

                return {
                    'variable': var_name,
                    'type': avg_type,
                    'operator': op,
                    'value': value
                }

        except Exception as e:
            print(f"    ❌ Error parsing assertion '{assertion}': {e}")

        return None

    def _build_and_solve_lp(self, mscc, constraints, product: ProductAutomaton):
        """Build and solve LP to check if MSCC can satisfy limit-average constraints"""
        print(f"    🧮 Building LP for MSCC with {len(mscc)} states...")

        try:
            # We'll use scipy for LP solving
            from scipy.optimize import linprog
            import numpy as np

            # Step 1: Extract all edges within the MSCC
            mscc_edges = self._extract_mscc_edges(mscc, product)

            if not mscc_edges:
                print(f"    ⚠️ No edges in MSCC - cannot form cycles")
                return False

            # Step 2: Set up LP variables (edge frequencies)
            edge_vars = list(mscc_edges.keys())
            num_vars = len(edge_vars)

            # Step 3: Build flow conservation constraints
            A_eq_flow, b_eq_flow = self._build_flow_constraints(mscc, edge_vars, mscc_edges)

            # Step 4: Add normalization constraint: ∑ y_e = 1
            A_eq_norm = np.ones((1, num_vars))
            b_eq_norm = np.array([1.0])

            # Combine flow conservation and normalization
            if A_eq_flow.shape[0] > 0:
                A_eq = np.vstack([A_eq_flow, A_eq_norm])
                b_eq = np.hstack([b_eq_flow, b_eq_norm])
            else:
                A_eq = A_eq_norm
                b_eq = b_eq_norm

            # Step 5: Build limit-average constraints
            A_ub, b_ub = self._build_limit_avg_constraints(edge_vars, mscc_edges, constraints, product)

            # Step 6: Objective function (minimize 0 - we just want feasibility)
            c = np.zeros(num_vars)

            # Step 7: Bounds (edge frequencies >= 0)
            bounds = [(0, None) for _ in range(num_vars)]

            # Step 8: Solve LP
            print(f"    🔧 Solving LP with {num_vars} variables, {A_eq.shape[0]} equality constraints...")
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')

            if result.success:
                print(f"    ✅ LP feasible - MSCC can satisfy limit-average constraints")

                # Debug: show some edge frequencies
                if result.x is not None:
                    non_zero_edges = [(edge_vars[i], freq) for i, freq in enumerate(result.x) if freq > 1e-6]
                    print(f"    📈 Non-zero edge frequencies: {len(non_zero_edges)}")
                    for edge, freq in non_zero_edges[:3]:  # Show first 3
                        print(f"        {edge}: {freq:.4f}")

                return True
            else:
                print(f"    ❌ LP infeasible - MSCC cannot satisfy constraints")
                if hasattr(result, 'message'):
                    print(f"    💡 Reason: {result.message}")
                return False

        except ImportError:
            print(f"    ⚠️ scipy not available - skipping LP check")
            return True  # Assume satisfiable for now
        except Exception as e:
            print(f"    💥 LP solving error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _extract_mscc_edges(self, mscc, product: ProductAutomaton):
        """Extract all edges that stay within the MSCC"""
        mscc_edges = {}

        for state in mscc:
            if state in product.transitions:
                for target in product.transitions[state]:
                    if target in mscc:  # Only edges within MSCC
                        edge = (state, target)
                        mscc_edges[edge] = {
                            'source': state,
                            'target': target,
                            'numeric_vals': product.get_numeric_valuation(state)  # Values from source state
                        }

        return mscc_edges

    def _build_flow_constraints(self, mscc, edge_vars, mscc_edges):
        """Build flow conservation constraints for the LP: ∑in y_e = ∑out y_e for each state"""
        print(f"      📐 Building flow conservation constraints for {len(mscc)} states...")

        import numpy as np

        states = list(mscc)
        num_states = len(states)
        num_edges = len(edge_vars)

        # Flow conservation: for each state, inflow = outflow
        A_eq = np.zeros((num_states, num_edges))
        b_eq = np.zeros(num_states)

        # Create state to index mapping
        state_to_idx = {state: i for i, state in enumerate(states)}

        # Build constraint matrix
        for edge_idx, edge in enumerate(edge_vars):
            src, tgt = edge

            # Outflow from source: positive coefficient
            if src in state_to_idx:
                A_eq[state_to_idx[src], edge_idx] += 1

            # Inflow to target: negative coefficient
            if tgt in state_to_idx:
                A_eq[state_to_idx[tgt], edge_idx] -= 1

        # Remove all-zero rows (states with no edges in the MSCC)
        non_zero_rows = ~np.all(A_eq == 0, axis=1)
        A_eq = A_eq[non_zero_rows]
        b_eq = b_eq[non_zero_rows]

        print(f"      📊 Flow constraints: {A_eq.shape[0]} equations, {num_edges} variables")
        return A_eq, b_eq

    def _build_limit_avg_constraints(self, edge_vars, mscc_edges, constraints, product):
        """Build limit-average constraints for the LP: ∑ y_e * w(e) ∼ value"""
        print(f"      📐 Building limit-average constraints...")

        import numpy as np

        num_edges = len(edge_vars)
        num_constraints = len(constraints)

        if num_constraints == 0:
            return np.zeros((0, num_edges)), np.zeros(0)

        A_ub = np.zeros((num_constraints, num_edges))
        b_ub = np.zeros(num_constraints)

        for const_idx, constraint in enumerate(constraints):
            var_name = constraint['variable']
            op = constraint['operator']
            target_value = constraint['value']

            # Build the sum: ∑ y_e * w_e(var_name)
            for edge_idx, edge in enumerate(edge_vars):
                # Get numeric value for this variable from the edge's source state
                source_state = edge[0]  # (k_state, b_state)
                numeric_vals = product.get_numeric_valuation(source_state)
                weight = numeric_vals.get(var_name, 0.0)

                A_ub[const_idx, edge_idx] = weight

            # Convert operator to LP form
            if op in ['>=', '>']:
                # ∑ y_e * w_e >= value  →  -∑ y_e * w_e <= -value
                A_ub[const_idx, :] = -A_ub[const_idx, :]
                b_ub[const_idx] = -target_value
            elif op in ['<=', '<']:
                # ∑ y_e * w_e <= value  →  ∑ y_e * w_e <= value
                b_ub[const_idx] = target_value
            elif op == '=':
                # For equality, we need to handle it differently
                # We'll treat it as two inequalities: <= and >=
                print(f"      ⚠️ Equality constraints not fully supported yet, using approximation")
                # Use >= for now
                A_ub[const_idx, :] = -A_ub[const_idx, :]
                b_ub[const_idx] = -target_value

        print(f"      📊 Limit-average constraints: {num_constraints} inequalities")
        return A_ub, b_ub

    def component_check(self, mscc, chi: str, product: ProductAutomaton):
        """ComponentCheck(M, χ) - Check if conv(M) ∩ A(χ) ≠ ∅ using LP"""
        print(f"    🧩 ComponentCheck: Checking conv(M) ∩ A(χ) ≠ ∅")
        print(f"    📐 conv(M) = convex hull of achievable limit-average vectors in MSCC")
        print(f"    📐 A(χ) = {{ x | x satisfies {chi} }}")

        # Parse χ into constraints that define A(χ)
        constraints = self._parse_limit_avg_formula(chi)

        if not constraints:
            print(f"    ⚠️ No valid constraints parsed - assuming satisfiable")
            return True

        # Build and solve LP to check conv(M) ∩ A(χ) ≠ ∅
        return self._check_convex_hull_intersection(mscc, constraints, product)

    def _check_convex_hull_intersection(self, mscc, constraints, product):
        """Check if conv(M) ∩ A(χ) ≠ ∅ using LP"""
        print(f"    🔍 Checking conv(M) ∩ A(χ) ≠ ∅ via LP...")

        try:
            from scipy.optimize import linprog
            import numpy as np

            # Step 1: Extract edges in MSCC (these define the cycles)
            mscc_edges = self._extract_mscc_edges(mscc, product)

            if not mscc_edges:
                print(f"    ⚠️ No edges in MSCC - conv(M) is empty")
                return False

            edge_vars = list(mscc_edges.keys())
            num_vars = len(edge_vars)

            print(f"    📊 conv(M) defined by {num_vars} edges (potential cycle segments)")

            # Step 2: conv(M) = { Σ y_e·w(e) | flow conservation, Σ y_e = 1, y_e ≥ 0 }
            A_eq_flow, b_eq_flow = self._build_flow_constraints(mscc, edge_vars, mscc_edges)
            A_eq_norm = np.ones((1, num_vars))  # Σ y_e = 1
            b_eq_norm = np.array([1.0])

            # Combine constraints for conv(M)
            if A_eq_flow.shape[0] > 0:
                A_eq = np.vstack([A_eq_flow, A_eq_norm])
                b_eq = np.hstack([b_eq_flow, b_eq_norm])
            else:
                A_eq = A_eq_norm
                b_eq = b_eq_norm

            # Step 3: A(χ) constraints
            A_ub, b_ub = self._build_limit_avg_constraints(edge_vars, mscc_edges, constraints, product)

            # Step 4: Check intersection via LP feasibility
            c = np.zeros(num_vars)  # Objective doesn't matter for feasibility
            bounds = [(0, None) for _ in range(num_vars)]

            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')

            if result.success:
                print(f"    ✅ conv(M) ∩ A(χ) ≠ ∅ - intersection FOUND")

                # Show the intersection point (the limit-average vector)
                if result.x is not None:
                    intersection_point = self._compute_limit_average_vector(result.x, edge_vars, mscc_edges, product)
                    print(f"    📍 Intersection point: {intersection_point}")

                return True
            else:
                print(f"    ❌ conv(M) ∩ A(χ) = ∅ - NO intersection")
                return False

        except Exception as e:
            print(f"    💥 Error in convex hull intersection check: {e}")
            return False

    def _compute_limit_average_vector(self, y_values, edge_vars, mscc_edges, product):
        """Compute the limit-average vector x = Σ y_e·w(e)"""
        # Get all numeric variables
        sample_state = next(iter(mscc_edges.values()))['source']
        all_vars = list(product.get_numeric_valuation(sample_state).keys())

        result = {}
        for var in all_vars:
            total = 0.0
            for i, edge in enumerate(edge_vars):
                source_state = edge[0]
                numeric_vals = product.get_numeric_valuation(source_state)
                weight = numeric_vals.get(var, 0.0)
                total += y_values[i] * weight
            result[var] = total

        return result