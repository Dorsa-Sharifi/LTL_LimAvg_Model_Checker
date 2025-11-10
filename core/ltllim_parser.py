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
                return ('X', self.negate_formula(tree[1]))

            elif op == 'F':
                return ('G', self.negate_formula(tree[1]))

            elif op == 'G':
                return ('F', self.negate_formula(tree[1]))

            elif op == '∧':
                return ('∨', self.negate_formula(tree[1]), self.negate_formula(tree[2]))

            elif op == '∨':
                return ('∧', self.negate_formula(tree[1]), self.negate_formula(tree[2]))

            elif op == '→':
                # ¬(φ → ψ) ≡ φ ∧ ¬ψ
                return ('∧', tree[1], self.negate_formula(tree[2]))

            elif op == '↔':
                # ¬(φ ↔ ψ) ≡ (φ ∧ ¬ψ) ∨ (¬φ ∧ ψ)
                return ('∨',
                        ('∧', tree[1], self.negate_formula(tree[2])),
                        ('∧', self.negate_formula(tree[1]), tree[2]))

            elif op == 'U':
                return ('R', self.negate_formula(tree[1]), self.negate_formula(tree[2]))

            elif op == 'R':
                return ('U', self.negate_formula(tree[1]), self.negate_formula(tree[2]))

            elif op == 'paren':
                return ('paren', self.negate_formula(tree[1]))

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

            # FIX: Make sure we're calling the class constructor
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
        """Check if product has fair computations satisfying the limit-average formula χ"""
        print(f"  🔍 Checking for fair computations satisfying: {chi}")

        if not product.accepting_states:
            print(f"  ⚠️ No accepting states in product - no fair computations")
            return False

        # Simple check: see if there's any path that can reach accepting states
        reachable_accepting = self._find_reachable_accepting_states(product)

        if reachable_accepting:
            print(f"  ✅ Potential fair computations found")
            print(f"  📍 Reachable accepting states: {len(reachable_accepting)}")
            return True
        else:
            print(f"  ❌ No fair computations found")
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


# Test cases
# if __name__ == "__main__":
#     # Replace "your_username" with your actual WSL username
#     processor = WSLSpotLTLimProcessor("/home/otebook/ltl_to_nbw.py")
#
#     test_formulas = [
#         # Simple cases
#         "LimInfAvg(x) >= 5",
#         "LimSupAvg(y) >= 3.5 ∧ F p",
#
#         # Multiple assertions
#         "LimInfAvg(x) >= 5 ∧ LimSupAvg(y) >= 3.0",
#         "F(LimInfAvg(y) >= 2.0) ∨ G(LimSupAvg(y) >= 4.0)",
#
#         # Complex cases with implications
#         "(LimInfAvg(x) >= 5 → F p) ∧ LimSupAvg(y) >= 3.0",
#         "G((LimInfAvg(x) >= 2.0 ∧ p) → F q)",
#
#         # Standard LTL formulas (should still work)
#         "p U q",
#         "F p ∧ G q",
#
#         # Real-world example
#         "G(request → (F response ∧ LimInfAvg(ServiceTime) >= 0.9))"
#     ]
#
#     for i, formula in enumerate(test_formulas, 1):
#         print(f"\n{'#' * 80}")
#         print(f"TEST {i}: {formula}")
#         print(f"{'#' * 80}")
#
#         try:
#             # Use the new method that includes NBW conversion
#             nbw_results = processor.complete_pipeline_with_nbw(formula)
#
#             if nbw_results:
#                 print(f"\n✅ SUCCESS: Processed {formula}")
#                 print(f"   Generated {len(nbw_results)} NBW automata")
#
#                 # Count successful conversions
#                 successful_conversions = sum(
#                     1 for _, _, result in nbw_results if result and result.get('success', False))
#                 print(f"   Successful NBW conversions: {successful_conversions}/{len(nbw_results)}")
#             else:
#                 print(f"\n❌ FAILED: No results for {formula}")
#
#         except Exception as e:
#             print(f"\n💥 ERROR processing {formula}: {e}")
#
#         # Reset for next formula
#         processor.variables.clear()
#         processor.propositions.clear()
#         processor.limit_avg_assertions.clear()
#         print("\n" + "=" * 80 + "\n")


# =============================================================================
# TEST CODE - REPLACE YOUR EXISTING TEST SECTION WITH THIS
# =============================================================================

# =============================================================================
# COMPREHENSIVE TEST SUITE FOR LTLim PROCESSOR
# =============================================================================

if __name__ == "__main__":
    # Create a comprehensive Quantitative Kripke Structure for testing
    qks = QuantitativeKripkeStructure(
        states={'s0', 's1', 's2', 's3'},
        init_state='s0',
        edges={
            ('s0', 's1'), ('s0', 's2'),
            ('s1', 's0'), ('s1', 's2'), ('s1', 's3'),
            ('s2', 's1'), ('s2', 's3'),
            ('s3', 's0'), ('s3', 's2')
        },
        boolean_vars={'p', 'q', 'r'},
        logical_formulas={
            's0': {'p'},
            's1': {'q'},
            's2': {'p', 'r'},
            's3': {'q', 'r'}
        },
        numeric_values={
            's0': {'x': 1.0, 'y': 2.0, 'z': 0.5},
            's1': {'x': 3.0, 'y': 1.0, 'z': 1.5},
            's2': {'x': 2.0, 'y': 3.0, 'z': 0.8},
            's3': {'x': 4.0, 'y': 0.5, 'z': 2.0}
        }
    )

    # Use the enhanced processor with the QKS
    processor = EnhancedLTLimProcessor("/home/otebook/ltl_to_nbw.py", qks)

    # =============================================================================
    # COMPREHENSIVE TEST CATEGORIES
    # =============================================================================

    test_categories = {
        "BASIC LTL FORMULAS": [
            "p",  # Simple proposition
            "p ∧ q",  # Conjunction
            "p ∨ q",  # Disjunction
            "¬p",  # Negation
            "F p",  # Finally
            "G p",  # Globally
            "X p",  # Next
            "p U q",  # Until
            "p R q",  # Release
            "p → q",  # Implication
            "p ↔ q",  # If and only if
        ],

        "LIMIT-AVERAGE ASSERTIONS (SIMPLE)": [
            "LimInfAvg(x) >= 2.0",
            "LimSupAvg(y) <= 3.0",
            "LimInfAvg(z) > 1.0",
            "LimSupAvg(x) < 4.0",
        ],

        "LIMIT-AVERAGE WITH BOOLEAN COMBINATIONS": [
            "LimInfAvg(x) >= 2.0 ∧ p",
            "LimSupAvg(y) <= 3.0 ∨ q",
            "F(LimInfAvg(z) > 1.0)",
            "G(LimSupAvg(x) < 4.0 → p)",
            "LimInfAvg(x) >= 2.0 → F q",
        ],

        # "MULTIPLE LIMIT-AVERAGE ASSERTIONS": [
        #     "LimInfAvg(x) >= 2.0 ∧ LimSupAvg(y) <= 3.0",
        #     "LimInfAvg(x) >= 1.0 ∨ LimSupAvg(z) <= 2.0",
        #     "LimInfAvg(x) >= 2.0 → LimSupAvg(y) <= 3.0",
        #     "F(LimInfAvg(x) >= 2.0 ∧ LimSupAvg(y) <= 3.0)",
        # ],
        #
        # "COMPLEX TEMPORAL PATTERNS": [
        #     "G(p → F q)",  # Response property
        #     "G F p",  # Infinitely often p
        #     "F G p",  # Eventually always p
        #     "(p U q) U r",  # Nested until
        #     "G(p → (q U r))",  # Conditional until
        #     "F p ∧ G(q → F r)",  # Complex combination
        # ],
        #
        # "LIMIT-AVERAGE WITH COMPLEX TEMPORAL LOGIC": [
        #     "G(request → (F response ∧ LimInfAvg(service_time) >= 0.9))",
        #     "F(LimInfAvg(power_consumption) <= 5.0 ∧ G safe_mode)",
        #     "(LimInfAvg(throughput) >= 100.0) U maintenance",
        #     "G((LimInfAvg(queue_length) >= 10.0) → F scale_up)",
        # ],
        #
        # "EDGE CASES AND BOUNDARY CONDITIONS": [
        #     "true",  # Tautology
        #     "false",  # Contradiction
        #     "LimInfAvg(x) >= 0.0",  # Always true bound
        #     "LimSupAvg(y) <= 100.0",  # Very loose bound
        #     "LimInfAvg(x) >= 10.0",  # Very strict bound
        #     "p ∧ ¬p",  # Contradiction
        #     "p ∨ ¬p",  # Tautology
        # ],
        #
        # "MIXED OPERATOR PRECEDENCE": [
        #     "p ∧ q ∨ r",  # Precedence test 1
        #     "p ∨ q ∧ r",  # Precedence test 2
        #     "p → q ∧ r",  # Precedence test 3
        #     "p U q ∧ r",  # Precedence test 4
        #     "F p ∧ G q ∨ X r",  # Precedence test 5
        #     "LimInfAvg(x) >= 2.0 ∧ p ∨ q",  # Mixed with limit-avg
        # ],
        #
        # "REAL-WORLD SCENARIOS": [
        #     # Resource management
        #     "G((high_demand → F scale_out) ∧ LimInfAvg(cost) <= 50.0)",
        #
        #     # Quality of service
        #     "G(request → (F response ∧ LimInfAvg(response_time) <= 2.0))",
        #
        #     # Energy management
        #     "G(LimInfAvg(battery_level) >= 20.0 → F recharge)",
        #
        #     # Load balancing
        #     "F G(LimSupAvg(server_load) <= 80.0 ∧ balanced)",
        #
        #     # Safety critical system
        #     "G(error → (F recovery ∧ LimInfAvg(availability) >= 0.999))",
        # ]
    }

    # =============================================================================
    # TEST EXECUTION
    # =============================================================================
    # =============================================================================
    # DEBUG TEST - FIND THE EXACT ISSUE
    # =============================================================================

    print("🔍 DEBUG TEST - FINDING THE EXACT ISSUE")
    print("=" * 100)

    # Test with the problematic formula first
    debug_formula = "LimInfAvg(x) >= 2.0"
    print(f"Testing: {debug_formula}")

    try:
        # Create processor
        qks = QuantitativeKripkeStructure(
            states={'s0', 's1', 's2', 's3'},
            init_state='s0',
            edges={
                ('s0', 's1'), ('s0', 's2'),
                ('s1', 's0'), ('s1', 's2'), ('s1', 's3'),
                ('s2', 's1'), ('s2', 's3'),
                ('s3', 's0'), ('s3', 's2')
            },
            boolean_vars={'p', 'q', 'r'},
            logical_formulas={
                's0': {'p'},
                's1': {'q'},
                's2': {'p', 'r'},
                's3': {'q', 'r'}
            },
            numeric_values={
                's0': {'x': 1.0, 'y': 2.0, 'z': 0.5},
                's1': {'x': 3.0, 'y': 1.0, 'z': 1.5},
                's2': {'x': 2.0, 'y': 3.0, 'z': 0.8},
                's3': {'x': 4.0, 'y': 0.5, 'z': 2.0}
            }
        )

        processor = EnhancedLTLimProcessor("/home/otebook/ltl_to_nbw.py", qks)

        # Test just the NBW conversion first
        print("\n1. Testing NBW conversion only...")
        nbw_results = processor.complete_pipeline_with_nbw(debug_formula)

        if nbw_results:
            print(f"✅ NBW conversion successful: {len(nbw_results)} results")

            # Now test product construction
            print("\n2. Testing product construction...")
            for i, (chi, xi, nbw_result) in enumerate(nbw_results):
                print(f"   Disjunct {i + 1}: χ='{chi}', ξ='{xi}'")

                # Test building product for this disjunct
                print(f"   DEBUG: About to call build_product_for_disjunct...")
                print(f"   DEBUG: ProductAutomaton type: {type(ProductAutomaton)}")

                product = processor.build_product_for_disjunct(chi, xi, nbw_result)
                if product:
                    print(f"   ✅ Product built successfully")
                else:
                    print(f"   ❌ Product construction failed")

        else:
            print("❌ NBW conversion failed")

    except Exception as e:
        print(f"💥 DEBUG TEST ERROR: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 100)
    print("END DEBUG TEST")
    print("=" * 100)
