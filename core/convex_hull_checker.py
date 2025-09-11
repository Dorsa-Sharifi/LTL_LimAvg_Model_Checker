import z3
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from scipy.spatial import ConvexHull
import itertools


class ConvexHullChecker:
    """
    Geometric approach to limiting average constraint checking using convex hulls.
    """

    def __init__(self):
        self.z3_solver = z3.Solver()

    def construct_convex_hull(self, max_scc: Set, variable_values: Dict[str, Dict]) -> ConvexHull:
        """
        Construct convex hull conv(M) for a maximal SCC M.

        Args:
            max_scc: Set of states in the maximal SCC
            variable_values: Dict mapping state -> {var_name: value}

        Returns:
            ConvexHull object representing conv(M)
        """
        # Extract variable names
        if not max_scc:
            raise ValueError("Empty SCC provided")

        sample_state = next(iter(max_scc))
        var_names = list(variable_values[sample_state].keys())

        # Build points matrix: each row is a state, each column is a variable
        points = []
        for state in max_scc:
            point = [variable_values[state][var] for var in var_names]
            points.append(point)

        points = np.array(points)

        # Handle degenerate cases
        if len(points) < len(var_names) + 1:
            # Not enough points for full-dimensional hull
            # Add artificial points to make it full-dimensional
            points = self._augment_points_for_hull(points)

        try:
            hull = ConvexHull(points)
            hull.var_names = var_names  # Store variable names
            return hull
        except Exception as e:
            raise ValueError(f"Failed to construct convex hull: {e}")

    def _augment_points_for_hull(self, points: np.ndarray) -> np.ndarray:
        """Add artificial points to make convex hull construction possible."""
        n_points, n_vars = points.shape

        if n_points == 1:
            # Single point - create a small hypercube around it
            center = points[0]
            epsilon = 1e-6
            augmented = [center]

            # Add 2^n corner points of hypercube
            for i in range(2 ** n_vars):
                corner = center.copy()
                for j in range(n_vars):
                    if (i >> j) & 1:
                        corner[j] += epsilon
                    else:
                        corner[j] -= epsilon
                augmented.append(corner)

            return np.array(augmented)

        # For other degenerate cases, just return original points
        return points

    def construct_constraint_region(self, chi_formula: str, var_names: List[str]) -> z3.BoolRef:
        """
        Construct Z3 constraint region A(χ) from chi formula.

        Args:
            chi_formula: String like "x >= 5 and y <= 10"
            var_names: List of variable names

        Returns:
            Z3 boolean constraint representing A(χ)
        """
        # Create Z3 variables
        z3_vars = {name: z3.Real(name) for name in var_names}

        # Parse and convert chi formula to Z3 constraint
        constraint = self._parse_chi_to_z3(chi_formula, z3_vars)

        return constraint

    def _parse_chi_to_z3(self, formula: str, z3_vars: Dict[str, z3.ArithRef]) -> z3.BoolRef:
        """
        Parse chi formula string to Z3 constraint.
        This is a simplified parser - extend as needed.
        """
        # Replace variable names with Z3 variables
        z3_formula = formula
        for var_name, z3_var in z3_vars.items():
            z3_formula = z3_formula.replace(var_name, f"z3_vars['{var_name}']")

        # Replace operators
        z3_formula = z3_formula.replace(">=", ">=")
        z3_formula = z3_formula.replace("<=", "<=")
        z3_formula = z3_formula.replace("and", "z3.And")
        z3_formula = z3_formula.replace("or", "z3.Or")
        z3_formula = z3_formula.replace("not", "z3.Not")

        # Evaluate in context with z3_vars and z3 module
        local_context = {"z3": z3, "z3_vars": z3_vars}

        try:
            constraint = eval(z3_formula, {"__builtins__": {}}, local_context)
            return constraint
        except Exception as e:
            raise ValueError(f"Failed to parse chi formula '{formula}': {e}")

    def check_intersection(self, hull: ConvexHull, chi_constraint: z3.BoolRef,
                           var_names: List[str]) -> bool:
        """
        Check if A(χ) ∩ conv(M) = ∅ using Z3.

        Args:
            hull: Convex hull conv(M)
            chi_constraint: Z3 constraint representing A(χ)
            var_names: Variable names

        Returns:
            True if intersection is non-empty (constraint satisfied)
            False if intersection is empty (constraint violated)
        """
        # Create Z3 variables
        z3_vars = {name: z3.Real(name) for name in var_names}

        # Convert convex hull to Z3 constraints
        hull_constraints = self._hull_to_z3_constraints(hull, z3_vars, var_names)

        # Create solver instance
        solver = z3.Solver()

        # Add hull constraints: point must be inside conv(M)
        for constraint in hull_constraints:
            solver.add(constraint)

        # Add chi constraint: point must satisfy A(χ)
        solver.add(chi_constraint)

        # Check satisfiability
        result = solver.check()

        if result == z3.sat:
            # Intersection is non-empty
            model = solver.model()
            witness_point = {var: model[z3_var].as_decimal(10) for var, z3_var in z3_vars.items()}
            print(f"  Witness point: {witness_point}")
            return True
        elif result == z3.unsat:
            # Intersection is empty
            return False
        else:
            # Unknown result
            raise RuntimeError("Z3 solver returned unknown result")

    def _hull_to_z3_constraints(self, hull: ConvexHull, z3_vars: Dict[str, z3.ArithRef],
                                var_names: List[str]) -> List[z3.BoolRef]:
        """
        Convert convex hull to Z3 linear constraints.
        Each facet of the hull becomes a linear inequality.
        """
        constraints = []

        # Each equation hull.equations[i] represents:
        # hull.equations[i][:-1] · x + hull.equations[i][-1] <= 0
        for eq in hull.equations:
            coeffs = eq[:-1]  # Coefficients for variables
            constant = eq[-1]  # Constant term

            # Build linear expression: coeffs[0]*x1 + coeffs[1]*x2 + ... + constant <= 0
            linear_expr = constant
            for i, var_name in enumerate(var_names):
                linear_expr += coeffs[i] * z3_vars[var_name]

            constraints.append(linear_expr <= 0)

        return constraints

    def verify_chi_constraint(self, max_scc: Set, variable_values: Dict[str, Dict],
                              chi_formula: str) -> Tuple[bool, Optional[Dict]]:
        """
        Main verification method: check if SCC satisfies chi constraint.

        Returns:
            (satisfied: bool, witness_point: Dict or None)
        """
        if not chi_formula or chi_formula.strip() == "None":
            return True, None

        try:
            # Step 1: Construct conv(M)
            hull = self.construct_convex_hull(max_scc, variable_values)
            var_names = hull.var_names

            # Step 2: Build A(χ)
            chi_constraint = self.construct_constraint_region(chi_formula, var_names)

            # Step 3: Check A(χ) ∩ conv(M) ≠ ∅
            is_satisfied = self.check_intersection(hull, chi_constraint, var_names)

            return is_satisfied, None

        except Exception as e:
            print(f"Error in convex hull verification: {e}")
            # Fallback to original method
            return False, None


class EnhancedChiParser:
    """Enhanced parser for more complex chi formulas."""

    @staticmethod
    def parse_to_z3(formula: str, z3_vars: Dict[str, z3.ArithRef]) -> z3.BoolRef:
        """
        More robust parser for chi formulas.
        """
        import re

        # Tokenize the formula
        tokens = re.findall(r'\w+|>=|<=|==|!=|>|<|\(|\)|and|or|not', formula)

        # Convert to Z3 expression using recursive descent parser
        # This is a simplified version - extend for full expression parsing

        def parse_expression(tokens, pos=0):
            # Implementation of recursive descent parser
            # Left as exercise - can use existing expression parsers
            pass

        # For now, use simple eval-based approach with safety checks
        safe_formula = formula
        for var_name, z3_var in z3_vars.items():
            safe_formula = re.sub(r'\b' + re.escape(var_name) + r'\b',
                                  f"z3_vars['{var_name}']", safe_formula)

        # Replace logical operators
        safe_formula = re.sub(r'\band\b', ' and ', safe_formula)
        safe_formula = re.sub(r'\bor\b', ' or ', safe_formula)
        safe_formula = re.sub(r'\bnot\b', 'z3.Not', safe_formula)
        safe_formula = re.sub(r'\band\b', 'z3.And', safe_formula)
        safe_formula = re.sub(r'\bor\b', 'z3.Or', safe_formula)

        context = {"z3": z3, "z3_vars": z3_vars}
        return eval(safe_formula, {"__builtins__": {}}, context)
