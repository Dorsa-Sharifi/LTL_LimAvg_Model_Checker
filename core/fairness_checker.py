from .convex_hull_checker import ConvexHullChecker


class FairnessChecker:
    def __init__(self):
        self.convex_hull_checker = ConvexHullChecker()

    def is_fair_scc_geometric(self, scc, product_kripke, chi_formula=None):

        if not chi_formula:
            return True

        # Extract variable values for states in SCC
        variable_values = {}
        for state in scc:
            kripke_state, _ = state  # (kripke_state, buchi_state)
            variable_values[state] = product_kripke.kripke.get_state_variables(kripke_state)

        # Use convex hull checker
        is_satisfied, witness = self.convex_hull_checker.verify_chi_constraint(
            scc, variable_values, chi_formula
        )

        return is_satisfied
