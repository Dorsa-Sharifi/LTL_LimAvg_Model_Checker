"""
LTL+LimAvg Model Checker Implementation
Main orchestration engine for the model checking workflow.
"""

from core.product_kripke_structure import ProductKripkeStructure
from core.fairness_checker import FairnessChecker
from ltl.ltl_to_buchi import SimpleLTLToBuchi


class LTLimModelChecker:
    def __init__(self):
        self.ltl_converter = SimpleLTLToBuchi()
        self.results = {}

    def check_model(self, kripke_structure, ltl_formula, chi_formula=None, verbose=False):
        """
        Main model checking function.

        Args:
            kripke_structure: The system model
            ltl_formula: LTL property to check
            chi_formula: Limiting average constraint (optional)
            verbose: Print detailed information

        Returns:
            dict: Results containing verification outcome and details
        """
        if verbose:
            print("=== LTL+LimAvg Model Checking ===")
            print(f"LTL Formula: {ltl_formula}")
            print(f"Chi Formula: {chi_formula}")
            print()

        # Step 1: Convert LTL to Büchi automaton
        if verbose:
            print("Step 1: Converting LTL to Büchi automaton...")

        buchi_automaton = self.ltl_converter.convert(ltl_formula)

        if verbose:
            print(f"  Created Büchi automaton with {len(buchi_automaton.states)} states")
            print(f"  Initial states: {buchi_automaton.initial_states}")
            print(f"  Accepting states: {buchi_automaton.accepting_states}")
            print()

        # Step 2: Construct product K × A
        if verbose:
            print("Step 2: Constructing product K × A...")

        product = ProductKripkeStructure(kripke_structure, buchi_automaton)

        if verbose:
            print(f"  Product has {len(product.states)} states")
            print(f"  Product has {len(product.get_accepting_states())} accepting states")
            print()

        # Step 3: Find SCCs and check fairness
        if verbose:
            print("Step 3: Analyzing fairness constraints...")

        fairness_checker = FairnessChecker(product)
        sccs = fairness_checker.find_sccs()

        if verbose:
            print(f"  Found {len(sccs)} strongly connected components")

        # Check each SCC for fairness
        fair_sccs = fairness_checker.get_fair_sccs(chi_formula)

        if verbose:
            print(f"  Found {len(fair_sccs)} fair SCCs")
            if chi_formula:
                print(f"  Chi formula constraint: {chi_formula}")

        # Step 4: Determine result
        model_satisfies = len(fair_sccs) > 0

        result = {
            'satisfies': model_satisfies,
            'ltl_formula': ltl_formula,
            'chi_formula': chi_formula,
            'product_states': len(product.states),
            'total_sccs': len(sccs),
            'fair_sccs': len(fair_sccs),
            'buchi_states': len(buchi_automaton.states),
            'kripke_states': len(kripke_structure.states),
            'fair_scc_details': []
        }

        # Add detailed information about fair SCCs
        for scc_index, scc in fair_sccs:
            scc_info = {
                'index': scc_index,
                'size': len(scc),
                'states': list(scc)[:5],  # Show first 5 states
                'has_accepting': any(product.is_accepting_state(s) for s in scc)
            }

            # Check chi formula details for this SCC
            if chi_formula:
                is_fair, reason = fairness_checker.is_fair_scc(scc, chi_formula)
                scc_info['chi_result'] = reason

            result['fair_scc_details'].append(scc_info)

        # Step 5: Generate counterexample/witness if needed
        if model_satisfies:
            # Generate witness path
            witness = self.get_witness_path(fairness_checker, fair_sccs[0][0])
            result['witness'] = witness
            if verbose and witness:
                print(f"  Generated witness path with prefix length {len(witness.get('prefix', []))}")
                print(f"  and cycle length {len(witness.get('cycle', []))}")
        else:
            result['counterexample'] = "No fair accepting cycles found"
            if verbose:
                print("  No fair accepting cycles exist - property violated")

        if verbose:
            print()
            print(f"Result: Model {'SATISFIES' if model_satisfies else 'VIOLATES'} the specification")
            print()

        self.results = result
        return result

    def get_witness_path(self, fairness_checker, fair_scc_index):
        """Generate a witness path for a fair SCC."""
        try:
            path_info = fairness_checker.construct_fair_path(fair_scc_index)
            if path_info:
                return {
                    'type': 'witness',
                    'prefix': path_info['prefix'],
                    'cycle': path_info['cycle'],
                    'description': f"Witness path: {len(path_info['prefix'])} prefix + {len(path_info['cycle'])} cycle states"
                }
        except Exception as e:
            return {
                'type': 'error',
                'message': f"Could not generate witness path: {str(e)}"
            }
        return None

    def get_counterexample(self, kripke_structure, ltl_formula, chi_formula=None):
        """
        Get a counterexample when the model doesn't satisfy the specification.
        For LTL+LimAvg, this means no fair paths exist.
        """
        result = self.check_model(kripke_structure, ltl_formula, chi_formula)

        if result['satisfies']:
            return {
                'type': 'no_counterexample',
                'message': 'Model satisfies the specification'
            }

        return {
            'type': 'counterexample',
            'message': 'No fair paths satisfy both LTL and limiting average constraints',
            'details': {
                'total_sccs': result['total_sccs'],
                'fair_sccs': result['fair_sccs'],
                'reason': 'Either no accepting cycles exist or limiting average constraints are violated'
            }
        }

    def print_detailed_results(self):
        """Print detailed results of the last model checking run."""
        if not self.results:
            print("No results available. Run check_model() first.")
            return

        result = self.results

        print("=" * 60)
        print("DETAILED MODEL CHECKING RESULTS")
        print("=" * 60)
        print(f"LTL Formula: {result['ltl_formula']}")
        print(f"Chi Formula: {result['chi_formula'] or 'None'}")
        print(f"Result: {'✓ SATISFIES' if result['satisfies'] else '✗ VIOLATES'}")
        print()

        print("Statistics:")
        print(f"  Kripke Structure States: {result['kripke_states']}")
        print(f"  Büchi Automaton States: {result['buchi_states']}")
        print(f"  Product Automaton States: {result['product_states']}")
        print(f"  Total SCCs Found: {result['total_sccs']}")
        print(f"  Fair SCCs: {result['fair_sccs']}")
        print()

        if result['fair_scc_details']:
            print("Fair SCC Details:")
            for scc_info in result['fair_scc_details']:
                print(f"  SCC {scc_info['index']}:")
                print(f"    Size: {scc_info['size']} states")
                print(f"    Has accepting states: {scc_info['has_accepting']}")
                if 'chi_result' in scc_info:
                    print(f"    Chi constraint: {scc_info['chi_result']}")
                print(f"    Sample states: {scc_info['states']}")
                print()

        if 'witness' in result and result['witness']:
            witness = result['witness']
            print("Witness Path:")
            print(f"  {witness.get('description', 'Available')}")
            if 'prefix' in witness:
                print(f"  Prefix: {witness['prefix'][:3]}{'...' if len(witness['prefix']) > 3 else ''}")
            if 'cycle' in witness:
                print(f"  Cycle: {witness['cycle'][:3]}{'...' if len(witness['cycle']) > 3 else ''}")

        print("=" * 60)
