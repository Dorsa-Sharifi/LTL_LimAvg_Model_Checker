# """
# Product Automaton Construction
# """
#
# from typing import Set, Dict, List, Tuple, Any
# from .kripke_structure import KripkeStructure
#
#
# class ProductAutomaton:
#     """Product automaton for LTL model checking."""
#
#     def __init__(self):
#         self.states: Set[Tuple[str, str]] = set()
#         self.initial_states: Set[Tuple[str, str]] = set()
#         self.transitions: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
#         self.accepting_states: Set[Tuple[str, str]] = set()
#
#     def construct_product(self, kripke: KripkeStructure,
#                           automaton_states: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Construct product automaton between Kripke structure and automaton.
#
#         Args:
#             kripke: The Kripke structure
#             automaton_states: Automaton states and transitions
#
#         Returns:
#             Dictionary with product automaton information
#         """
#         print(f"🔗 Constructing Product Automaton")
#
#         product_states = 0
#         product_transitions = 0
#
#         # This is a placeholder for product construction
#         # In practice, you would implement the full product automaton algorithm
#         for kripke_state in kripke.states:
#             for auto_state in automaton_states.get('states', ['q0']):
#                 product_state = (kripke_state, auto_state)
#                 self.states.add(product_state)
#                 product_states += 1
#
#                 # Add as initial if both components are initial
#                 if (kripke_state in kripke.initial_states and
#                         auto_state in automaton_states.get('initial', ['q0'])):
#                     self.initial_states.add(product_state)
#
#         # Construct transitions (simplified)
#         for (k_from, k_to) in kripke.transitions:
#             for auto_state in automaton_states.get('states', ['q0']):
#                 from_state = (k_from, auto_state)
#                 to_state = (k_to, auto_state)
#                 if from_state in self.states and to_state in self.states:
#                     transition = (from_state, to_state)
#                     self.transitions.append(transition)
#                     product_transitions += 1
#
#         result = {
#             'states': product_states,
#             'initial_states': len(self.initial_states),
#             'transitions': product_transitions,
#             'accepting_states': len(self.accepting_states),
#             'construction_successful': True
#         }
#
#         print(f"   Product states: {product_states}")
#         print(f"   Product transitions: {product_transitions}")
#
#         return result
#
#     def find_accepting_cycles(self) -> List[List[Tuple[str, str]]]:
#         """Find accepting cycles in the product automaton."""
#         # Placeholder for cycle detection algorithm
#         cycles = []
#
#         # In practice, implement proper cycle detection (e.g., Tarjan's algorithm)
#         print(f" Found {len(cycles)} accepting cycles")
#
#         return cycles
#
#     def get_counterexample_trace(self) -> List[str]:
#         """Extract counterexample trace if formula is not satisfied."""
#         # Placeholder for counterexample extraction
#         trace = []
#
#         print(f" Counterexample trace length: {len(trace)}")
#
#         return trace
