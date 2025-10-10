# """
# LTL+LimAvg Model Checker - Main Entry Point
# Comprehensive CLI interface with all implemented features
# """
#
# import sys
# import spot
# import logging
# from typing import Dict, Any
# from core.kripke_structure import KripkeStructure
# from core.ltlim_checker import LTLimModelChecker
# from core.fairness_checker import FairnessChecker
# from core.convex_hull_checker import ConvexHullChecker
# from core.product_automaton import ProductAutomaton, create_simple_buchi_automaton
#
#
# def create_traffic_light_system() -> KripkeStructure:
#     """Create the Traffic Light System example."""
#     ks = KripkeStructure()
#
#     # Add states
#     states = ["red", "yellow", "green"]
#     for state in states:
#         ks.add_state(state)
#
#     # Set initial state
#     ks.set_initial_state("red")
#
#     # Add transitions (cyclic)
#     ks.add_transition("red", "green")
#     ks.add_transition("green", "yellow")
#     ks.add_transition("yellow", "red")
#
#     # Add propositions
#     ks.set_proposition("red", "stop")
#     ks.set_proposition("red", "urgent")
#     ks.set_proposition("yellow", "caution")
#     ks.set_proposition("green", "go")
#     ks.set_proposition("green", "safe")
#
#     return ks
#
#
# def create_geometric_example() -> KripkeStructure:
#     """Create geometric convex hull example."""
#     ks = KripkeStructure()
#
#     # Create states with weight vectors
#     state_weights = {
#         "s0": [1.0, 2.0],
#         "s1": [3.0, 1.0],
#         "s2": [2.0, 3.0],
#         "s3": [0.5, 1.5]
#     }
#
#     for state, weights in state_weights.items():
#         ks.add_state(state)
#         ks.set_weight_vector(state, weights)
#
#     ks.set_initial_state("s0")
#
#     # Add transitions
#     ks.add_transition("s0", "s1")
#     ks.add_transition("s1", "s2")
#     ks.add_transition("s2", "s3")
#     ks.add_transition("s3", "s0")
#     ks.add_transition("s1", "s3")  # Additional edge
#
#     # Add propositions
#     ks.set_proposition("s0", "init")
#     ks.set_proposition("s1", "high_x")
#     ks.set_proposition("s2", "high_y")
#     ks.set_proposition("s3", "low")
#
#     return ks
#
#
# def run_default_example():
#     """Run the default Traffic Light System example."""
#     print("🚦 LTL+LimAvg Model Checker - Traffic Light System")
#     print("=" * 60)
#
#     # Create and display system
#     ks = create_traffic_light_system()
#     print("\n📊 System Overview:")
#     print(f"  States: {list(ks.states)}")
#     print(f"  Initial: {list(ks.initial_states)}")
#     print(f"  Transitions: {len(ks.transitions)}")
#     print(f"  Propositions: {dict(ks.propositions)}")
#
#     # Initialize checker
#     checker = LTLimModelChecker(ks)
#
#     # Test LTL formulas
#     print("\n🔍 LTL Formula Verification:")
#     print("-" * 40)
#
#     test_formulas = [
#         "G F go",  # Always eventually green light
#         "G F stop",  # Always eventually red light
#         "F G safe",  # Eventually always safe (should fail)
#         "G (stop -> F go)",  # If stop, then eventually go
#     ]
#
#     for formula in test_formulas:
#         print(f"\n🧪 Testing: {formula}")
#         result, error = checker.check_ltl_formula(formula)
#
#         status = "✅ SATISFIED" if result else "❌ NOT SATISFIED"
#         print(f"   Result: {status}")
#
#         if error:
#             print(f"   Error: {error}")
#
#         # Show product automaton analysis
#         try:
#             analysis = checker.analyze_product_automaton()
#             print(f"   Product: {analysis['num_states']} states, "
#                   f"{analysis['num_transitions']} transitions")
#         except:
#             pass
#
#     # Test limit-average constraints
#     print("\n📈 Limit-Average Constraint Checking:")
#     print("-" * 45)
#
#     fairness_checker = FairnessChecker()
#
#     # Add weight vectors for limit-average analysis
#     ks.set_weight_vector("red", [1.0])  # High cost for red
#     ks.set_weight_vector("yellow", [0.5])  # Medium cost for yellow
#     ks.set_weight_vector("green", [0.1])  # Low cost for green
#
#     constraints = [
#         ("avg_cost <= 0.6", 0.6),
#         ("avg_cost <= 0.3", 0.3),
#     ]
#
#     for constraint, threshold in constraints:
#         print(f"\n🎯 Testing: {constraint}")
#         result, error = fairness_checker.check_limit_average_constraint(
#             ks, constraint, threshold
#         )
#
#         status = "✅ SATISFIED" if result else "❌ NOT SATISFIED"
#         print(f"   Result: {status}")
#         if error:
#             print(f"   Details: {error}")
#
#
# def run_interactive_mode():
#     """Run interactive mode for custom model building."""
#     print("🔧 Interactive Model Builder")
#     print("=" * 40)
#
#     ks = KripkeStructure()
#
#     # Get number of states
#     while True:
#         try:
#             num_states = int(input("\n📝 Enter number of states: "))
#             if num_states > 0:
#                 break
#             print("❌ Please enter a positive number.")
#         except ValueError:
#             print("❌ Please enter a valid number.")
#
#     # Add states
#     states = []
#     for i in range(num_states):
#         while True:
#             state_name = input(f"   State {i + 1} name: ").strip()
#             if state_name and state_name not in states:
#                 states.append(state_name)
#                 ks.add_state(state_name)
#                 break
#             print("❌ Please enter a unique, non-empty state name.")
#
#     # Set initial state
#     print(f"\n🏁 Available states: {states}")
#     while True:
#         initial = input("   Enter initial state: ").strip()
#         if initial in states:
#             ks.set_initial_state(initial)
#             break
#         print("❌ Please enter a valid state name.")
#
#     # Add transitions
#     print("\n🔗 Add transitions (format: from_state to_state, empty line to finish):")
#     while True:
#         transition = input("   Transition: ").strip()
#         if not transition:
#             break
#
#         try:
#             from_state, to_state = transition.split()
#             if from_state in states and to_state in states:
#                 ks.add_transition(from_state, to_state)
#                 print(f"   ✅ Added: {from_state} → {to_state}")
#             else:
#                 print("❌ Invalid state names.")
#         except ValueError:
#             print("❌ Format: from_state to_state")
#
#     # Add propositions
#     print("\n🏷️  Add propositions (format: state proposition, empty line to finish):")
#     while True:
#         prop_input = input("   Proposition: ").strip()
#         if not prop_input:
#             break
#
#         try:
#             state, proposition = prop_input.split()
#             if state in states:
#                 ks.set_proposition(state, proposition)
#                 print(f"   ✅ Added: {state} has {proposition}")
#             else:
#                 print("❌ Invalid state name.")
#         except ValueError:
#             print("❌ Format: state proposition")
#
#     # Display created model
#     print("\n📊 Created Model:")
#     print(f"   States: {list(ks.states)}")
#     print(f"   Initial: {list(ks.initial_states)}")
#     print(f"   Transitions: {len(ks.transitions)}")
#     print(f"   Propositions: {dict(ks.propositions)}")
#
#     # Test LTL formulas
#     checker = LTLimModelChecker(ks)
#
#     print("\n🔍 Test LTL formulas (empty line to finish):")
#     while True:
#         formula = input("   Formula: ").strip()
#         if not formula:
#             break
#
#         try:
#             result, error = checker.check_ltl_formula(formula)
#             status = "✅ SATISFIED" if result else "❌ NOT SATISFIED"
#             print(f"   Result: {status}")
#             if error:
#                 print(f"   Error: {error}")
#         except Exception as e:
#             print(f"   ❌ Error: {e}")
#
#
# def run_geometric_example():
#     """Run geometric convex hull analysis example."""
#     print("📐 Geometric Convex Hull Analysis")
#     print("=" * 45)
#
#     # Create geometric system
#     ks = create_geometric_example()
#
#     print("\n📊 System Overview:")
#     print(f"  States: {list(ks.states)}")
#     print(f"  Initial: {list(ks.initial_states)}")
#     print(f"  Weight vectors:")
#     for state in ks.states:
#         weights = ks.weight_vectors.get(state, [])
#         print(f"    {state}: {weights}")
#
#     # Initialize convex hull checker
#     hull_checker = ConvexHullChecker()
#
#     print("\n🔺 Convex Hull Analysis:")
#     print("-" * 30)
#
#     # Analyze reachable weight vectors
#     reachable_weights = []
#     visited = set()
#     stack = list(ks.initial_states)
#
#     while stack:
#         current = stack.pop()
#         if current in visited:
#             continue
#         visited.add(current)
#
#         weights = ks.weight_vectors.get(current, [])
#         if weights:
#             reachable_weights.append(weights)
#
#         # Add successors
#         for from_state, to_state in ks.transitions:
#             if from_state == current and to_state not in visited:
#                 stack.append(to_state)
#
#     print(f"📍 Reachable weight vectors: {reachable_weights}")
#
#     if len(reachable_weights) >= 3:
#         # Compute convex hull
#         try:
#             hull_points = hull_checker.compute_convex_hull(reachable_weights)
#             print(f"🔺 Convex hull vertices: {hull_points}")
#             print(f"📏 Hull area: {hull_checker.compute_hull_area(hull_points):.3f}")
#
#             # Test point inclusion
#             test_points = [[1.5, 1.5], [0.0, 0.0], [4.0, 4.0]]
#             for point in test_points:
#                 is_inside = hull_checker.point_in_convex_hull(point, hull_points)
#                 status = "✅ INSIDE" if is_inside else "❌ OUTSIDE"
#                 print(f"🎯 Point {point}: {status}")
#
#         except Exception as e:
#             print(f"❌ Convex hull error: {e}")
#     else:
#         print("⚠️  Need at least 3 points for convex hull analysis.")
#
#     # Test LTL properties on geometric system
#     print("\n🔍 LTL Analysis:")
#     print("-" * 20)
#
#     checker = LTLimModelChecker(ks)
#
#     geo_formulas = [
#         "F high_x",  # Eventually reach high x-coordinate
#         "G F init",  # Always eventually return to initial state
#         "F G low",  # Eventually always stay in low region
#     ]
#
#     for formula in geo_formulas:
#         print(f"\n🧪 Testing: {formula}")
#         result, error = checker.check_ltl_formula(formula)
#         status = "✅ SATISFIED" if result else "❌ NOT SATISFIED"
#         print(f"   Result: {status}")
#         if error:
#             print(f"   Error: {error}")
#
#
# def test_spot_integration():
#     """Test Spot library integration."""
#     print("🔗 Testing Spot Library Integration")
#     print("=" * 45)
#
#     try:
#         import spot
#         print("✅ Spot library is available")
#         print(f"📦 Spot version: {spot.version()}")
#     except ImportError:
#         print("❌ Spot library not available")
#         print("📦 Install with: pip install spot")
#         return
#
#     # Create test system
#     ks = create_traffic_light_system()
#     checker = LTLimModelChecker(ks)
#
#     print("\n🔍 Comparing Spot vs Manual verification:")
#     print("-" * 50)
#
#     test_formulas = [
#         "G F go",
#         "F G stop",
#         "G (stop -> F go)",
#     ]
#
#     for formula in test_formulas:
#         print(f"\n🧪 Formula: {formula}")
#
#         # Test with Spot
#         try:
#             spot_result, spot_error = checker.check_ltl_formula(formula, use_spot=True)
#             spot_status = "✅ SAT" if spot_result else "❌ UNSAT"
#             print(f"   Spot:   {spot_status}")
#             if spot_error:
#                 print(f"           Error: {spot_error}")
#         except Exception as e:
#             print(f"   Spot:   ❌ Error: {e}")
#
#         # Test with manual approach
#         try:
#             manual_result, manual_error = checker.check_ltl_formula(formula, use_spot=False)
#             manual_status = "✅ SAT" if manual_result else "❌ UNSAT"
#             print(f"   Manual: {manual_status}")
#             if manual_error:
#                 print(f"           Error: {manual_error}")
#         except Exception as e:
#             print(f"   Manual: ❌ Error: {e}")
#
#
# def test_product_construction():
#     """Test product automaton construction."""
#     print("⚙️ Testing Product Automaton Construction")
#     print("=" * 50)
#
#     # Create simple test system
#     ks = KripkeStructure()
#     ks.add_state("s0")
#     ks.add_state("s1")
#     ks.add_state("s2")
#     ks.set_initial_state("s0")
#     ks.add_transition("s0", "s1")
#     ks.add_transition("s1", "s2")
#     ks.add_transition("s2", "s0")
#     ks.set_proposition("s0", "p")
#     ks.set_proposition("s1", "q")
#     ks.set_proposition("s2", "r")
#
#     print("\n📊 Test System:")
#     print(f"  States: {list(ks.states)}")
#     print(f"  Propositions: {dict(ks.propositions)}")
#
#     checker = LTLimModelChecker(ks)
#
#     # Test various LTL patterns
#     print("\n🔍 Product Automaton Analysis:")
#     print("-" * 40)
#
#     formulas = [
#         "F p",  # Eventually p
#         "G q",  # Always q
#         "G F r",  # Infinitely often r
#         "F G p",  # Eventually always p
#     ]
#
#     for formula in formulas:
#         print(f"\n🧪 Testing: {formula}")
#
#         try:
#             result, error = checker.check_ltl_formula(formula, use_spot=False)
#             status = "✅ SATISFIED" if result else "❌ NOT SATISFIED"
#             print(f"   Result: {status}")
#
#             if error:
#                 print(f"   Error: {error}")
#
#             # Analyze constructed product automaton
#             analysis = checker.analyze_product_automaton()
#             print(f"   Product Analysis:")
#             print(f"     • States: {analysis['num_states']}")
#             print(f"     • Transitions: {analysis['num_transitions']}")
#             print(f"     • Accepting states: {analysis['num_accepting_states']}")
#             print(f"     • Reachable states: {analysis['num_reachable_states']}")
#             print(f"     • SCCs: {analysis['num_sccs']}")
#             print(f"     • Is empty: {analysis['is_empty']}")
#             print(f"     • Has accepting SCC: {analysis['has_accepting_scc']}")
#
#         except Exception as e:
#             print(f"   ❌ Error: {e}")
#
#
# def test_comprehensive_features():
#     """Test all major features comprehensively."""
#     print("🧪 Comprehensive Feature Testing")
#     print("=" * 45)
#
#     # Test 1: Basic Kripke Structure
#     print("\n1️⃣ Testing Kripke Structure Creation...")
#     ks = create_traffic_light_system()
#     assert len(ks.states) == 3
#     assert len(ks.transitions) == 3
#     print("   ✅ Kripke structure created successfully")
#
#     # Test 2: LTL Model Checking
#     print("\n2️⃣ Testing LTL Model Checking...")
#     checker = LTLimModelChecker(ks)
#     result, _ = checker.check_ltl_formula("G F go")
#     print(f"   ✅ LTL checking works: {result}")
#
#     # Test 3: Product Automaton
#     print("\n3️⃣ Testing Product Automaton...")
#     analysis = checker.analyze_product_automaton()
#     assert analysis['num_states'] > 0
#     print("   ✅ Product automaton constructed")
#
#     # Test 4: Fairness Checking
#     print("\n4️⃣ Testing Fairness Constraints...")
#     fairness = FairnessChecker()
#     ks.set_weight_vector("red", [1.0])
#     ks.set_weight_vector("yellow", [0.5])
#     ks.set_weight_vector("green", [0.1])
#     result, _ = fairness.check_limit_average_constraint(ks, "avg <= 0.6", 0.6)
#     print(f"   ✅ Fairness checking works: {result}")
#
#     # Test 5: Convex Hull
#     print("\n5️⃣ Testing Convex Hull Analysis...")
#     hull_checker = ConvexHullChecker()
#     points = [[0, 0], [1, 0], [0, 1], [1, 1]]
#     hull = hull_checker.compute_convex_hull(points)
#     assert len(hull) >= 3
#     print("   ✅ Convex hull computation works")
#
#     print("\n🎉 All features tested successfully!")
#
#
# def print_help():
#     """Print help information."""
#     print("🔧 LTL+LimAvg Model Checker - Help")
#     print("=" * 40)
#     print("\nAvailable commands:")
#     print("  python main.py              - Run default traffic light example")
#     print("  python main.py interactive  - Interactive model builder")
#     print("  python main.py geometric    - Geometric/convex hull example")
#     print("  python main.py spot         - Test Spot library integration")
#     print("  python main.py product      - Test product automaton construction")
#     print("  python main.py test         - Run comprehensive feature tests")
#     print("  python main.py help         - Show this help message")
#     print("\nFeatures:")
#     print("  ✅ LTL model checking (manual + Spot)")
#     print("  ✅ Product automaton construction (K × A)")
#     print("  ✅ Limit-average constraint checking")
#     print("  ✅ Fairness analysis with Z3")
#     print("  ✅ Convex hull geometric analysis")
#     print("  ✅ Interactive model building")
#     print("  ✅ Comprehensive CLI interface")
#
#
# def main():
#     """Main entry point."""
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(levelname)s: %(message)s'
#     )
#
#     if len(sys.argv) == 1:
#         run_default_example()
#     elif sys.argv[1] == "interactive":
#         run_interactive_mode()
#     elif sys.argv[1] == "geometric":
#         run_geometric_example()
#     elif sys.argv[1] == "spot":
#         test_spot_integration()
#     elif sys.argv[1] == "product":
#         test_product_construction()
#     elif sys.argv[1] == "test":
#         test_comprehensive_features()
#     elif sys.argv[1] in ["help", "-h", "--help"]:
#         print_help()
#     else:
#         print(f"❌ Unknown command: {sys.argv[1]}")
#         print(f"💡 Use 'python {sys.argv[0]} help' for available commands")
#         sys.exit(1)
#
#
# if __name__ == "__main__":
#     main()
# Instead of "import owl", you "import subprocess".
# This is the Python library that lets you run ANY external command.
# import subprocess
# import os  # Often useful for handling file paths
#
#
# # This function is your new "import owl".
# # You can put this in a utility file (e.g., owl_helper.py) and import it into other scripts.
# def run_owl_command(arguments):
#     """
#     A general-purpose function to run the OWL model checker with given arguments.
#
#     Args:
#         arguments (list): A list of command-line arguments for owl.
#                           For example: ['--formula', 'EF final', '--file', 'my_net.pnml']
#
#     Returns:
#         A tuple containing (stdout, stderr) from the OWL process.
#     """
#     # The command always starts with 'owl', the name of the executable.
#     command = ["owl"] + arguments
#
#     print(f"--- Running Command ---\n{' '.join(command)}\n")
#
#     try:
#         # This is the line that actually "runs" OWL. It's like typing the command
#         # in your terminal and hitting Enter.
#         result = subprocess.run(
#             command,
#             capture_output=True,  # Capture what OWL prints to the console
#             text=True,  # Convert the output to a string
#             check=True  # Automatically check if OWL reported an error
#         )
#
#         # The result from OWL is now available in your Python script to use.
#         # You can parse this text, check for "TRUE" or "FALSE", extract data, etc.
#         return result.stdout, result.stderr
#
#     except FileNotFoundError:
#         print("ERROR: 'owl' command not found in your PATH.")
#         return None, None
#     except subprocess.CalledProcessError as e:
#         print(f"ERROR: OWL exited with an error (status code: {e.returncode})")
#         print("\n--- OWL Error Output (stderr) ---")
#         print(e.stderr)
#         return e.stdout, e.stderr
#
#
# # --- How to Use Your "Imported" Function ---
# if __name__ == "__main__":
#     # 1. Define the input file for OWL.
#     #    (Using the same simple Petri Net example as before)
#     pnml_content = """<?xml version="1.0" encoding="UTF-8"?>
# <pnml>
#   <net id="Net" type="P/T net">
#     <page id="P1">
#       <place id="p1"><initialMarking><text>1</text></initialMarking></place>
#       <place id="p2"><finalMarking><text>1</text></finalMarking></place>
#       <transition id="t1"/>
#       <arc id="a1" source="p1" target="t1"/><arc id="a2" source="t1" target="p2"/>
#     </page>
#   </net>
# </pnml>"""
#
#     file_to_check = "my_test_net.pnml"
#     absolute_file_path = os.path.abspath(file_to_check)
#     with open(absolute_file_path, "w") as f:
#         f.write(pnml_content)
#
#     # 2. Define the arguments you want to pass to OWL.
#     #    This is equivalent to `owl --formula "EF final" --file my_test_net.pnml`
#     owl_args = [
#         "--formula", "EF final",
#         "--file", absolute_file_path
#     ]
#
#     # 3. Call your wrapper function to execute OWL.
#     stdout, stderr = run_owl_command(owl_args)
#
#     # 4. Process the results in Python.
#     if stdout:
#         print("--- Python Processing OWL's Result ---")
#         if "The formula is TRUE" in stdout:
#             print("Success! Python has confirmed that the final marking is reachable.")
#         elif "The formula is FALSE" in stdout:
#             print("Analysis complete. Python has confirmed the final marking is NOT reachable.")
#         else:
#             print("OWL ran, but the output was not as expected.")
#             print(stdout)


"""
Enhanced Main Module for LTL+LimAvg Model Checker with OWL Integration
"""
# from core.LTLlimFormula import LTLlimFormula
from core.QuantitativeKripkeStructure import QuantitativeKripkeStructure

# import sys
# import os
# from typing import List, Dict, Any
# from core import KripkeStructure, LTLLimAvgChecker
# from owl_helper import check_owl_installation
#
#
# def create_example_kripke() -> KripkeStructure:
#     """Create an example Kripke structure for demonstration."""
#     kripke = KripkeStructure()
#
#     # Add states
#     kripke.add_state("s0", is_initial=True)
#     kripke.add_state("s1")
#     kripke.add_state("s2")
#     kripke.add_state("s3")
#
#     # Add transitions with weights
#     kripke.add_transition("s0", "s1", weight=1.0)
#     kripke.add_transition("s1", "s2", weight=2.0)
#     kripke.add_transition("s2", "s3", weight=1.5)
#     kripke.add_transition("s3", "s0", weight=0.5)
#     kripke.add_transition("s1", "s3", weight=3.0)
#
#     # Add propositions
#     kripke.add_proposition("s0", "start")
#     kripke.add_proposition("s1", "process")
#     kripke.add_proposition("s2", "ready")
#     kripke.add_proposition("s3", "final")
#
#     return kripke
#
#
# def run_demo_checks(checker: LTLLimAvgChecker):
#     """Run demonstration checks."""
#     print("\n🎮 Running Demo Checks")
#     print("=" * 50)
#
#     # Demo LTL formulas
#     ltl_formulas = [
#         "EF final",  # Eventually reach final state
#         "AG !error",  # Never reach error (no error state, so true)
#         "EF (process & F ready)",  # Eventually process and then ready
#         "AG (start -> EF final)"  # From start, always eventually final
#     ]
#
#     # Check LTL formulas
#     print("\n📝 LTL Formula Checks:")
#     for formula in ltl_formulas:
#         result = checker.check_ltl_formula(formula)
#         status = "✅ SATISFIED" if result.get('satisfied') else "❌ NOT SATISFIED"
#         print(f"   {formula}: {status}")
#
#     # Demo limit-average checks
#     print("\n📊 Limit-Average Checks:")
#     weight_sequences = [
#         ([1.0, 2.0, 1.5, 0.5], 1.0),
#         ([3.0, 1.0, 2.0], 2.0),
#         ([0.5, 0.5, 0.5], 1.0)
#     ]
#
#     for weights, threshold in weight_sequences:
#         result = checker.check_limit_average(weights, threshold)
#         status = "✅ SATISFIED" if result.get('satisfied') else "❌ NOT SATISFIED"
#         avg = result.get('limit_average', 0)
#         print(f"   Weights {weights}, threshold {threshold}: {status} (avg: {avg:.2f})")
#
#     # Demo combined check
#     print("\n🔄 Combined LTL + Limit-Average Check:")
#     combined_result = checker.check_combined_ltl_limavg(
#         "EF final",
#         [1.0, 2.0, 1.5],
#         1.2
#     )
#     status = "✅ SATISFIED" if combined_result.get('satisfied') else "❌ NOT SATISFIED"
#     print(f"   Formula: EF final, weights: [1.0, 2.0, 1.5], threshold: 1.2")
#     print(f"   Result: {status}")
#
#     # Model statistics
#     print("\n📈 Model Statistics:")
#     stats = checker.get_model_statistics()
#     for key, value in stats.items():
#         print(f"   {key.replace('_', ' ').title()}: {value}")
#
#
# def interactive_mode(checker: LTLLimAvgChecker):
#     """Run interactive mode."""
#     print("\n🎯 Interactive LTL+LimAvg Model Checker")
#     print("=" * 50)
#     print("Commands:")
#     print("  ltl <formula>              - Check LTL formula")
#     print("  limavg <weights> <threshold> - Check limit-average")
#     print("  combined <formula> <weights> <threshold> - Combined check")
#     print("  export <format> [file]     - Export model (pnml/dot)")
#     print("  stats                      - Show model statistics")
#     print("  structure                  - Display Kripke structure")
#     print("  help                       - Show this help")
#     print("  quit                       - Exit interactive mode")
#     print()
#
#     while True:
#         try:
#             command = input("Checker> ").strip()
#
#             if not command:
#                 continue
#
#             parts = command.split()
#             cmd = parts[0].lower()
#
#             if cmd == "quit" or cmd == "exit":
#                 print("👋 Goodbye!")
#                 break
#
#             elif cmd == "help":
#                 print("\n📚 Available Commands:")
#                 print("  ltl <formula>              - Check LTL formula (e.g., ltl 'EF final')")
#                 print("  limavg <weights> <threshold> - Check limit-average (e.g., limavg [1.0,2.0] 1.5)")
#                 print("  combined <formula> <weights> <threshold> - Combined check")
#                 print("  export <format> [file]     - Export model (e.g., export pnml model.pnml)")
#                 print("  stats                      - Show model statistics")
#                 print("  structure                  - Display Kripke structure")
#                 print("  help                       - Show this help")
#                 print("  quit                       - Exit")
#
#             elif cmd == "ltl":
#                 if len(parts) < 2:
#                     print("❌ Usage: ltl <formula>")
#                     continue
#
#                 formula = " ".join(parts[1:])
#                 result = checker.check_ltl_formula(formula)
#
#                 if result.get('satisfied') is not None:
#                     status = "✅ SATISFIED" if result['satisfied'] else "❌ NOT SATISFIED"
#                     print(f"Result: {status}")
#                     if result.get('counterexample'):
#                         print(f"Counterexample: {result['counterexample']}")
#                 else:
#                     print(f"❌ Error: {result.get('error', 'Unknown error')}")
#
#             elif cmd == "limavg":
#                 if len(parts) < 3:
#                     print("❌ Usage: limavg <weights> <threshold>")
#                     print("   Example: limavg [1.0,2.0,1.5] 1.2")
#                     continue
#
#                 try:
#                     weights_str = parts[1]
#                     threshold = float(parts[2])
#
#                     # Parse weights list
#                     weights_str = weights_str.strip('[]')
#                     weights = [float(w.strip()) for w in weights_str.split(',')]
#
#                     result = checker.check_limit_average(weights, threshold)
#                     status = "✅ SATISFIED" if result.get('satisfied') else "❌ NOT SATISFIED"
#                     avg = result.get('limit_average', 0)
#                     print(f"Result: {status} (average: {avg:.3f})")
#
#                 except ValueError as e:
#                     print(f"❌ Invalid input: {e}")
#
#             elif cmd == "combined":
#                 if len(parts) < 4:
#                     print("❌ Usage: combined <formula> <weights> <threshold>")
#                     continue
#
#                 try:
#                     formula = parts[1]
#                     weights_str = parts[2].strip('[]')
#                     weights = [float(w.strip()) for w in weights_str.split(',')]
#                     threshold = float(parts[3])
#
#                     result = checker.check_combined_ltl_limavg(formula, weights, threshold)
#                     status = "✅ SATISFIED" if result.get('satisfied') else "❌ NOT SATISFIED"
#                     print(f"Combined Result: {status}")
#
#                 except ValueError as e:
#                     print(f"❌ Invalid input: {e}")
#
#             elif cmd == "export":
#                 if len(parts) < 2:
#                     print("❌ Usage: export <format> [filename]")
#                     print("   Formats: pnml, dot")
#                     continue
#
#                 format_type = parts[1].lower()
#                 filename = parts[2] if len(parts) > 2 else None
#
#                 try:
#                     exported_file = checker.export_model(format_type, filename)
#                     print(f"✅ Model exported to: {exported_file}")
#                 except ValueError as e:
#                     print(f"❌ Export error: {e}")
#
#             elif cmd == "stats":
#                 stats = checker.get_model_statistics()
#                 print("\n📊 Model Statistics:")
#                 for key, value in stats.items():
#                     print(f"  {key.replace('_', ' ').title()}: {value}")
#
#             elif cmd == "structure":
#                 checker.kripke.display_structure()
#
#             else:
#                 print(f"❌ Unknown command: {cmd}")
#                 print("   Type 'help' for available commands")
#
#         except KeyboardInterrupt:
#             print("\n👋 Goodbye!")
#             break
#         except Exception as e:
#             print(f"❌ Error: {e}")
#
#
# def main():
#     """Main function."""
#     print("🦉 LTL+LimAvg Model Checker with OWL Integration")
#     print("=" * 55)
#
#     # Check OWL availability
#     print("\n🔍 Checking OWL availability...")
#     if not check_owl_installation():
#         print("\n⚠️  OWL model checker not found!")
#         print("📋 Please install OWL in WSL first:")
#         print("   1. Run: python setup_owl.py")
#         print("   2. Follow the installation instructions")
#         print("\n❓ Continue anyway? (LTL checking will be disabled)")
#
#         response = input("Continue? (y/N): ").strip().lower()
#         if response != 'y':
#             print("👋 Exiting. Please install OWL first.")
#             return
#
#     # Create example Kripke structure
#     print("\n🏗️  Creating example Kripke structure...")
#     kripke = create_example_kripke()
#     kripke.display_structure()
#
#     # Initialize checker
#     print("\n⚙️  Initializing LTL+LimAvg checker...")
#     checker = LTLLimAvgChecker(kripke)
#
#     # Choose mode
#     if len(sys.argv) > 1 and sys.argv[1] == "--demo":
#         # Demo mode
#         run_demo_checks(checker)
#     elif len(sys.argv) > 1 and sys.argv[1] == "--export":
#         # Export mode
#         format_type = sys.argv[2] if len(sys.argv) > 2 else "pnml"
#         filename = sys.argv[3] if len(sys.argv) > 3 else None
#         exported_file = checker.export_model(format_type, filename)
#         print(f"✅ Model exported to: {exported_file}")
#     else:
#         # Interactive mode
#         interactive_mode(checker)
#
#
if __name__ == "__main__":

    from core.ltllim_processor import *
    # Define a structure with multiple boolean variables
    qks = QuantitativeKripkeStructure(
        states={'s1', 's2', 's3'},
        init_state='s1',
        edges={('s1', 's2'), ('s1', 's3'), ('s2', 's3'), ('s3', 's1')},
        boolean_vars={'p', 'q', 'r'},  # All possible boolean variables
        logical_formulas={
            's1': {'p', 'q'},  # p and q are true, r is false
            's2': {'q'},  # q is true, p and r are false
            's3': set()  # All false
        },
        numeric_values={
            's1': {'x': 3.0, 'y': 1.5},
            's2': {'x': -5.0, 'y': 2.0},
            's3': {'x': 1.0, 'y': 0.0}
        }
    )

    print(qks)

    print("##################################")
    processor = process_formula("G(p -> F(q & LimInfAvg(r) > 0.5))")
