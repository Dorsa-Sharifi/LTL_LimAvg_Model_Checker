#!/usr/bin/env python3

import sys
import os

# Add the path to your main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ltllim_parser import EnhancedLTLimProcessor, QuantitativeKripkeStructure


def run_comprehensive_test_suite():
    """Run the comprehensive test suite"""

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

        "MULTIPLE LIMIT-AVERAGE ASSERTIONS": [
            "LimInfAvg(x) >= 2.0 ∧ LimSupAvg(y) <= 3.0",
            "LimInfAvg(x) >= 1.0 ∨ LimSupAvg(z) <= 2.0",
            "LimInfAvg(x) >= 2.0 → LimSupAvg(y) <= 3.0",
            "F(LimInfAvg(x) >= 2.0 ∧ LimSupAvg(y) <= 3.0)",
        ]
    }

    # =============================================================================
    # TEST EXECUTION
    # =============================================================================

    print(" COMPREHENSIVE LTLim PROCESSOR TEST SUITE")
    print("=" * 100)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for category, formulas in test_categories.items():
        print(f"\n{'#' * 100}")
        print(f"📂 TEST CATEGORY: {category}")
        print(f"{'#' * 100}")

        for i, formula in enumerate(formulas, 1):
            total_tests += 1
            print(f"\n🔬 Test {total_tests}: {formula}")
            print("-" * 80)

            try:
                # Reset processor state for clean test
                processor.variables.clear()
                processor.propositions.clear()
                processor.limit_avg_assertions.clear()

                # Parse the formula first to check syntax
                parse_tree = processor.parse(formula)
                parsed_str = processor.tree_to_string(parse_tree)
                print(f"   ✅ Parsed successfully: {parsed_str}")

                # Test the complete pipeline with product construction
                results = processor.complete_pipeline_with_product(formula)

                if results:
                    successful_products = sum(1 for _, _, product in results if product is not None)
                    print(f"   ✅ Pipeline completed: {successful_products}/{len(results)} successful products")

                    # Analyze the results
                    for j, (chi, xi, product) in enumerate(results):
                        if product:
                            print(
                                f"      Disjunct {j + 1}: ✓ Product built | States: {len(product.states)} | Accepting: {len(product.accepting_states)}")
                        else:
                            print(f"      Disjunct {j + 1}: ✗ Product failed")

                    passed_tests += 1
                else:
                    print(f"   ❌ Pipeline failed - no results generated")
                    failed_tests += 1

            except SyntaxError as e:
                print(f"   ❌ Syntax error: {e}")
                failed_tests += 1
            except Exception as e:
                print(f"   💥 Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                failed_tests += 1

            print("-" * 80)

    # =============================================================================
    # TEST SUMMARY
    # =============================================================================

    print(f"\n{'=' * 100}")
    print("📊 TEST SUMMARY")
    print(f"{'=' * 100}")
    print(f"Total tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"📈 Success rate: {success_rate:.1f}%")

    # =============================================================================
    # ADDITIONAL FOCUSED TESTS FOR SPECIFIC FEATURES
    # =============================================================================

    print(f"\n{'=' * 100}")
    print("🎯 FOCUSED FEATURE TESTS")
    print(f"{'=' * 100}")

    # Test negation rules specifically
    negation_tests = [
        "¬p",
        "¬F p",
        "¬G p",
        "¬(p ∧ q)",
        "¬(p U q)",
        "¬LimInfAvg(x) >= 2.0",
    ]

    print("\n🔍 Testing Negation Rules:")
    for test in negation_tests:
        try:
            processor.variables.clear()
            processor.propositions.clear()

            tree = processor.parse(test)
            negated = processor.negate_formula(tree)
            negated_str = processor.tree_to_string(negated)

            print(f"   {test} → {negated_str}")

        except Exception as e:
            print(f"   ❌ {test} failed: {e}")

    # Test limit-average assertion extraction
    print("\n🔍 Testing Limit-Average Extraction:")
    extraction_tests = [
        "LimInfAvg(x) >= 2.0 ∧ F p",
        "G(LimSupAvg(y) <= 3.0 ∨ F q)",
        "LimInfAvg(x) >= 1.0 ∧ LimSupAvg(y) <= 2.0 ∧ LimInfAvg(z) > 0.5",
    ]

    for test in extraction_tests:
        try:
            processor.variables.clear()
            processor.propositions.clear()

            tree = processor.parse(test)
            assertions = processor.extract_limit_avg_assertions(tree)

            print(f"   {test}")
            for j, assertion in enumerate(assertions):
                print(f"      Assertion {j + 1}: {processor.tree_to_string(assertion)}")

        except Exception as e:
            print(f"   ❌ {test} failed: {e}")

    print(f"\n🎉 TESTING COMPLETE!")

    return passed_tests, failed_tests

#
if __name__ == "__main__":
    # Run the comprehensive test suite
    passed, failed = run_comprehensive_test_suite()

    # Exit with appropriate code for CI/CD
    if failed > 0:
        sys.exit(1)  # Exit with error if any tests failed
    else:
        sys.exit(0)  # Exit successfully if all tests passed

# def test_limavg_algorithm():
#     qks = QuantitativeKripkeStructure(
#         states={'s0', 's1', 's2', 's3'},
#         init_state='s0',
#         edges={
#             ('s0', 's1'), ('s0', 's2'),
#             ('s1', 's0'), ('s1', 's2'), ('s1', 's3'),
#             ('s2', 's1'), ('s2', 's3'),
#             ('s3', 's0'), ('s3', 's2')
#         },
#         boolean_vars={'p', 'q', 'r'},
#         logical_formulas={
#             's0': {'p'},
#             's1': {'q'},
#             's2': {'p', 'r'},
#             's3': {'q', 'r'}
#         },
#         numeric_values={
#             's0': {'x': 1.0, 'y': 2.0, 'z': 0.5},
#             's1': {'x': 3.0, 'y': 1.0, 'z': 1.5},
#             's2': {'x': 2.0, 'y': 3.0, 'z': 0.8},
#             's3': {'x': 4.0, 'y': 0.5, 'z': 2.0}
#         }
#     )
#
#     processor = EnhancedLTLimProcessor("/home/otebook/ltl_to_nbw.py", qks)
#
#     # SPECIAL DEEP DEBUG for the failing case
#     print(f"\n{'#' * 120}")
#     print(f"🔍 DEEP DEBUG ANALYSIS: LimSupAvg(y) <= 3.0")
#     print(f"{'#' * 120}")
#
#     formula = "LimSupAvg(y) <= 3.0"
#
#     # STEP 1: FORMULA PROCESSING
#     print("\n" + "=" * 80)
#     print("STEP 1: FORMULA PROCESSING & NEGATION")
#     print("=" * 80)
#
#     try:
#         disjuncts, negated_formula = processor.process_formula_negation(formula)
#         print(f"✅ Original formula: {formula}")
#         print(f"✅ Negated formula: {negated_formula}")
#         print(f"✅ Generated {len(disjuncts)} disjuncts")
#
#         for i, (chi, xi) in enumerate(disjuncts):
#             print(f"   Disjunct {i}:")
#             print(f"     χ (limit-avg): {chi}")
#             print(f"     ξ (LTL): {xi}")
#
#     except Exception as e:
#         print(f"❌ Formula processing failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return
#
#     if not disjuncts:
#         print("❌ No disjuncts generated!")
#         return
#
#     # STEP 2: NBW CONVERSION
#     print("\n" + "=" * 80)
#     print("STEP 2: NBW CONVERSION")
#     print("=" * 80)
#
#     chi, xi = disjuncts[0]  # Test first disjunct
#     print(f"🔧 Converting LTL to NBW: '{xi}'")
#
#     nbw_result = processor.wsl_converter.ltl_to_nbw(xi)
#     print(f"📋 NBW Conversion Result:")
#     print(f"   Success: {nbw_result.get('success', False)}")
#     print(f"   Error: {nbw_result.get('error', 'None')}")
#     print(f"   States: {nbw_result.get('states', 'N/A')}")
#     print(f"   Acceptance: {nbw_result.get('acceptance', 'N/A')}")
#
#     if not nbw_result.get('success', False):
#         print("❌ NBW conversion failed - cannot proceed")
#         return
#
#     # STEP 3: PRODUCT AUTOMATON CONSTRUCTION
#     print("\n" + "=" * 80)
#     print("STEP 3: PRODUCT AUTOMATON CONSTRUCTION")
#     print("=" * 80)
#
#     product = processor.build_product_for_disjunct(chi, xi, nbw_result)
#     if not product:
#         print("❌ Product construction failed")
#         return
#
#     print(f"✅ Product automaton built:")
#     print(f"   Total states (S×Q): {len(product.states)}")
#     print(f"   Initial states: {len(product.initial_states)}")
#     print(f"   Accepting states (S×α): {len(product.accepting_states)}")
#     print(f"   Transitions: {sum(len(t) for t in product.transitions.values())}")
#
#     # Show some example states
#     print(f"   Sample states (first 5): {list(product.states)[:5]}")
#     if product.accepting_states:
#         print(f"   Sample accepting states: {list(product.accepting_states)[:3]}")
#
#     # STEP 4: SCC DETECTION
#     print("\n" + "=" * 80)
#     print("STEP 4: STRONGLY CONNECTED COMPONENTS (SCC) DETECTION")
#     print("=" * 80)
#
#     sccs = product.find_maximal_sccs()
#     print(f"📊 Found {len(sccs)} maximal SCCs")
#
#     for i, scc in enumerate(sccs):
#         print(f"   SCC {i + 1}:")
#         print(f"     Size: {len(scc)} states")
#         print(f"     Has accepting states: {any(s in product.accepting_states for s in scc)}")
#         print(f"     Sample states: {list(scc)[:3]}")
#
#         # Check valuations in this SCC
#         valuations = product.limit_avg_checker.extract_scc_valuations(scc)
#         print(f"     Valuations extracted: {len(valuations)}")
#         for j, val in enumerate(valuations[:3]):  # Show first 3
#             print(f"       State {j}: y = {val.get('y', 'MISSING')}")
#
#     if not sccs:
#         print("❌ No SCCs found - this is the problem!")
#         return
#
#     # STEP 5: LIMIT-AVERAGE CHECKING
#     print("\n" + "=" * 80)
#     print("STEP 5: LIMIT-AVERAGE CHECKING")
#     print("=" * 80)
#
#     # Test the first SCC that has accepting states
#     accepting_sccs = [scc for scc in sccs if any(s in product.accepting_states for s in scc)]
#     print(f"🔍 Found {len(accepting_sccs)} SCCs with accepting states")
#
#     if not accepting_sccs:
#         print("❌ No SCCs with accepting states - fairness condition fails!")
#         return
#
#     test_scc = accepting_sccs[0]
#     print(f"🔧 Testing SCC with {len(test_scc)} states for: {chi}")
#
#     # Manual limit-average check
#     print(f"🔍 Manual limit-average check:")
#     valuations = product.limit_avg_checker.extract_scc_valuations(test_scc)
#     print(f"   Extracted {len(valuations)} valuations")
#
#     y_values = []
#     for i, val in enumerate(valuations):
#         y_val = val.get('y', None)
#         y_values.append(y_val)
#         print(f"   Valuation {i}: y = {y_val}")
#
#     if y_values:
#         avg_y = sum(y_values) / len(y_values)
#         print(f"   Average y in SCC: {avg_y}")
#         print(f"   Constraint: LimSupAvg(y) <= 3.0")
#         print(f"   Satisfied: {avg_y <= 3.0}")
#     else:
#         print("❌ No y-values found in valuations!")
#
#     # STEP 6: FORMAL COMPONENT CHECK
#     print("\n" + "=" * 80)
#     print("STEP 6: FORMAL COMPONENT CHECK")
#     print("=" * 80)
#
#     result = product.component_check(test_scc, chi)
#     print(f"🎯 Formal component check result: {result}")
#
#     # STEP 7: CONVEX HULL DEBUG
#     print("\n" + "=" * 80)
#     print("STEP 7: CONVEX HULL DEBUG")
#     print("=" * 80)
#
#     try:
#         if len(valuations) >= 2:
#             hull, variables = product.limit_avg_checker.compute_convex_hull(valuations)
#             print(f"✅ Convex hull computed:")
#             print(f"   Variables: {variables}")
#             print(f"   Vertices: {len(hull.vertices)}")
#             print(f"   Points shape: {hull.points.shape}")
#
#             # Show hull vertices
#             vertices = hull.points[hull.vertices]
#             print(f"   Vertex values:")
#             for i, vertex in enumerate(vertices):
#                 print(f"     Vertex {i}: {dict(zip(variables, vertex))}")
#         else:
#             print(f"⚠️ Not enough points for convex hull: {len(valuations)} valuations")
#
#     except Exception as e:
#         print(f"❌ Convex hull computation failed: {e}")
#         import traceback
#         traceback.print_exc()
#
#     # STEP 8: Z3 SOLVER CHECK
#     print("\n" + "=" * 80)
#     print("STEP 8: Z3 SOLVER AVAILABILITY")
#     print("=" * 80)
#
#     if product.limit_avg_checker.z3:
#         print("✅ Z3 is available")
#
#         # Test constraint parsing
#         assertions = product.limit_avg_checker._parse_limit_avg_assertions(chi)
#         print(f"🔍 Parsed assertions: {assertions}")
#
#         for assertion in assertions:
#             avg_type, var, op, value = assertion
#             print(f"   Assertion: {avg_type}({var}) {op} {value}")
#     else:
#         print("❌ Z3 is NOT available - using fallback methods")
#
#     print(f"\n{'#' * 120}")
#     print(f"🎯 FINAL DIAGNOSIS COMPLETE")
#     print(f"{'#' * 120}")
#
#
# def test_all_formulas():
#     qks = QuantitativeKripkeStructure(
#         states={'s0', 's1', 's2', 's3'},
#         init_state='s0',
#         edges={
#             ('s0', 's1'), ('s0', 's2'),
#             ('s1', 's0'), ('s1', 's2'), ('s1', 's3'),
#             ('s2', 's1'), ('s2', 's3'),
#             ('s3', 's0'), ('s3', 's2')
#         },
#         boolean_vars={'p', 'q', 'r'},
#         logical_formulas={
#             's0': {'p'},
#             's1': {'q'},
#             's2': {'p', 'r'},
#             's3': {'q', 'r'}
#         },
#         numeric_values={
#             's0': {'x': 1.0, 'y': 2.0, 'z': 0.5},
#             's1': {'x': 3.0, 'y': 1.0, 'z': 1.5},
#             's2': {'x': 2.0, 'y': 3.0, 'z': 0.8},
#             's3': {'x': 4.0, 'y': 0.5, 'z': 2.0}
#         }
#     )
#
#     processor = EnhancedLTLimProcessor("/home/otebook/ltl_to_nbw.py", qks)
#
#     # Test cases from your comprehensive test suite
#     test_formulas = [
#         "LimInfAvg(x) >= 2.0",
#         "LimSupAvg(y) <= 3.0",
#         "LimInfAvg(x) >= 2.0 ∧ p",
#         "LimSupAvg(y) <= 3.0 ∨ q",
#         "LimInfAvg(x) >= 2.0 ∧ LimSupAvg(y) <= 3.0"
#     ]
#
#     for formula in test_formulas:
#         print(f"\n{'#' * 100}")
#         print(f"TESTING: {formula}")
#         print(f"{'#' * 100}")
#
#         try:
#             results = processor.complete_pipeline_with_limavg_checking(formula)
#             print(f"✅ Test completed for: {formula}")
#         except Exception as e:
#             print(f"❌ Test failed for {formula}: {e}")
#             import traceback
#             traceback.print_exc()
#
#
# if __name__ == "__main__":
#     # Run deep debug for the problematic case
#     test_limavg_algorithm()
#
#     print(f"\n{'#' * 120}")
#     print(f"RUNNING ALL TESTS")
#     print(f"{'#' * 120}")
#
#     # Then run all tests
#     test_all_formulas()