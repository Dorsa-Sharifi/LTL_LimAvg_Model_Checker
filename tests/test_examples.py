"""
Test Examples for LTL+LimAvg Model Checker
Various test cases to validate the implementation.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kripke_structure import KripkeStructure
from checker.ltlim_model_checker import LTLimModelChecker


def test_simple_system():
    """Test a simple system with basic LTL properties."""
    print("=== Test: Simple System ===")

    # Create a simple Kripke structure: s0 -> s1 -> s0 (cycle)
    kripke = KripkeStructure()
    kripke.add_initial_state('s0')
    kripke.add_state('s1')

    # Transitions
    kripke.add_transition('s0', 's1')
    kripke.add_transition('s1', 's0')

    # Atomic propositions
    kripke.add_atomic_prop('s0', 'p')
    kripke.add_atomic_prop('s1', 'q')

    # Variables for limit-average
    kripke.set_variable('s0', 'cost', 2)
    kripke.set_variable('s1', 'cost', 4)

    # Model checker
    checker = LTLimModelChecker()

    # Test 1: F(q) - Eventually q
    print("\nTest 1: F(q) - Eventually q")
    result1 = checker.check_model(kripke, 'F(q)', verbose=True)
    print(f"Result: {'PASS' if result1['satisfies'] else 'FAIL'}")

    # Test 2: GF(p) - Infinitely often p
    print("\nTest 2: GF(p) - Infinitely often p")
    result2 = checker.check_model(kripke, 'GF p', verbose=True)
    print(f"Result: {'PASS' if result2['satisfies'] else 'FAIL'}")

    # Test 3: F(q) with chi constraint
    print("\nTest 3: F(q) with cost >= 3")
    result3 = checker.check_model(kripke, 'F(q)', 'cost >= 3', verbose=True)
    print(f"Result: {'PASS' if result3['satisfies'] else 'FAIL'}")

    # Test 4: F(q) with strict chi constraint
    print("\nTest 4: F(q) with cost >= 5")
    result4 = checker.check_model(kripke, 'F(q)', 'cost >= 5', verbose=True)
    print(f"Result: {'PASS' if result4['satisfies'] else 'FAIL'}")

    return [result1['satisfies'], result2['satisfies'], result3['satisfies'], not result4['satisfies']]


def test_mutual_exclusion():
    """Test mutual exclusion system."""
    print("\n=== Test: Mutual Exclusion System ===")

    # States: (n0,n1), (c0,n1), (n0,c1), (c0,c1) - not allowed
    kripke = KripkeStructure()

    # Add states
    states = ['n0_n1', 'c0_n1', 'n0_c1']  # No c0_c1 (mutual exclusion)
    for state in states:
        kripke.add_state(state)

    kripke.add_initial_state('n0_n1')

    # Transitions
    kripke.add_transition('n0_n1', 'c0_n1')  # Process 0 enters critical section
    kripke.add_transition('n0_n1', 'n0_c1')  # Process 1 enters critical section
    kripke.add_transition('c0_n1', 'n0_n1')  # Process 0 exits critical section
    kripke.add_transition('n0_c1', 'n0_n1')  # Process 1 exits critical section

    # Atomic propositions
    kripke.add_atomic_prop('c0_n1', 'critical0')
    kripke.add_atomic_prop('n0_c1', 'critical1')

    # Variables (e.g., time spent in critical section)
    kripke.set_variable('n0_n1', 'time', 1)
    kripke.set_variable('c0_n1', 'time', 5)  # Process 0 in critical section takes more time
    kripke.set_variable('n0_c1', 'time', 3)  # Process 1 in critical section

    checker = LTLimModelChecker()

    # Test 1: GF(critical0) - Process 0 gets critical section infinitely often
    print("\nTest 1: GF(critical0)")
    result1 = checker.check_model(kripke, 'GF critical0', verbose=True)
    print(f"Result: {'PASS' if result1['satisfies'] else 'FAIL'}")

    # Test 2: Average time constraint
    print("\nTest 2: GF(critical0) with time <= 4")
    result2 = checker.check_model(kripke, 'GF critical0', 'time <= 4', verbose=True)
    print(f"Result: {'PASS' if result2['satisfies'] else 'FAIL'}")

    return [result1['satisfies'], result2['satisfies']]


def test_producer_consumer():
    """Test producer-consumer system."""
    print("\n=== Test: Producer-Consumer System ===")

    # States represent buffer states: empty, item1, item2, full
    kripke = KripkeStructure()

    states = ['empty', 'item1', 'item2', 'full']
    for state in states:
        kripke.add_state(state)

    kripke.add_initial_state('empty')

    # Transitions (producer adds, consumer removes)
    kripke.add_transition('empty', 'item1')  # Produce
    kripke.add_transition('item1', 'item2')  # Produce
    kripke.add_transition('item2', 'full')  # Produce
    kripke.add_transition('full', 'item2')  # Consume
    kripke.add_transition('item2', 'item1')  # Consume
    kripke.add_transition('item1', 'empty')  # Consume

    # Add some cycles for testing
    kripke.add_transition('item1', 'item1')  # Producer-consumer balance
    kripke.add_transition('item2', 'item2')  # Producer-consumer balance

    # Atomic propositions
    kripke.add_atomic_prop('empty', 'buffer_empty')
    kripke.add_atomic_prop('full', 'buffer_full')
    kripke.add_atomic_prop('item1', 'has_items')
    kripke.add_atomic_prop('item2', 'has_items')

    # Variables (buffer utilization)
    kripke.set_variable('empty', 'utilization', 0)
    kripke.set_variable('item1', 'utilization', 1)
    kripke.set_variable('item2', 'utilization', 2)
    kripke.set_variable('full', 'utilization', 3)

    checker = LTLimModelChecker()

    # Test 1: GF(has_items) - Always eventually has items
    print("\nTest 1: GF(has_items)")
    result1 = checker.check_model(kripke, 'GF has_items', verbose=True)
    print(f"Result: {'PASS' if result1['satisfies'] else 'FAIL'}")

    # Test 2: With utilization constraint
    print("\nTest 2: GF(has_items) with utilization >= 1.5")
    result2 = checker.check_model(kripke, 'GF has_items', 'utilization >= 1.5', verbose=True)
    print(f"Result: {'PASS' if result2['satisfies'] else 'FAIL'}")

    # Test 3: Buffer never stays full forever
    print("\nTest 3: FG(buffer_full) should fail")
    result3 = checker.check_model(kripke, 'FG buffer_full', verbose=True)
    print(f"Result: {'PASS' if not result3['satisfies'] else 'FAIL'}")

    return [result1['satisfies'], result2['satisfies'], not result3['satisfies']]


def test_complex_chi_formula():
    """Test complex chi formulas with Boolean combinations."""
    print("\n=== Test: Complex Chi Formulas ===")

    # Create a simple system
    kripke = KripkeStructure()
    kripke.add_initial_state('s0')
    kripke.add_state('s1')
    kripke.add_transition('s0', 's1')
    kripke.add_transition('s1', 's0')

    kripke.add_atomic_prop('s0', 'p')
    kripke.set_variable('s0', 'x', 2)
    kripke.set_variable('s0', 'y', 6)
    kripke.set_variable('s1', 'x', 4)
    kripke.set_variable('s1', 'y', 2)

    checker = LTLimModelChecker()

    # Test complex chi formulas
    test_cases = [
        ('x >= 2', True, "Simple constraint"),
        ('x >= 5', False, "Failing constraint"),
        ('x >= 2 and y >= 3', True, "AND constraint"),
        ('x >= 5 and y >= 3', False, "Failing AND constraint"),
        ('x >= 5 or y >= 3', True, "OR constraint"),
        ('x <= 2 or y <= 2', True, "OR with <="),
        ('not (x >= 5)', True, "NOT constraint"),
        ('(x >= 2 and y >= 2) or (x <= 1)', True, "Complex parentheses")
    ]

    results = []
    for chi, expected, description in test_cases:
        print(f"\nTesting: {description} - '{chi}'")
        result = checker.check_model(kripke, 'GF p', chi, verbose=False)
        success = result['satisfies'] == expected
        print(f"Expected: {expected}, Got: {result['satisfies']} - {'PASS' if success else 'FAIL'}")
        results.append(success)

    return results


def run_all_tests():
    """Run all test cases."""
    print("Running LTL+LimAvg Model Checker Tests")
    print("=" * 50)

    all_results = []

    # Run individual test suites
    simple_results = test_simple_system()
    mutex_results = test_mutual_exclusion()
    prodcons_results = test_producer_consumer()
    complex_results = test_complex_chi_formula()

    all_results.extend(simple_results)
    all_results.extend(mutex_results)
    all_results.extend(prodcons_results)
    all_results.extend(complex_results)

    # Summary
    passed = sum(all_results)
    total = len(all_results)

    print(f"\n{'=' * 50}")
    print("TEST SUMMARY")
    print(f"{'=' * 50}")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed / total * 100:.1f}%")

    if passed == total:
        print("🎉 All tests PASSED!")
    else:
        print(f"❌ {total - passed} tests FAILED")

    return passed == total


if __name__ == "__main__":
    run_all_tests()
