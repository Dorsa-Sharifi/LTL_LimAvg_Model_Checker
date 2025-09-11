import sys
import os
from core.kripke_structure import KripkeStructure
from checker.ltlim_model_checker import LTLimModelChecker


def create_default_example():
    """Create a default example for demonstration."""
    print("Creating default example: Simple Traffic Light System")
    print("-" * 50)

    # Traffic light system: Red -> Yellow -> Green -> Red
    kripke = KripkeStructure()

    # States
    states = ['red', 'yellow', 'green']
    for state in states:
        kripke.add_state(state)

    kripke.add_initial_state('red')

    # Transitions (cycle)
    kripke.add_transition('red', 'yellow')
    kripke.add_transition('yellow', 'green')
    kripke.add_transition('green', 'red')

    # Atomic propositions
    kripke.add_atomic_prop('red', 'stop')
    kripke.add_atomic_prop('yellow', 'caution')
    kripke.add_atomic_prop('green', 'go')

    # Variables (e.g., time duration for each light)
    kripke.set_variable('red', 'duration', 30)  # Red light: 30 seconds
    kripke.set_variable('yellow', 'duration', 5)  # Yellow light: 5 seconds
    kripke.set_variable('green', 'duration', 25)  # Green light: 25 seconds

    return kripke


def run_default_example():
    """Run the default traffic light example."""
    print("🚦 LTL+LimAvg Model Checker - Traffic Light Example")
    print("=" * 60)

    kripke = create_default_example()
    checker = LTLimModelChecker()

    # Test various properties
    test_cases = [
        {
            'name': 'Traffic Light Cycle',
            'ltl': 'GF go',
            'chi': None,
            'description': 'Green light occurs infinitely often'
        },
        {
            'name': 'Average Duration Constraint',
            'ltl': 'GF go',
            'chi': 'duration >= 15',
            'description': 'Green light occurs with average duration ≥ 15 seconds'
        },
        {
            'name': 'Strict Duration Constraint',
            'ltl': 'GF stop',
            'chi': 'duration >= 25',
            'description': 'Red light occurs with average duration ≥ 25 seconds'
        },
        {
            'name': 'Complex Constraint',
            'ltl': 'GF caution',
            'chi': 'duration >= 5 and duration <= 10',
            'description': 'Yellow light with duration between 5-10 seconds'
        }
    ]

    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test['name']}")
        print(f"Description: {test['description']}")
        print(f"LTL Formula: {test['ltl']}")
        print(f"Chi Formula: {test['chi'] or 'None'}")
        print("-" * 40)

        result = checker.check_model(
            kripke,
            test['ltl'],
            test['chi'],
            verbose=True
        )

        results.append(result)

        if result['satisfies']:
            print("✅ PROPERTY SATISFIED")
        else:
            print("❌ PROPERTY VIOLATED")

        print()

    # Summary
    satisfied = sum(1 for r in results if r['satisfies'])
    total = len(results)

    print("=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Properties Satisfied: {satisfied}/{total}")
    print(f"Success Rate: {satisfied / total * 100:.1f}%")

    if satisfied == total:
        print(" All properties satisfied!")
    else:
        print(f"⚠  {total - satisfied} properties violated")

    return results


def interactive_mode():
    """Interactive mode for custom model checking."""
    print("🔧 Interactive LTL+LimAvg Model Checker")
    print("=" * 50)
    print("Build your own Kripke structure and test properties!")
    print()

    checker = LTLimModelChecker()
    kripke = KripkeStructure()

    while True:
        print("\nOptions:")
        print("1. Add state")
        print("2. Set initial state")
        print("3. Add transition")
        print("4. Add atomic proposition")
        print("5. Set variable value")
        print("6. View current structure")
        print("7. Check LTL+LimAvg property")
        print("8. Load example structure")
        print("9. Exit")

        try:
            choice = input("\nEnter choice (1-9): ").strip()

            if choice == '1':
                state = input("Enter state name: ").strip()
                kripke.add_state(state)
                print(f"✅ Added state '{state}'")

            elif choice == '2':
                state = input("Enter initial state name: ").strip()
                kripke.add_initial_state(state)
                print(f"✅ Set '{state}' as initial state")

            elif choice == '3':
                from_state = input("Enter source state: ").strip()
                to_state = input("Enter target state: ").strip()
                kripke.add_transition(from_state, to_state)
                print(f"✅ Added transition {from_state} -> {to_state}")

            elif choice == '4':
                state = input("Enter state name: ").strip()
                prop = input("Enter atomic proposition: ").strip()
                kripke.add_atomic_prop(state, prop)
                print(f"✅ Added proposition '{prop}' to state '{state}'")

            elif choice == '5':
                state = input("Enter state name: ").strip()
                var_name = input("Enter variable name: ").strip()
                try:
                    value = float(input("Enter variable value: ").strip())
                    kripke.set_variable(state, var_name, value)
                    print(f"✅ Set {var_name} = {value} in state '{state}'")
                except ValueError:
                    print("❌ Invalid numeric value")

            elif choice == '6':
                print("\n📋 Current Kripke Structure:")
                print("-" * 30)
                print(kripke)

            elif choice == '7':
                if not kripke.states:
                    print("❌ No states defined. Build structure first.")
                    continue

                ltl_formula = input("Enter LTL formula: ").strip()
                chi_formula = input("Enter Chi formula (or press Enter for none): ").strip()

                if not chi_formula:
                    chi_formula = None

                verbose = input("Verbose output? (y/n): ").strip().lower() == 'y'

                print("\n🔍 Running model checker...")
                result = checker.check_model(kripke, ltl_formula, chi_formula, verbose)

                if result['satisfies']:
                    print("✅ PROPERTY SATISFIED")
                else:
                    print("❌ PROPERTY VIOLATED")

                show_details = input("Show detailed results? (y/n): ").strip().lower() == 'y'
                if show_details:
                    checker.print_detailed_results()

            elif choice == '8':
                print("\nExample structures:")
                print("1. Traffic Light")
                print("2. Mutex System")
                print("3. Producer-Consumer")

                ex_choice = input("Choose example (1-3): ").strip()

                if ex_choice == '1':
                    kripke = create_default_example()
                    print("✅ Loaded traffic light example")
                elif ex_choice == '2':
                    kripke = create_mutex_example()
                    print("✅ Loaded mutex example")
                elif ex_choice == '3':
                    kripke = create_producer_consumer_example()
                    print("✅ Loaded producer-consumer example")
                else:
                    print("❌ Invalid choice")

            elif choice == '9':
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please enter 1-9.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def create_mutex_example():
    """Create mutual exclusion example."""
    kripke = KripkeStructure()

    states = ['n0_n1', 'c0_n1', 'n0_c1']
    for state in states:
        kripke.add_state(state)

    kripke.add_initial_state('n0_n1')

    kripke.add_transition('n0_n1', 'c0_n1')
    kripke.add_transition('n0_n1', 'n0_c1')
    kripke.add_transition('c0_n1', 'n0_n1')
    kripke.add_transition('n0_c1', 'n0_n1')

    kripke.add_atomic_prop('c0_n1', 'critical0')
    kripke.add_atomic_prop('n0_c1', 'critical1')

    kripke.set_variable('n0_n1', 'time', 1)
    kripke.set_variable('c0_n1', 'time', 5)
    kripke.set_variable('n0_c1', 'time', 3)

    return kripke


def create_producer_consumer_example():
    """Create producer-consumer example."""
    kripke = KripkeStructure()

    states = ['empty', 'item1', 'item2', 'full']
    for state in states:
        kripke.add_state(state)

    kripke.add_initial_state('empty')

    kripke.add_transition('empty', 'item1')
    kripke.add_transition('item1', 'item2')
    kripke.add_transition('item2', 'full')
    kripke.add_transition('full', 'item2')
    kripke.add_transition('item2', 'item1')
    kripke.add_transition('item1', 'empty')

    kripke.add_atomic_prop('empty', 'buffer_empty')
    kripke.add_atomic_prop('full', 'buffer_full')
    kripke.add_atomic_prop('item1', 'has_items')
    kripke.add_atomic_prop('item2', 'has_items')

    kripke.set_variable('empty', 'utilization', 0)
    kripke.set_variable('item1', 'utilization', 1)
    kripke.set_variable('item2', 'utilization', 2)
    kripke.set_variable('full', 'utilization', 3)

    return kripke


def main():
    """Main entry point."""
    print("🔍 LTL+LimAvg Model Checker")
    print("Anthropic Claude Implementation")
    print("=" * 50)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            # Run test suite
            from tests.test_examples import run_all_tests
            run_all_tests()
        elif sys.argv[1] == 'interactive':
            # Interactive mode
            interactive_mode()
        else:
            print("Usage: python main.py [test|interactive]")
            print("  test        - Run test suite")
            print("  interactive - Interactive mode")
            print("  (no args)   - Run default example")
    else:
        # Default example
        run_default_example()



def geometric_verification_example():
    """Example using geometric convex hull approach."""

    print("🔺 Geometric Verification with Convex Hulls")
    print("=" * 50)

    # Create simple system
    kripke = KripkeStructure()
    kripke.add_state("s0", {"x": 2, "y": 6})
    kripke.add_state("s1", {"x": 4, "y": 2})
    kripke.add_state("s2", {"x": 1, "y": 8})

    kripke.add_transition("s0", "s1")
    kripke.add_transition("s1", "s2")
    kripke.add_transition("s2", "s0")

    kripke.set_initial_state("s0")

    # Test geometric verification
    from core.convex_hull_checker import ConvexHullChecker

    checker = ConvexHullChecker()
    scc = {"s0", "s1", "s2"}
    variable_values = {
        "s0": {"x": 2, "y": 6},
        "s1": {"x": 4, "y": 2},
        "s2": {"x": 1, "y": 8}
    }

    # Test different constraints
    test_cases = [
        "x >= 2",
        "x >= 5",
        "x >= 2 and y >= 3",
        "x >= 5 or y >= 5"
    ]

    for chi_formula in test_cases:
        print(f"\nTesting: {chi_formula}")
        satisfied, witness = checker.verify_chi_constraint(scc, variable_values, chi_formula)
        result = "✅ SATISFIED" if satisfied else "❌ VIOLATED"
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
    # geometric_verification_example()