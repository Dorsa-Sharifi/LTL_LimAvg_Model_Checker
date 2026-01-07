import argparse
import sys
import json
import os
from typing import Set, Tuple


from core.platform_detector import PLATFORM_STR, print_platform_info
from core.config import get_config
from core.QuantitativeKripkeStructure import QuantitativeKripkeStructure
from core.ltllim_parser import EnhancedLTLimProcessor



def load_qks_from_file(filename: str) -> QuantitativeKripkeStructure:
    try:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            sys.exit(1)

        with open(filename, 'r') as f:
            data = json.load(f)

        edges: Set[Tuple[str, str]] = set()
        for edge in data['edges']:
            if len(edge) == 2:
                edges.add((str(edge[0]), str(edge[1])))
            else:
                raise ValueError(f"Invalid edge format: {edge}")

        return QuantitativeKripkeStructure(
            states=set(str(s) for s in data['states']),
            init_state=str(data['init_state']),
            edges=edges,
            boolean_vars=set(str(v) for v in data['boolean_vars']),
            logical_formulas={str(k): set(str(v) for v in vs) for k, vs in data['logical_formulas'].items()},
            numeric_values={str(k): {str(vk): float(vv) for vk, vv in v.items()} for k, v in
                            data['numeric_values'].items()}
        )
    except Exception as e:
        print(f"Error loading QKS from {filename}: {e}")
        sys.exit(1)


def create_example_qks() -> QuantitativeKripkeStructure:
    """Create the default example QKS"""
    return QuantitativeKripkeStructure(
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


def verify_single(processor, formula: str, quiet: bool = False):
    """Verify a single formula"""
    if not quiet:
        print(f"Verifying: {formula}")
        print("=" * 60)

    try:
        results = processor.complete_pipeline_with_product(formula)

        if not results:
            print("No results returned")
            sys.exit(1)

        satisfiable = False
        for i, (chi, xi, product, result) in enumerate(results):
            if not quiet:
                print(f"\nDisjunct {i + 1}:")
                print(f"  Limit-average: {chi}")
                print(f"  LTL: {xi}")
                print(f"  Fair MSCCs: {result['fair_msccs_exist']}")
                print(f"  Satisfiable: {result.get('limit_avg_satisfiable', 'N/A')}")

            if result.get('limit_avg_satisfiable', False):
                satisfiable = True

        if satisfiable:
            print(f"\n-RESULT: SATISFIABLE")
            return 0
        else:
            print(f"\n-RESULT: UNSATISFIABLE")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def verify_batch(processor, filename: str, quiet: bool = False):
    """Verify multiple formulas from a file"""
    try:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return 1

        with open(filename, 'r') as f:
            formulas = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return 1

    if not quiet:
        print(f"Batch verifying {len(formulas)} formulas")
        print("=" * 60)

    results = {}
    for formula in formulas:
        if not quiet:
            print(f"\nVerifying: {formula}")

        try:
            batch_results = processor.complete_pipeline_with_product(formula)
            satisfiable = any(
                result.get('limit_avg_satisfiable', False)
                for _, _, _, result in batch_results
            )
            results[formula] = satisfiable

            status = "-SAT" if satisfiable else "-UNSAT"
            print(f"  Result: {status}")

        except Exception as e:
            print(f"Error: {e}")
            results[formula] = False


    sat_count = sum(1 for r in results.values() if r)
    print(f"\nSummary: {sat_count}/{len(formulas)} satisfiable")


    return 0 if sat_count > 0 else 1


def main():
    parser = argparse.ArgumentParser(
        description=f'LTLim Model Checker - Running on {PLATFORM_STR}',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'''
Supported Platforms: Windows+WSL, Linux

Examples:
  python main.py "LimInfAvg(x) > 2.0"                    # Verify with example system
  python main.py --qks-file system.json "G(p → F q)"     # Verify with custom QKS
  python main.py --batch formulas.txt                    # Batch verify formulas
  python main.py --quiet "F p"                           # Quiet mode
  python main.py --config myconfig.json "F p"            # Use custom config
  python main.py --show-platform                         # Show platform information
  python main.py --setup                                 # Interactive setup
        '''
    )

    parser.add_argument('formula', nargs='?', help='LTLim formula to verify')
    parser.add_argument('--qks-file', help='JSON file containing Quantitative Kripke Structure')
    parser.add_argument('--batch', help='File containing multiple formulas (one per line)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    parser.add_argument('--config', help='Configuration file (JSON format)')
    parser.add_argument('--show-platform', action='store_true', help='Show platform information')
    parser.add_argument('--setup', action='store_true', help='Interactive setup wizard')

    args = parser.parse_args()


    if args.show_platform:
        print("=" * 60)
        print("PLATFORM INFORMATION")
        print("=" * 60)
        print_platform_info()

        config = get_config(args.config)
        config.print_summary()
        return 0

    if args.setup:
        config = get_config(args.config)
        print(f"\nInteractive Setup for {PLATFORM_STR}")
        print("-" * 40)

        print(f"\nCurrent paths:")
        print(f"1. Spot script: {config.get_path('spot_script')}")
        print(f"2. Z3 LP script: {config.get_path('z3_lp_script')}")
        print(f"3. Z3 Python: {config.get_path('z3_python', 'python3')}")

        print(f"\nEnter new paths (press Enter to keep current):")

        new_spot = input(f"Spot script [{config.get_path('spot_script')}]: ").strip()
        if new_spot:
            config.set_path('spot_script', new_spot)

        new_z3 = input(f"Z3 LP script [{config.get_path('z3_lp_script')}]: ").strip()
        if new_z3:
            config.set_path('z3_lp_script', new_z3)

        new_py = input(f"Z3 Python [{config.get_path('z3_python', 'python3')}]: ").strip()
        if new_py:
            config.set_path('z3_python', new_py)

        save_file = args.config or 'model_checker_config.json'
        config.save_config(save_file)
        print(f"\n✓ Configuration saved to {save_file}")
        return 0


    if not args.formula and not args.batch:
        print("Error: Either a formula or --batch file is required")
        print()
        parser.print_help()
        return 1

    if not args.quiet:
        print("=" * 60)
        print(f"LTLim Model Checker - {PLATFORM_STR}")
        print("=" * 60)
        print_platform_info()

    try:
        if args.qks_file:
            qks = load_qks_from_file(args.qks_file)
            if not args.quiet:
                print(f"Loaded QKS from {args.qks_file}")
        else:
            qks = create_example_qks()
            if not args.quiet:
                print("Using default example system")

        config = get_config(args.config)
        validation = config.validate()

        if not args.quiet and validation['warnings']:
            print("\nConfiguration warnings:")
            for warning in validation['warnings']:
                print(f"  ⚠ {warning}")

        if validation['errors']:
            print("\nConfiguration errors:")
            for error in validation['errors']:
                print(f"  ✗ {error}")
            sys.exit(1)

        if not args.quiet:
            print(f"\nInitializing processor...")

        processor = EnhancedLTLimProcessor(args.config, qks)

        if args.batch:
            exit_code = verify_batch(processor, args.batch, args.quiet)
        else:
            exit_code = verify_single(processor, args.formula, args.quiet)

        sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()