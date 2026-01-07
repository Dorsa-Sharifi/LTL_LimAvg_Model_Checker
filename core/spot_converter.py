from typing import Dict, Any

from core.config import get_config
from core.tools import ToolExecutor
class SpotConverter:

    def __init__(self, config=None):
        self.config = config or get_config()
        self.executor = ToolExecutor.create()

    def ltl_to_nbw(self, ltl_formula: str) -> Dict[str, Any]:
        spot_script = self.config.get_path('spot_script')
        if not spot_script:
            return {'success': False, 'error': 'Spot script path not configured'}

        return self.executor.execute_spot(spot_script, ltl_formula)

    def print_automaton_details(self, result, formula):
        if not result.get('success', False):
            print(f"Failed to convert: {formula}")
            print(f"   Error: {result.get('error', 'Unknown error')}")
            return

        print(f"\n{'=' * 60}")
        print(f"LTL Formula: {formula}")
        if 'formula_used' in result:
            print(f"   (Converted to: {result['formula_used']})")
        print(f"{'=' * 60}")
        print(f"States: {result['states']}")
        print(f"Edges: {result['edges']}")
        print(f"Acceptance: {result['acceptance']}")
        print(f"Deterministic: {result['is_deterministic']}")

        if 'hoa_format' in result:
            safe_name = formula.replace(' ', '_').replace('&', 'and').replace('|', 'or')
            filename = f"automaton_{safe_name}.hoa"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(result['hoa_format'])
            print(f"HOA format saved to: {filename}")