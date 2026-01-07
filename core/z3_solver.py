from typing import Dict, Any, List
from core.config import get_config
from core.tools import ToolExecutor

class Z3Solver:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.executor = ToolExecutor.create()

    def check_feasibility(self,
                          cycle_vectors: List[Dict[str, float]],
                          variables: List[str],
                          limit_avg_formula: str) -> Dict[str, Any]:
        z3_script = self.config.get_path('z3_lp_script')
        z3_python = self.config.get_path('z3_python', 'python3')

        if not z3_script:
            return {'success': False, 'error': 'Z3 script path not configured'}

        problem_data = {
            'cycle_vectors': cycle_vectors,
            'variables': variables,
            'limit_avg_formula': limit_avg_formula
        }

        return self.executor.execute_z3(z3_script, z3_python, problem_data)