import json
import subprocess
import shlex
from typing import Dict, Any

from core.platform_detector import TOOL_STRATEGY

class ToolExecutor:
    @staticmethod
    def create():
        if TOOL_STRATEGY == "wsl_bridge":
            return WSLBridgeExecutor()
        else:  # native_linux
            return NativeLinuxExecutor()

    def execute_spot(self, script_path: str, formula: str) -> Dict[str, Any]:
        raise NotImplementedError

    def execute_z3(self, script_path: str, python_exe: str,
                   problem_data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class WSLBridgeExecutor(ToolExecutor):

    def execute_spot(self, script_path: str, formula: str) -> Dict[str, Any]:
        try:
            escaped_formula = shlex.quote(formula)

            cmd = f'wsl python3 {script_path} {escaped_formula}'

            print(f"Executing via WSL: python3 {script_path} ...")

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    'success': False,
                    'error': f"Spot failed: {result.stderr}"
                }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Spot timeout (30s)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def execute_z3(self, script_path: str, python_exe: str,
                   problem_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if 'limit_avg_formula' in problem_data:
                formula = problem_data['limit_avg_formula']
                escaped_formula = (
                    formula
                    .replace('<', '^<')
                    .replace('>', '^>')
                    .replace('&', '^&')
                    .replace('|', '^|')
                )
                problem_data['limit_avg_formula'] = escaped_formula

            json_str = json.dumps(problem_data).replace('"', '\\"')

            cmd = f'wsl {python_exe} {script_path} "{json_str}"'

            print(f"Executing Z3 via WSL: {python_exe} {script_path} ...")

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    'success': False,
                    'error': f"Z3 failed: {result.stderr}"
                }

        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Z3 timeout (30s)'}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class NativeLinuxExecutor(ToolExecutor):

    def execute_spot(self, script_path: str, formula: str) -> Dict[str, Any]:
        try:
            escaped_formula = shlex.quote(formula)

            cmd = ['python3', script_path, escaped_formula]

            print(f"Executing natively: python3 {script_path} ...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    'success': False,
                    'error': f"Spot failed: {result.stderr}"
                }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def execute_z3(self, script_path: str, python_exe: str,
                   problem_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            json_str = json.dumps(problem_data)

            if not python_exe or python_exe == "python3":
                python_exe = "python3"

            cmd = [python_exe, script_path, json_str]

            print(f"Executing Z3 natively: {python_exe} {script_path} ...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    'success': False,
                    'error': f"Z3 failed: {result.stderr}"
                }

        except Exception as e:
            return {'success': False, 'error': str(e)}