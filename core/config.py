import os
import json
from pathlib import Path
from typing import Optional, Union

from core.platform_detector import PLATFORM, PLATFORM_STR, get_tool_strategy


class MultiOSConfig:
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.platform = PLATFORM

        if config_file is not None:
            self.config_file = str(config_file) if isinstance(config_file, Path) else config_file
        else:
            self.config_file = None

        self.paths = {
            'spot_script': '',
            'z3_lp_script': '',
            'z3_python': 'python3',
        }

        if PLATFORM.get('mode') == 'windows_wsl':
            self.paths.update({
                'spot_script': '/home/otebook/ltl_to_nbw.py',
                'z3_lp_script': '/home/otebook/z3_lp_solver.py',
                'z3_python': '/home/otebook/.local/share/pipx/venvs/z3-solver/bin/python',
            })

        if self.config_file and os.path.exists(self.config_file):
            self.load_config(self.config_file)
        else:
            default_config = Path.home() / '.model_checker_config.json'
            if default_config.exists():
                self.load_config(str(default_config))


        self.validate()

    def load_config(self, config_file: Union[str, Path]):
        try:
            config_path = str(config_file) if isinstance(config_file, Path) else config_file

            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'paths' in data:
                for key in ['spot_script', 'z3_lp_script', 'z3_python']:
                    if key in data['paths'] and data['paths'][key]:
                        self.paths[key] = data['paths'][key]

            print(f"Loaded configuration from {config_path}")

        except Exception as e:
            print(f"Warning: Could not load config: {e}")

    def save_config(self, config_file: Optional[Union[str, Path]] = None):
        if config_file is not None:
            save_to = config_file
        elif self.config_file:
            save_to = self.config_file
        else:
            save_to = 'model_checker_config.json'

        save_to_str = str(save_to) if isinstance(save_to, Path) else save_to

        try:
            save_dir = os.path.dirname(save_to_str)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            data = {
                'platform': PLATFORM_STR,
                'paths': self.paths
            }

            json_str = json.dumps(data, indent=2)
            with open(save_to_str, 'w', encoding='utf-8') as f:
                f.write(json_str)

            print(f"Configuration saved to {save_to_str}")

        except Exception as e:
            print(f"Error saving config: {e}")

    def validate(self):
        errors = []
        warnings = []

        if self.platform.get('mode') == 'windows_wsl':
            if not self.platform.get('wsl_available', False):
                errors.append("WSL is not available. Required for Windows+WSL mode.")

        elif self.platform.get('mode') == 'windows_native':
            errors.append("Native Windows is not supported. Please install WSL.")

        for tool_name, path_key in [('Spot', 'spot_script'), ('Z3 LP', 'z3_lp_script')]:
            path = self.get_path(path_key)
            if path:
                expanded = os.path.expanduser(path)
                if not os.path.exists(expanded):
                    warnings.append(f"{tool_name} script not found: {expanded}")
            else:
                warnings.append(f"{tool_name} script path not configured")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def get_path(self, key: str, default: str = "") -> str:
        return self.paths.get(key, default)

    def set_path(self, key: str, value: str):
        self.paths[key] = value

    def print_summary(self):
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Platform: {PLATFORM_STR}")
        print(f"Strategy: {get_tool_strategy(self.platform)}")
        print(f"\nTool Paths:")
        for key, path in self.paths.items():
            print(f"  {key}: {path}")

        validation = self.validate()
        if validation['warnings']:
            print(f"\nWarnings:")
            for warning in validation['warnings']:
                print(f"{warning}")

        if validation['errors']:
            print(f"\nErrors:")
            for error in validation['errors']:
                print(f"{error}")
        else:
            print(f"\nConfiguration is valid")
        print()


_config_instance = None

def get_config(config_file: Optional[Union[str, Path]] = None):
    global _config_instance
    if _config_instance is None:
        _config_instance = MultiOSConfig(config_file)
    return _config_instance