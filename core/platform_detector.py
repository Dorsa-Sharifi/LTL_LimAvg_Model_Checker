import platform
import subprocess
from typing import Dict, Any


def detect_platform() -> Dict[str, Any]:
    system = platform.system().lower()
    release = platform.release().lower()

    result = {
        'system': system,
        'release': release,
        'is_wsl': False,
        'wsl_available': False,
        'python_exe': 'python3',
    }

    if system == 'linux' and 'microsoft' in release:
        result['is_wsl'] = True
        result['mode'] = 'wsl_linux'  # Inside WSL (treat as Linux)

    elif system == 'windows':
        result['python_exe'] = 'python'
        try:
            subprocess.run(['wsl', '--status'],
                           capture_output=True, timeout=2, check=False)
            result['wsl_available'] = True
            result['mode'] = 'windows_wsl'
        except:
            result['mode'] = 'windows_native'  # Windows without WSL (not supported)

    elif system == 'linux':
        result['mode'] = 'native_linux'

    else:
        result['mode'] = 'unsupported'

    return result


def get_platform_string(platform_info: Dict[str, Any]) -> str:
    mode = platform_info.get('mode', 'unknown')

    if mode == 'windows_wsl':
        return "Windows with WSL"
    elif mode == 'wsl_linux':
        return "WSL (Linux)"
    elif mode == 'native_linux':
        return f"Linux ({platform_info['release']})"
    elif mode == 'windows_native':
        return "Windows (no WSL)"
    else:
        return "Unsupported platform"


def get_tool_strategy(platform_info: Dict[str, Any]) -> str:
    mode = platform_info.get('mode', 'unknown')

    if mode == 'windows_wsl':
        return "wsl_bridge"
    else:
        return "native_linux"


PLATFORM = detect_platform()
PLATFORM_STR = get_platform_string(PLATFORM)
TOOL_STRATEGY = get_tool_strategy(PLATFORM)

def print_platform_info():
    print(f"Platform: {PLATFORM_STR}")
    print(f"System: {PLATFORM['system']}")
    print(f"Release: {PLATFORM['release']}")
    print(f"Tool Strategy: {TOOL_STRATEGY}")
    print(f"Python: {PLATFORM['python_exe']}")