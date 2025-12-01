# LTLim Model Checker - How to Use Guide

## Overview

The **LTLim Model Checker** is a tool for verifying **Linear Temporal Logic with Limit-Average (LTLim)** formulas on **Quantitative Kripke Structures (QKS)**. It combines qualitative (LTL) and quantitative (limit-average) verification, offering a powerful way to analyze systems with both temporal and average-cost constraints.

---

## Installation

### Prerequisites
To run the model checker, you need the following:
* **Python 3.8+**
* **WSL (Windows Subsystem for Linux)** with Ubuntu (required for the Spot library)
* **Spot library** installed in WSL
* **Z3 solver** installed in WSL

### Installation
1.  **Clone or download** the project files.
2.  **Install Python dependencies** (run in your main terminal environment):
    ```bash
    pip install ply
    ```

3.  **Set up WSL components** (run in the **WSL/Ubuntu terminal**):
    ```bash
    # Update package list
    sudo apt-get update
    
    # Install Spot command-line tool
    sudo apt-get install spot
    
    # Install Python Spot bindings (for the model checker)
    pip install spot
    
    # Install Z3 Python bindings
    pip install z3-solver
    ```

---

## How to use

### 1. Prepare Input Files

#### Create a Quantitative Kripke Structure (`system.json`)
The QKS defines the system's states, transitions, boolean variable labeling, and numerical costs associated with each state.

```json
{
    "states": ["s0", "s1", "s2", "s3"],
    "init_state": "s0",
    "edges": [
        ["s0", "s1"],
        ["s0", "s2"],
        ["s1", "s0"],
        ["s1", "s2"],
        ["s1", "s3"],
        ["s2", "s1"],
        ["s2", "s3"],
        ["s3", "s0"],
        ["s3", "s2"]
    ],
    "boolean_vars": ["p", "q", "r"],
    "logical_formulas": {
        "s0": ["p"],
        "s1": ["q"],
        "s2": ["p", "r"],
        "s3": ["q", "r"]
    },
    "numeric_values": {
        "s0": {"x": 1.0, "y": 2.0, "z": 0.5},
        "s1": {"x": 3.0, "y": 1.0, "z": 1.5},
        "s2": {"x": 2.0, "y": 3.0, "z": 0.8},
        "s3": {"x": 4.0, "y": 0.5, "z": 2.0}
    }
}
````
### 2\. Run the Model Checker

Execute `main.py` using Python, providing the paths to your QKS and formula files.

```bash
# Basic usage
python main.py "LimInfAvg(x) > 2.0"

# With custom QKS file
python main.py --qks-file system.json "G(p → F q)"

# Batch verification
python main.py --batch formulas.txt

# Quiet mode
python main.py --quiet "F p"

# Help
python main.py --help
```

