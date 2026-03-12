---
description: Guidelines for executing terminal commands within Conda environments
applyTo: '.*' 
---

# Project Context & Coding Guidelines

## Conda Environment Command Execution

When suggesting, generating, or executing terminal commands that require a specific Conda environment, **strictly avoid using `conda run`**. 

Using `conda run -n <env_name> <command>` often suppresses standard output/error streams or buffers them in a way that makes debugging and reading outputs difficult.

Instead, you must explicitly activate the Conda environment first, and then run the command(s) consecutively.

### ❌ Incorrect (Do Not Use)
Do not use inline `conda run` commands:
`conda run -n my_env python main.py`
`conda run -n my_env pip install -r requirements.txt`

### ✅ Correct (Required Pattern)
Always separate the activation from the execution:
`conda activate my_env`
`python main.py`

`conda activate my_env`
`pip install -r requirements.txt`

**Key Takeaways for the AI:**
1. Always check if a command needs to be run in a Conda environment.
2. If yes, output `conda activate <env_name>` as the very first step.
3. Output the actual execution commands on the subsequent lines.