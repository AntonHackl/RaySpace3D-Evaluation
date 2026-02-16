---
description: How to run or recompile RaySpace3D applications
---

There are some irregularities when ssh-ing into this cluster: Sometimes we land directly in the container session and sometimes just on the host node.

When we are on just on the host node and not in the container, you MUST wrap the command using the `enroot` container.

**Command Wrapper:**
`enroot start --root --rw --mount /sc/home/anton.hackl:/sc/home/anton.hackl pyxis_rayspace bash -c "<command>"`

**Rules:**
1. Only use this wrapper when executing commands directly via `run_command` in the chat.
2. DO NOT include this wrapper in any source code files, scripts (e.g., `.sh` files), or Makefiles.
3. Ignore any errors related to the `/scratch` directory when running inside the container.

When we are directly inside the container, you need not wrap the command.