# GitHub Copilot Instructions

## Purpose

These instructions provide project context and guidelines that AI assistants (such as GitHub Copilot) must follow when generating code, scripts, or making changes within this repository.

The goal is to ensure generated code works correctly with repository policies and that automated actions can be safely approved.

---

# File Path Rules

When referencing files or directories inside this repository, **always use explicit repository-relative paths**.

## Use Explicit Paths

Always write out the full path relative to the repository root.

Example (Correct):

```
./src/utils/fileHelper.ts
./scripts/build.sh
./config/app.config.json
```

## Do Not Use Environment Variables for Repository Paths

Avoid constructing paths using environment variables or temporary variables.

Examples (Incorrect):

```
$PROJECT_ROOT/src/utils/fileHelper.ts
$BASE_DIR/scripts/build.sh
${PWD}/src/file.ts
$TEMP_DIR/output.txt
```

## Avoid Dynamic Path Construction

Do not generate or compute repository paths dynamically if it prevents Copilot from clearly recognizing that the file is inside the repository.

Example (Incorrect):

```bash
BASE_PATH="./src"
FILE="$BASE_PATH/utils/helper.ts"
```

Example (Correct):

```bash
./src/utils/helper.ts
```

---

# Temporary Directory Rules

Copilot **must NOT write files into system temporary directories** such as:

```
/tmp
/tmp/*
/var/tmp
C:\Temp
C:\Windows\Temp
```

## Correct Temporary File Handling

If temporary files are needed, Copilot should:

1. **Use an existing temporary directory inside the repository**, or
2. **Create a temporary directory inside the repository**

Recommended locations:

```
./.tmp/
./tmp/
./temp/
./build/tmp/
```

Example:

Correct:

```
./tmp/generated-config.json
./temp/build-output.log
```

If the directory does not exist, Copilot may create it inside the project:

```
./tmp/
```

and then store temporary files there.

---

# Building Applications

When compiling or building applications within this repository, Copilot **must only use the dedicated build script**. 

## Dedicated Build Script

Always use the following script to build applications:

```bash
./build_all.sh
```

## Building Specific Applications

If you need to recompile only specific applications, do not guess the build commands or use standard toolchain commands directly. Instead, run the build script with the `--help` flag to determine the correct arguments:

```bash
./build_all.sh --help
```

---

# Reason for These Rules

GitHub Copilot's auto-approval and safety systems must be able to clearly determine that file operations occur **inside the project directory**. 

Using:

* system temporary directories
* environment variables
* dynamically constructed paths

may prevent the system from verifying file locations and can block automatic approvals. Additionally, strict adherence to the build script ensures consistency across all development environments.

---

# Summary

Copilot must follow these rules:

* Always use **explicit repository-relative paths**
* Do **not** use environment variables for repository paths
* Avoid **dynamic path generation**
* Do **not write to system temp directories**
* Use or create a **temporary directory inside the repository** instead
* Ensure all referenced paths clearly reside **inside the repository**
* **Only** build applications using `./build_all.sh`
* Use `./build_all.sh --help` to determine the correct commands for recompiling specific applications