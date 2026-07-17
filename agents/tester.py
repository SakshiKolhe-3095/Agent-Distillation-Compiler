"""
Stopgap tester agent — runs generated code + test cases via subprocess.
NOTE: Faiza is building a hardened Docker-based sandbox on faiza/sandbox-executor.
Once that PR merges into main, swap the subprocess call below for her sandbox
executor. Flagged here so nobody's surprised when that file changes.
"""
import subprocess
import tempfile
import os

def run_tests(code: str, test_code: str, timeout: int = 10) -> tuple[bool, str]:
    full_script = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_script)
        path = f.name

    try:
        result = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        passed = result.returncode == 0
        error = result.stderr if not passed else ""
        return passed, error
    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    finally:
        os.remove(path)