"""
Docker-based sandbox executor.
Runs untrusted code (from Coder agent) + its pytest tests inside an isolated
container, captures result, returns pass/fail + logs to the pipeline.
"""

import docker
import tempfile
import os
import textwrap

SANDBOX_IMAGE = "adc-sandbox:latest"
TIMEOUT_SECONDS = 30


def run_in_sandbox(code: str, test_code: str) -> dict:
    """
    code: the solution code produced by the Coder agent
    test_code: pytest test cases produced by the Tester agent
    Returns: {"passed": bool, "stdout": str, "stderr": str, "exit_code": int}
    """
    client = docker.from_env()

    with tempfile.TemporaryDirectory() as tmpdir:
        # write solution + test file into shared temp folder
        with open(os.path.join(tmpdir, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
            # f.write(test_code)
            f.write("from solution import *\n\n" + test_code)

        try:
            container = client.containers.run(
                SANDBOX_IMAGE,
                # command="pytest test_solution.py -q",
                command="python test_solution.py",
                volumes={tmpdir: {"bind": "/sandbox", "mode": "rw"}},
                working_dir="/sandbox",
                network_disabled=True,      # no internet access for untrusted code
                mem_limit="256m",           # basic resource cap (Sakshi hardens further later)
                detach=True,
            )
            result = container.wait(timeout=TIMEOUT_SECONDS)
            logs = container.logs().decode("utf-8", errors="replace")
            exit_code = result.get("StatusCode", 1)
            container.remove(force=True)

            return {
                "passed": exit_code == 0,
                "stdout": logs,
                "stderr": "",
                "exit_code": exit_code,
            }
        except Exception as e:
            return {
                "passed": False,
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
            }


if __name__ == "__main__":
    # quick manual test
    sample_code = "def add(a, b):\n    return a + b\n"
    sample_test = textwrap.dedent("""
        from solution import add
        def test_add():
            assert add(2, 3) == 5
    """)
    print(run_in_sandbox(sample_code, sample_test))