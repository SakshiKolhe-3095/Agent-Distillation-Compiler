# """
# Docker-based sandbox executor.
# Runs untrusted code (from Coder agent) + its pytest tests inside an isolated
# container, captures result, returns pass/fail + logs to the pipeline.
# """

# import docker
# import tempfile
# import os
# import textwrap

# SANDBOX_IMAGE = "adc-sandbox:latest"
# TIMEOUT_SECONDS = 30


# def run_in_sandbox(code: str, test_code: str) -> dict:
#     """
#     code: the solution code produced by the Coder agent
#     test_code: pytest test cases produced by the Tester agent
#     Returns: {"passed": bool, "stdout": str, "stderr": str, "exit_code": int}
#     """
#     client = docker.from_env()

#     with tempfile.TemporaryDirectory() as tmpdir:
#         # write solution + test file into shared temp folder
#         with open(os.path.join(tmpdir, "solution.py"), "w") as f:
#             f.write(code)
#         with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
#             # f.write(test_code)
#             f.write("from solution import *\n\n" + test_code)

#         try:
#             container = client.containers.run(
#                 SANDBOX_IMAGE,
#                 command="python test_solution.py",
#                 volumes={tmpdir: {"bind": "/sandbox", "mode": "rw"}},
#                 working_dir="/sandbox",
#                 network_disabled=True,      # no internet access for untrusted code
#                 mem_limit="256m",           # hard memory cap
#                 memswap_limit="256m",       # disable swap, prevent OOM workaround
#                 cpu_quota=50000,            # 50% of one CPU core (100000 = 1 core)
#                 cpu_period=100000,
#                 pids_limit=64,              # cap process/thread count (fork-bomb protection)
#                 security_opt=["no-new-privileges"],
#                 read_only=True,             # root filesystem read-only, only /sandbox is writable
#                 detach=True,
#             )
#             result = container.wait(timeout=TIMEOUT_SECONDS)
#             logs = container.logs().decode("utf-8", errors="replace")
#             exit_code = result.get("StatusCode", 1)
#             container.remove(force=True)

#             return {
#                 "passed": exit_code == 0,
#                 "stdout": logs,
#                 "stderr": "",
#                 "exit_code": exit_code,
#             }
#         except Exception as e:
#             try:
#                 container.kill()
#             except Exception:
#                 pass
#             return {
#                 "passed": False,
#                 "stdout": "",
#                 "stderr": f"Sandbox execution failed or timed out: {e}",
#                 "exit_code": -1,
#             }
#         finally:
#             try:
#                 container.remove(force=True)
#             except Exception:
#                 pass


# if __name__ == "__main__":
#     # quick manual test
#     sample_code = "def add(a, b):\n    return a + b\n"
#     sample_test = textwrap.dedent("""
#         from solution import add
#         def test_add():
#             assert add(2, 3) == 5
#     """)
#     print(run_in_sandbox(sample_code, sample_test))

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

# Capabilities dropped by default in Docker are already substantial, but we
# explicitly drop ALL and add back nothing — untrusted code needs no special
# Linux capabilities (no raw sockets, no ptrace, no mount, no module loading).
CAP_DROP = ["ALL"]


def run_in_sandbox(code: str, test_code: str, timeout_seconds: int = TIMEOUT_SECONDS) -> dict:
    """
    code: the solution code produced by the Coder agent
    test_code: pytest test cases produced by the Tester agent
    timeout_seconds: max wall-clock time before the container is force-killed
    Returns: {"passed": bool, "stdout": str, "stderr": str, "exit_code": int}
    """
    client = docker.from_env()
    container = None
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(tmpdir, "test_solution.py"), "w", encoding="utf-8") as f:
            f.write("from solution import *\n\n" + test_code)
        try:
            container = client.containers.run(
                SANDBOX_IMAGE,
                command="python test_solution.py",
                volumes={tmpdir: {"bind": "/sandbox", "mode": "rw"}},
                working_dir="/sandbox",
                network_disabled=True,          # no internet access for untrusted code
                mem_limit="256m",               # hard memory cap
                memswap_limit="256m",           # disable swap, prevent OOM workaround
                cpu_quota=50000,                # 50% of one CPU core (100000 = 1 core)
                cpu_period=100000,
                pids_limit=64,                  # cap process/thread count (fork-bomb protection)
                cap_drop=CAP_DROP,               # drop all Linux capabilities (blocks raw
                                                  # sockets, ptrace, mount, module loading, etc.)
                security_opt=["no-new-privileges"],  # can't gain new privileges via setuid etc.
                read_only=True,                 # root filesystem read-only, only /sandbox is writable
                tmpfs={"/tmp": "size=32m,noexec"},   # writable scratch space, but no exec allowed there
                environment={"PYTHONDONTWRITEBYTECODE": "1"},
                detach=True,
            )
            result = container.wait(timeout=timeout_seconds)
            logs = container.logs().decode("utf-8", errors="replace")
            exit_code = result.get("StatusCode", 1)
            return {
                "passed": exit_code == 0,
                "stdout": logs,
                "stderr": "",
                "exit_code": exit_code,
            }
        except Exception as e:
            stderr_msg = f"Sandbox execution failed or timed out after {timeout_seconds}s: {e}"
            return {
                "passed": False,
                "stdout": "",
                "stderr": stderr_msg,
                "exit_code": -1,
            }
        finally:
            if container is not None:
                try:
                    container.kill()
                except Exception:
                    pass
                try:
                    container.remove(force=True)
                except Exception:
                    pass


if __name__ == "__main__":
    # quick manual test
    sample_code = "def add(a, b):\n    return a + b\n"
    sample_test = textwrap.dedent("""
        from solution import add
        def test_add():
            assert add(2, 3) == 5
    """)
    print(run_in_sandbox(sample_code, sample_test))

    # timeout test — infinite loop should get killed at the timeout, not hang forever
    infinite_code = "def add(a, b):\n    while True:\n        pass\n"
    print(run_in_sandbox(infinite_code, sample_test, timeout_seconds=5))