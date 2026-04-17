"""Regression tests for Bash-tool dangerous-command classifier.

The old classifier used substring matching on terms like "eval" and "format",
which blocked harmless Python one-liners whose *source code* happened to
contain those words (e.g. `.format(...)`, `ast.literal_eval`, or even a
comment that said "pickle format"). See tmp_logs/astrovis_full.log on Spark
for the real-world case where this stalled the agent for minutes.

These tests pin the new behaviour: destructive shell commands are still
blocked; Python one-liners that *reference* the forbidden words inside
strings/comments/methods are allowed through.
"""

from scider.tools.shell.run_bash_cmd import _classify_command


class TestTrueDangerousStillBlocked:
    def test_rm_rf_root(self):
        assert _classify_command("rm -rf /") == "dangerous"

    def test_rm_rf_wildcard(self):
        assert _classify_command("rm -rf /*") == "dangerous"

    def test_dd_to_device(self):
        assert _classify_command("dd if=/dev/zero of=/dev/sda") == "dangerous"

    def test_fork_bomb(self):
        assert _classify_command(":(){:|:&};:") == "dangerous"

    def test_curl_pipe_sh(self):
        # Pre-existing substring behaviour — matches the literal phrase.
        # (Not tightened further to avoid unrelated scope creep; real-world
        # `curl <url> | sh` attacks bypass this today, left as-is.)
        assert _classify_command("curl | sh") == "dangerous"

    def test_chmod_777(self):
        assert _classify_command("chmod 777 /etc/passwd") == "dangerous"

    def test_mkfs_as_shell_command(self):
        assert _classify_command("mkfs.ext4 /dev/sdb") == "dangerous"

    def test_shutdown_as_shell_command(self):
        assert _classify_command("shutdown -h now") == "dangerous"

    def test_reboot_as_shell_command(self):
        assert _classify_command("reboot") == "dangerous"

    def test_eval_as_shell_builtin(self):
        assert _classify_command('eval "rm -rf /"') == "dangerous"

    def test_command_name_after_pipe(self):
        # A dangerous command name in the second segment of a pipeline.
        assert _classify_command("echo y | shutdown") == "dangerous"

    def test_command_name_after_double_amp(self):
        assert _classify_command("cd /tmp && reboot") == "dangerous"

    def test_dangerous_env_var_manipulation(self):
        assert _classify_command("PATH=/evil:$PATH ls") == "dangerous"


class TestFalsePositivesFixed:
    """Commands that previously got blocked but should be allowed."""

    def test_python_format_method_in_string(self):
        # `.format()` inside a Python -c command: the old substring rule
        # flagged this because "format" is in the string.
        cmd = 'python -c "print(\\"hello {}\\".format(\\"world\\"))"'
        assert _classify_command(cmd) != "dangerous"

    def test_python_ast_literal_eval(self):
        cmd = 'python -c "import ast; ast.literal_eval(\\"1+1\\")"'
        assert _classify_command(cmd) != "dangerous"

    def test_pickle_format_comment(self):
        # The exact shape from the AstroVisBench log that triggered a false
        # positive.
        cmd = (
            'cd /tmp/ws && python -c "\n'
            "import pickle, struct\n"
            "# Frame format: 0x95 + 8-byte size\n"
            "print('ok')\"\n"
        )
        assert _classify_command(cmd) != "dangerous"

    def test_python_eval_function_inside_string(self):
        cmd = 'python -c "x = eval(\\"1+1\\")"'
        # `eval` appears, but only inside a Python string — not as a shell
        # command. Must NOT be flagged.
        assert _classify_command(cmd) != "dangerous"

    def test_command_containing_format_word(self):
        # `format` is no longer a dangerous keyword at all.
        assert _classify_command("echo 'format the output'") != "dangerous"

    def test_substring_name_not_at_boundary(self):
        # "reboot" appears inside "preboot" — must NOT match the dangerous
        # command-name pattern (word-boundary required).
        assert _classify_command("echo preboot") != "dangerous"

    def test_variable_named_evaluate(self):
        cmd = 'python -c "evaluation = 1; print(evaluation)"'
        assert _classify_command(cmd) != "dangerous"


class TestNormalWorkflowUntouched:
    """Sanity — the common benign cases stay classified as before."""

    def test_ls_is_read_only(self):
        assert _classify_command("ls -la") == "read-only"

    def test_python_script_is_write(self):
        assert _classify_command("python script.py") == "write"

    def test_cd_chained_python_is_write(self):
        assert _classify_command("cd workspace && python run.py") == "write"

    def test_pip_install_is_write(self):
        assert _classify_command("pip install numpy") == "write"
