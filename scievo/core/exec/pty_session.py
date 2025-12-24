import os
import re
import time

import pexpect
from loguru import logger

from .manager import CommandContextManager, SessionManager
from .session_base import CommandContextBase, CommandState, SessionBase

_PROMPT = r"AISHELL(\w)> "  # \w shows current working directory
PROMPT = "AISHELL(.*)> "
CONT_PROMPT = "AISHELL_CONT> "  # unique PS2


class LocalShellContext(CommandContextBase):
    """Context object for managing non-blocking command execution in local PTY sessions."""

    @property
    def session(self) -> "LocalShellSession":
        """Get the session instance associated with this context."""
        s = SessionManager().get_session(self.session_id)
        if s is None:
            raise ValueError(f"Session with ID {self.session_id} not found")
        # check type
        if not isinstance(s, LocalShellSession):
            raise TypeError(f"Session with ID {self.session_id} is not a LocalShellSession")
        return s

    def _send_command(self):
        """Send the command to the PTY process."""
        self.session.process.sendline(self.command)

    def _cancel_command(self):
        """Cancel the command by sending Ctrl-C."""
        self.session.ctrlc(n=3)

    def get_input_output(self) -> str:
        """Get the input and output of the command. Used for AI conversation context."""
        raw_content = self.session.get_history(self.start_buffer_position)
        # regex find last match by PRMOPT
        matches = list(re.finditer(PROMPT, raw_content))
        last_match = matches[-1] if len(matches) > 0 else None
        if last_match:
            prompt_start = last_match.group(0)
        else:
            prompt_start = "AISHELL> "

        res = prompt_start + raw_content
        if self.end_buffer_position is not None:
            res = res[: self.end_buffer_position - self.start_buffer_position]
        res = res.removesuffix(prompt_start)
        return res

    def _monitor_completion(self):
        """Monitor thread that checks if command has completed."""
        start_time = time.time()
        check_interval = 0.25  # Check every 250ms

        while not self._stop_monitoring.is_set():
            try:
                # Check if we've exceeded timeout
                if self.timeout is not None and time.time() - start_time > self.timeout:
                    with self._lock:
                        self.state = CommandState.TIMEOUT
                        self.error = "Command timed out"
                        logger.error(f"Command timed out: {self.command}")
                    break

                # Try to match the prompt without blocking
                idx = self.session.process.expect([PROMPT, CONT_PROMPT], timeout=check_interval)

                # Command completed
                with self._lock:
                    if idx == 1:
                        # Continuation prompt - syntax error
                        self.session.process.sendcontrol("c")
                        self.session.process.expect(PROMPT, timeout=5)
                        self.state = CommandState.ERROR
                        self.error = "Command is incomplete (syntax error)"
                    else:
                        # Normal completion
                        self.state = CommandState.COMPLETED
                    # Record end buffer position
                    self.end_buffer_position = self.session.get_history_position()
                    logger.debug(f"Command completed with state: {self.state}")

                break  # Exit monitoring loop

            except pexpect.TIMEOUT:
                # No prompt yet, keep waiting
                continue
            except pexpect.EOF:
                with self._lock:
                    self.state = CommandState.ERROR
                    self.error = "Session terminated unexpectedly"
                    logger.error("Session terminated unexpectedly")
                break
            except Exception as e:
                with self._lock:
                    self.state = CommandState.ERROR
                    self.error = f"Error: {e}"
                    logger.error(f"Error monitoring command: {e}")
                break


class LocalShellSession(SessionBase):
    """Manages an interactive Bash shell session using pexpect."""

    def __init__(self, shell_path: str = "/bin/bash", cwd: str | None = None):
        super().__init__()

        session_env = {
            **os.environ,
            "TERM": "dumb",  # disable color
            "GIT_PAGER": "cat",  # disable paging for git commands
            "PAGER": "cat",  # disable paging for other tools
        }

        if "bash" in shell_path:
            sh_args = ["--norc", "--noprofile"]
        elif "zsh" in shell_path:
            raise NotImplementedError("Zsh shell is not yet supported")
        else:
            sh_args = []
        # Start a bare-bones bash
        self.process = pexpect.spawn(
            shell_path,
            sh_args,
            cwd=cwd,
            encoding="utf-8",
            echo=False,
            env=session_env,  # type: ignore
        )

        # Set up logging to capture all input/output
        self.process.logfile = self.history_buffer

        # Wait for the initial system prompt and then set ours
        try:
            self.process.expect([r"\$ ", r"# ", r"\(.*\)"], timeout=5)
        except pexpect.TIMEOUT:
            pass

        # Custom PS1/PS2 so we always know where we are
        self.process.sendline(f"export PS1='{_PROMPT}'")
        self.process.expect(PROMPT, timeout=5)
        self.process.sendline(f"export PS2='{CONT_PROMPT}'")
        self.process.expect(PROMPT, timeout=5)

        # Register this session with SessionManager and store the session ID
        self.session_id = SessionManager().register_session(self)

    def send_control(self, key: str):
        """
        Send Ctrl+<key> to the shell session.

        Args:
            key (str): The key to send with Ctrl (e.g., 'c', 'd', 'z').
        """
        if self.process.isalive():
            self.process.sendcontrol(key.lower())

    def ctrlc(self, n: int = 2):
        """
        Send Ctrl-C to the shell session.

        This method checks if the process is alive, sends a Ctrl-C signal to interrupt it,
        and then waits for the shell prompt to reappear within a 5-second timeout.

        Args:
            n (int, optional): Times to send Ctrl-C. Defaults to 2 to make sure the shell is interrupted.
        """
        assert n >= 1, "n must be at least 1"
        if self.process.isalive():
            for _ in range(n):
                self.send_control("c")
            self.process.expect(PROMPT, timeout=5)

    def terminate_session(self):
        if self.process.isalive():
            self.process.terminate(force=True)
        # Unregister from SessionManager
        SessionManager().unregister_session(self.session_id)

    def exec(self, command: str, timeout: float | None = None) -> LocalShellContext:
        """
        Execute command in non-blocking mode and return a context object.

        Args:
            command: The command to execute
            timeout: Maximum time to wait for command completion in seconds (default: None, no timeout)

        Returns:
            CommandContext: A context object that tracks the execution state

        Example:
            >>> session = LocalShellSession()
            >>> ctx = session.exec("sleep 5")
            >>> while ctx.is_running():
            ...     print("Still running...")
            ...     time.sleep(1)
            >>> print(ctx.get_output())
        """
        if self.is_running_command():
            raise RuntimeError("A command is already running in this session")
        ctx = LocalShellContext(self.session_id, command, timeout)
        # Register context with CommandContextManager and store the context ID
        context_id = CommandContextManager().register_context(ctx)
        with self._context_lock:
            self.current_context_id = context_id
        return ctx.start()
