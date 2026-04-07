"""Tests for scider.core.task — background task system."""

import time

from scider.core.task import TaskManager, TaskStatus


class TestTaskManager:
    def test_spawn_and_complete(self):
        task_id = TaskManager.spawn_shell(command="echo hello", timeout=10)
        assert task_id.startswith("task_")

        task = TaskManager.wait_for_task(task_id, timeout=5)
        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.exit_code == 0

    def test_read_output(self):
        task_id = TaskManager.spawn_shell(command="echo test_output", timeout=10)
        TaskManager.wait_for_task(task_id, timeout=5)
        output = TaskManager.read_output(task_id)
        assert "test_output" in output

    def test_timeout_kills(self):
        task_id = TaskManager.spawn_shell(command="sleep 100", timeout=2)
        task = TaskManager.wait_for_task(task_id, timeout=5)
        assert task is not None
        assert task.status == TaskStatus.KILLED

    def test_stop_task(self):
        task_id = TaskManager.spawn_shell(command="sleep 100", timeout=60)
        time.sleep(0.5)  # let it start
        result = TaskManager.stop(task_id)
        assert "stopped" in result.lower()

        task = TaskManager.get(task_id)
        assert task.status == TaskStatus.KILLED

    def test_notifications(self):
        # Drain any leftover notifications from previous tests
        TaskManager.drain_notifications()

        task_id = TaskManager.spawn_shell(command="echo notify_test", timeout=10)
        TaskManager.wait_for_task(task_id, timeout=5)
        notifications = TaskManager.drain_notifications()
        assert len(notifications) >= 1
        assert any(task_id in n for n in notifications)

    def test_nonexistent_task(self):
        task = TaskManager.get("task_nonexistent")
        assert task is None
        output = TaskManager.read_output("task_nonexistent")
        assert "not found" in output.lower()

    def test_tail_output(self):
        task_id = TaskManager.spawn_shell(
            command="echo line1 && echo line2 && echo line3", timeout=10
        )
        TaskManager.wait_for_task(task_id, timeout=5)
        output = TaskManager.read_output(task_id, tail=1)
        assert "line3" in output
        assert "line1" not in output
