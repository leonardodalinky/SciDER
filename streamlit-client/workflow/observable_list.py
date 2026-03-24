"""Observable list that fires a callback on every append.

Used to hook into agent ``intermediate_state.append(...)`` calls so that the
Streamlit UI can receive real-time node completion notifications without
modifying any agent code.
"""

from typing import Any, Callable


class ObservableList(list):
    """A list subclass that invokes *on_append* after every ``append`` call."""

    def __init__(self, *args: Any, on_append: Callable[[Any], None] | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._on_append = on_append

    def append(self, item: Any) -> None:
        super().append(item)
        if self._on_append is not None:
            self._on_append(item)
