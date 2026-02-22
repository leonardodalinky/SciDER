import shutil


def get_separator(margin: int = 4, char: str = "=") -> str:
    """
    Generate a separator that fits the terminal width.

    Args:
        margin: Number of characters to leave as margin (default: 4)
        char: Character to use for separator (default: '=')

    Returns:
        Separator string that fits terminal width
    """
    terminal_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    # Leave margin to be safe and ensure minimum width
    separator_width = max(terminal_width - margin, 10)
    return char * separator_width
