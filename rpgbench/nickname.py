"""Nickname handling: display form, safe file-name part and the prompt."""
import re

DEFAULT_NICKNAME = "player"
MAX_LEN = 32

_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")
_RESERVED = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + ["COM%d" % i for i in range(1, 10)]
    + ["LPT%d" % i for i in range(1, 10)]
)


def display_nickname(raw):
    """Strip surrounding whitespace, truncate to MAX_LEN, default if empty."""
    text = "" if raw is None else str(raw).strip()
    text = text[:MAX_LEN].strip()
    return text or DEFAULT_NICKNAME


def file_part(display):
    """Return a file-name-safe form of a nickname (only [A-Za-z0-9_-])."""
    text = _UNSAFE.sub("_", str(display))[:MAX_LEN]
    if not text or not text.strip("_"):
        return DEFAULT_NICKNAME
    if text.upper() in _RESERVED:
        if len(text) < MAX_LEN:
            text += "_"
        else:
            text = text[:-1] + "_"
    return text


def resolve_nickname(cli_value, stdin, input_fn, out=None):
    """Return (display_name, file_part).

    The user is prompted only when no name was given on the command line and
    stdin is an interactive terminal. EOF, I/O errors or empty input fall back
    to the default nickname.
    """
    raw = cli_value
    if raw is None:
        interactive = False
        try:
            interactive = bool(stdin.isatty())
        except Exception:
            interactive = False
        if interactive:
            if out is not None:
                out("Enter a nickname for your character "
                    "(press Enter for '%s'):" % DEFAULT_NICKNAME)
            try:
                raw = input_fn("> ")
            except (EOFError, OSError):
                raw = None
    display = display_nickname(raw)
    return display, file_part(display)
