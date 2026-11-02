import re

from rpgbench import nickname as nn


class FakeStdin:
    def __init__(self, tty):
        self._tty = tty

    def isatty(self):
        return self._tty


def _no_input(prompt=""):
    raise AssertionError("input must not be called")


def test_cli_value_used_without_prompt():
    assert nn.resolve_nickname("Ana", FakeStdin(True), _no_input) == ("Ana", "Ana")


def test_non_tty_uses_default_without_prompt():
    assert nn.resolve_nickname(None, FakeStdin(False), _no_input) == (
        "player", "player")


def test_tty_prompts_and_uses_answer():
    seen = []
    result = nn.resolve_nickname(
        None, FakeStdin(True), lambda p="": "  Zed  ", out=seen.append)
    assert result == ("Zed", "Zed")
    assert seen


def test_tty_empty_input_defaults():
    assert nn.resolve_nickname(None, FakeStdin(True), lambda p="": "") == (
        "player", "player")


def test_tty_eof_defaults():
    def raiser(prompt=""):
        raise EOFError

    assert nn.resolve_nickname(None, FakeStdin(True), raiser) == (
        "player", "player")


def test_tty_oserror_defaults():
    def raiser(prompt=""):
        raise OSError("closed")

    assert nn.resolve_nickname(None, FakeStdin(True), raiser)[0] == "player"


def test_isatty_raising_is_non_interactive():
    class Broken:
        def isatty(self):
            raise ValueError("closed file")

    assert nn.resolve_nickname(None, Broken(), _no_input)[0] == "player"


def test_display_strip_and_truncate():
    assert nn.display_nickname("   hi  ") == "hi"
    assert nn.display_nickname("x" * 50) == "x" * 32
    assert nn.display_nickname("") == "player"
    assert nn.display_nickname("   ") == "player"
    assert nn.display_nickname(None) == "player"


def test_display_keeps_non_ascii():
    assert nn.display_nickname("José Ñandú") == "José Ñandú"


def test_file_part_sanitises_example():
    display = nn.display_nickname("  Ana/Maria: <x>  ")
    assert display == "Ana/Maria: <x>"
    part = nn.file_part(display)
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,32}", part)
    assert part == "Ana_Maria___x_"


def test_file_part_length_limit():
    assert len(nn.file_part("a" * 100)) == 32


def test_file_part_all_underscore_fallback():
    assert nn.file_part("///") == "player"
    assert nn.file_part("___") == "player"
    assert nn.file_part("éé") == "player"
    assert nn.file_part("") == "player"


def test_file_part_reserved_names():
    for name in ("CON", "nul", "Com1", "LPT9", "aux", "PRN"):
        part = nn.file_part(name)
        assert part.upper().rstrip("_") == name.upper()
        assert part.endswith("_")
        assert len(part) <= 32


def test_file_part_non_reserved_similar_names_untouched():
    assert nn.file_part("COM10") == "COM10"
    assert nn.file_part("console") == "console"
