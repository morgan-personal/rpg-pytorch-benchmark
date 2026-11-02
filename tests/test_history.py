import re

import pytest

from rpgbench import history


def test_reference_table_shape():
    table = history.REFERENCE_MACHINES
    assert len(table) >= 6
    names = [name for name, _ in table]
    assert "Apollo Guidance Computer" in names
    pattern = re.compile(r"^#1 supercomputer of \d{4}")
    assert sum(1 for n in names if pattern.match(n)) >= 3
    values = [v for _, v in table]
    assert values == sorted(values)


@pytest.mark.parametrize("value", [1e-6, 1, 100, 10000, 1e9])
def test_compare_values(value):
    text = history.compare(value)
    assert text
    assert text.isascii()


@pytest.mark.parametrize("value", [0, -5, float("nan"), float("inf"), 1e300,
                                   None, "abc"])
def test_compare_never_raises(value):
    text = history.compare(value)
    assert text and text.isascii()


def test_compare_wording():
    assert history.compare(1e-6).startswith("about 1/")
    assert "about" in history.compare(100)
    assert history.compare(100).endswith("(CM-5)")
    assert history.compare(100).startswith("about 1.7x the")
