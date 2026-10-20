"""Keep long evidence identifiers from triggering whole-document PDF scaling."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_print_tables_wrap_hashes_without_reducing_type_size():
    css = (ROOT / "assets/print.css").read_text()
    table = re.search(r"^table\s*\{([^}]+)\}", css, re.M).group(1)
    cells = re.search(r"^th, td\s*\{([^}]+)\}", css, re.M).group(1)
    body = re.search(r"^body\s*\{([^}]+)\}", css, re.M).group(1)
    assert "table-layout: fixed" in table
    assert "width: 100%" in table
    assert "overflow-wrap: anywhere" in cells
    assert "font-size: 9pt" in table
    assert "font-size: 10.5pt" in body
