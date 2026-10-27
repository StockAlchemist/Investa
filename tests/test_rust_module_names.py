"""Every module attribute the Rust port reads at call time exists.

The ported functions look sibling functions, constants and imported names up
on their Python module when they run, so a patched attribute is still the one
used. That makes a name the Python code no longer mentions look unused to a
linter: `ruff --fix` once deleted `portfolio_analyzer`'s import of
`USFederalHolidayCalendar`, and the periodic-returns holiday filter failed
silently into its `except` branch. This scans the Rust sources for those
lookups and fails on any the module cannot answer.
"""

import glob
import importlib
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _reads():
    reads = {}
    for path in glob.glob(os.path.join(ROOT, "rust", "investa_core", "src", "*.rs")):
        text = open(path).read()
        helpers = re.findall(
            r"fn (\w+)\(py: Python<'_>\) -> PyResult<Bound<'_, PyModule>> \{\s*py\.import\(\"(\w+)\"\)", text
        )
        for helper, module in helpers:
            for name in re.findall(rf'{helper}\(py\)\?\s*\.getattr\("(\w+)"\)', text):
                reads.setdefault(module, set()).add(name)
        for module, name in re.findall(r'py\.import\("(\w+)"\)\?\s*\.getattr\("(\w+)"\)', text):
            reads.setdefault(module, set()).add(name)
        # `let m = py.import("module")?;` then `m.getattr("name")`, per function;
        # a name bound to two modules in one function is skipped as ambiguous.
        for body in re.split(r"\n(?:pub )?fn ", text):
            lets = {}
            for var, module in re.findall(r'let (\w+) = py\.import\("(\w+)"\)\?;', body):
                lets.setdefault(var, set()).add(module)
            for var, modules in lets.items():
                if len(modules) == 1:
                    for name in re.findall(rf'\b{var}\s*\.getattr\("(\w+)"\)', body):
                        reads.setdefault(next(iter(modules)), set()).add(name)
    return reads


def test_every_name_the_port_reads_exists():
    missing = []
    for module, names in sorted(_reads().items()):
        try:
            m = importlib.import_module(module)
        except ImportError:
            continue
        missing += [f"{module}.{n}" for n in sorted(names) if not hasattr(m, n)]
    assert not missing, missing
