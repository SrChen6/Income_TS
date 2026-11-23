#!/usr/bin/env python3
"""git clean filter: remove outputs from a notebook (stdin -> stdout), so plots or
printed values of the private data never reach the repository.

Enabled per clone with:
    git config filter.nbstrip.clean "python3 tools/strip_notebook.py"
    git config filter.nbstrip.smudge cat
(.gitattributes already maps *.ipynb to this filter.)"""

import json
import sys

nb = json.load(sys.stdin)
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    cell.get("metadata", {}).pop("execution", None)
json.dump(nb, sys.stdout, indent=1, ensure_ascii=False)
sys.stdout.write("\n")
