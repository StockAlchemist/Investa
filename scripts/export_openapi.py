#!/usr/bin/env python3
"""
Dump the FastAPI app's OpenAPI schema to stdout (or a file with --out).

Used by `npm run generate:api:offline` so TypeScript types can be regenerated
without needing a backend running on :8000.

Usage:
    python scripts/export_openapi.py > openapi.json
    python scripts/export_openapi.py --out openapi.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from server.main import app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="Write to this path instead of stdout.")
    args = parser.parse_args()

    schema = app.openapi()
    payload = json.dumps(schema, indent=2)

    if args.out:
        with open(args.out, "w") as f:
            f.write(payload)
        print(f"OpenAPI schema written to {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
