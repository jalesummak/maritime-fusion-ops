"""Audit tracked release files without printing any secret value."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
paths = [name for name in paths if name]
known_secrets = []
for private in [ROOT / ".env", ROOT / "dashboard/.env"]:
    if private.exists():
        for line in private.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("NASA_FIRMS_MAP_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                if len(value) > 12:
                    known_secrets.append(value)

problems = []
patterns = [r"gh[pousr]_[A-Za-z0-9]{25,}", r"github_pat_[A-Za-z0-9_]{30,}", r"/api/area/csv/[a-fA-F0-9]{32}/"]
for name in paths:
    path = ROOT / name
    if path.name == ".env" or any(part in {"node_modules", "project_checkpoint", "__pycache__"} for part in path.parts) or path.suffix in {".pkl", ".zip", ".rar", ".log"}:
        problems.append(f"Private or generated file tracked: {name}")
    if path.suffix.lower() in {".png", ".jpg"}:
        continue
    text = path.read_text(encoding="utf-8-sig")
    if any(value in text for value in known_secrets) or any(re.search(pattern, text) for pattern in patterns):
        problems.append(f"Potential secret in {name}; value suppressed")
    if path.suffix == ".py":
        ast.parse(text, filename=name)
    if path.suffix == ".ipynb":
        notebook = json.loads(text)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                if cell.get("outputs") or cell.get("execution_count") is not None:
                    problems.append(f"Notebook contains execution output: {name} cell {index}")
                ast.parse("".join(cell["source"]), filename=f"{name}:{index}")
if problems:
    raise SystemExit("\n".join(problems))
print(f"Release audit passed: {len(paths)} tracked files; no matching private key or token; notebook code parses and outputs are empty.")
