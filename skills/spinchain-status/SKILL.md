---
name: spinchain-status
description: Check SpinChain MCP server health. Verifies the server is registered in Claude Code config, all modules can be imported, and shows trace summary. Use to diagnose setup issues.
---

# SpinChain Status Check

Run a health check on the SpinChain MCP server installation.

## Steps

1. **Config check** — verify MCP registration:

```bash
cat ~/.claude.json | python3 -c "
import sys, json
d = json.load(sys.stdin)
sc = d.get('mcpServers', {}).get('spinchain')
if sc:
    print('MCP registered: YES')
    print(f'  Command: {sc.get(\"command\")} {\" \".join(sc.get(\"args\", []))}')
    print(f'  Trace dir: {sc.get(\"env\", {}).get(\"SPINCHAIN_TRACE_DIR\", \"(default: ~/.spinchain/traces)\")}')
else:
    print('MCP registered: NO — spinchain not found in mcpServers')
"
```

2. **Dependency check** — verify imports:

```bash
uv run --directory <spinchain-project-dir> python -c "
from spinchain.server import optimize_reasoning
from spinchain.tracing import TraceLogger
from spinchain.analyze import TraceAnalyzer
print('All imports OK')
"
```

3. **Trace stats** — quick summary:

```bash
uv run --directory <spinchain-project-dir> python -m spinchain.analyze 2>/dev/null || echo "No traces yet"
```

4. Present a clear status report:

| Check | Status |
|-------|--------|
| MCP registered | YES / NO |
| Modules importable | YES / NO |
| Trace records | count |
| Last call | timestamp |
| Recent anomalies | count |

If the server is not registered, show the user the JSON config block they need to add to `~/.claude.json`.
