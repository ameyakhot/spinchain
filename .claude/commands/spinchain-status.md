---
allowed-tools: Bash(uv run:*), Bash(cat:*), Bash(wc:*), Bash(ls:*), Bash(grep:*), Read
description: Check SpinChain MCP server health — config, trace stats, and server readiness
---

## Context

- MCP config: `~/.claude.json` under `mcpServers.spinchain`
- Trace dir: `$SPINCHAIN_TRACE_DIR` or `~/.spinchain/traces/`
- Server entry: `python -m spinchain.server`

## Your task

Run a health check on the SpinChain MCP server and report status.

### Steps

1. **Config check** — verify the MCP server is registered:

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

2. **Dependency check** — verify the package can be imported:

```bash
uv run --directory /Users/maverick/quantum/spinchain python -c "
from spinchain.server import optimize_reasoning
from spinchain.tracing import TraceLogger
from spinchain.analyze import TraceAnalyzer
print('All imports OK')
"
```

3. **Trace stats** — quick summary if traces exist:

```bash
uv run --directory /Users/maverick/quantum/spinchain python -m spinchain.analyze 2>/dev/null || echo "No traces yet"
```

4. Present a clear status report:
   - Is the MCP server configured? (YES/NO)
   - Can all modules be imported? (YES/NO)
   - How many trace records exist?
   - When was the last call?
   - Any recent anomalies?
