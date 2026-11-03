# AGENTS.md — `.github/`

- Keep action references pinned to full commit SHAs with the release in a
  trailing comment.
- Both jobs stay offline and deterministic: no provider key, no live model call,
  no network beyond the pinned GNN fetch.
- The `gnn conformance (pinned v5.0.3)` job name is the required-check identity.
  Its commit comes from `NHIM/gnn-pin.json`; do not duplicate the hash here.
- Parent rules: [../AGENTS.md](../AGENTS.md).
