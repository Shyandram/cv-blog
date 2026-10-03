@AGENTS.md

## Claude Code notes

- All repo rules are in `AGENTS.md` (imported above); they apply to subagents too. When you brief a
  subagent to write or review content, point it at `AGENTS.md` and restate the git-safety rules
  (section 1) and the reference rule (section 2).
- Node is installed via nvm and is not on PATH in non-login shells: prefix commands with
  `export PATH=$HOME/.nvm/versions/node/v22.23.3/bin:$PATH`.
- Use the project venv for Python (`.venv/bin/python`); the system/anaconda base Python has a
  broken NumPy/matplotlib combination.
- Reply to the owner in Traditional Chinese when they write in Chinese.
