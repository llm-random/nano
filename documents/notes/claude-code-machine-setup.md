# Replicating the Claude Code setup on a new machine

Written: 2026-10-23 · Status: **established** (set up and verified 2026-10-23)

Claude Code's own configuration — settings, hand-written skills, the executables behind them, and the
per-project agent memory — lives in `~/.claude`, which is outside this repo and does **not** travel with a
`git clone`. It is mirrored by a separate private repo, `github.com:madragonse/claude-sync`, so a second
machine on the same Anthropic account gets an identical agent.

This note is how to bring up that second machine. It is tooling, not Projected Compression work; it is
here because `documents/` is where the repo's written procedures live (cf. `notes/merge_to_main.md`).

## What the account already syncs by itself

Logging in with the same Anthropic account is enough for:

- `~/.claude/skills/synced/` — the Anthropic/org-pushed skills (`deep-research`, `docs`, `docx`, `pdf`,
  `pptx`, `xlsx`, `skill-creator`, `morning`, `computer-use`, `chrome-browser`, `built-in-browser`,
  `google-workspace`, `import-memory`)
- `~/.claude/plugins/synced/`

These are deliberately **not** in the sync repo. Do not copy them by hand; the login refreshes them.

## What the sync repo carries

| Piece | How it is wired |
|---|---|
| `settings.json` — model, effort level, theme, TUI mode, the `UserPromptSubmit` prompt gate | copied, with paths rendered |
| Hand-written skills: `control`, `google`, `mouse`, `webcam-photo`, `wincap` | symlinked, live |
| The executables they call: `lib/winctl/` (`engine.ps1`, `run.sh`, `prompt-gate.sh`), `lib/google/` | symlinked, live |
| Per-project agent memory (`projects/<slug>/memory/`) | symlinked, live, slug remapped per machine |
| `plugins/known_marketplaces.json` | copied, with paths rendered |
| Per-project `allowedTools` out of `~/.claude.json` | filtered export, `files/project-settings.json` |
| `agents/`, `commands/` | symlinked; empty today, so anything added later syncs automatically |

Directories are **symlinked**, so an edit takes effect immediately on the machine that made it — no push
needed for it to work locally. The two single files Claude Code rewrites itself (`settings.json`,
`known_marketplaces.json`) are **copied** instead: the app replaces them atomically, which would turn a
symlink into a plain file on the next write.

## What is excluded, and why

| Excluded | Reason |
|---|---|
| `projects/*/*.jsonl`, `history.jsonl` | ~417 MB of conversation transcripts as of 2026-10-23, grows without bound, and two machines writing one session file conflict |
| `.credentials.json` | the Anthropic auth token; `claude login` handles it per machine |
| `lib/google/tokens/`, `lib/google/client_secret.json` | Google OAuth secrets |
| `lib/google/.venv/` | 159 MB and platform-specific; rebuilt per machine |
| `daemon*`, `cache/`, `sessions/`, `session-env/`, `shell-snapshots/`, `ide/`, `telemetry/`, `file-history/`, `jobs/`, `backups/` | machine-local runtime state |
| most of `~/.claude.json` | `machineID`, `userID`, counters and caches are meaningless on another machine |

The repo is private, and no credentials are committed — verified 2026-10-23 by cloning it back from
GitHub and confirming an anonymous fetch is refused. Keep it private regardless: the memory files
describe real projects and people.

## Bringing up a new machine

```bash
git clone git@github.com:madragonse/claude-sync.git ~/.claude-sync
cd ~/.claude-sync
cp local.conf.example local.conf     # set CODE_ROOT to this machine's repo root
./install.sh
```

`install.sh` is idempotent and backs up anything it replaces under `~/.claude/backups/`. Restart Claude
Code afterwards so the new settings and skills load.

Then restore the three things the repo deliberately does not carry:

```bash
# 1. the google skill's virtualenv
python3 -m venv ~/.claude/lib/google/.venv
~/.claude/lib/google/.venv/bin/pip install google-api-python-client \
    google-auth-httplib2 google-auth-oauthlib

# 2. its OAuth client secret — copy across out of band, then re-authorise
#    (chmod 600; ~/.claude/skills/google/selftest.sh checks this)
~/.claude/lib/google/g auth add

# 3. the wincap binary the wincap skill shells out to
#    built from the separate ~/code/wincap repo, installed to ~/.local/bin/wincap
```

Verify with `~/.claude/skills/google/selftest.sh` (28 offline checks, no API calls).

## Day-to-day

```bash
cd ~/.claude-sync
./push.sh [message]   # pick up app-rewritten files, commit, push
./pull.sh             # pull, then re-wire into ~/.claude
./install.sh          # re-wire only
```

`pull.sh` refuses to run while local changes have not been pushed, so a settings change made on one
machine is not silently overwritten by the other. Restart Claude Code after a pull.

## Gotchas

- **New memories need an explicit push.** The agent memory folder is a symlink into the sync repo, so a
  memory written during a session lands in the repo immediately but reaches the other machine only after
  `./push.sh`. Without it the two machines drift apart silently.
- **Empty memory folders do not survive the clone.** Git cannot track an empty directory, so a project
  whose memory folder has no files yet is absent from the remote; `install.sh` recreates it
  (`install.sh:58`). As of 2026-10-23, 5 of the 11 folders hold files.
- **Memory is keyed by absolute path.** Claude Code slugifies it, turning `/`, `_` and `.` into `-`:
  `/home/maciej/code/nano` → `-home-maciej-code-nano`. `projects.map` therefore stores a *logical* name
  plus a path relative to `CODE_ROOT`, and `local.conf` supplies `CODE_ROOT` per machine. For a repo
  outside that root, add `OVERRIDE_<logical-name-with-underscores>=/abs/path` to `local.conf`.
- **Adding a project later:** append a line to `projects.map`, then `./install.sh`.
- **No absolute paths are committed.** `settings.json` and `known_marketplaces.json` store
  `__CLAUDE_HOME__`, rendered to the real path on install and back on push. This matters because the
  prompt gate is a `UserPromptSubmit` hook: a path that does not resolve fires on *every* prompt.
- **`control`, `mouse`, `wincap` and `webcam-photo` need WSL2 on Windows.** They drive the Windows host
  through `lib/winctl/engine.ps1` and `powershell.exe` over WSL interop, so they sync everywhere but only
  work there. `control` is additionally hard-gated behind the prompt-gate hook and only arms on a
  `control:`-prefixed request.
