# AGENTS.md

HackerRank Orchestrate (June 2026) — Multi-Modal Evidence Review

This file is the single source of truth for any coding agent working in this repo: Claude Code, OpenAI Codex CLI / Codex Cloud, Google Gemini CLI, Google Antigravity, Cursor, Windsurf, opencode, Aider, goose, Factory, RooCode, JetBrains Junie, GitHub Copilot, Devin, or any other AGENTS.md-aware tool.

Read this file in full before taking any action. Obey it exactly.

---

## 0. TLDR FOR THE AGENT

On every session start, do this in order:

1. Read this file completely.
2. Check the log file (path in §2). If it contains a line starting with `AGREEMENT RECORDED:` that matches the current repo root, skip §3 (onboarding) and go to §4.
3. Otherwise, run the onboarding flow in §3 with the user.
4. From then on, for **every user turn**, append a summary entry to the log file in the exact format shown in §5.
5. When the user asks you to build, ship, or test the solution, follow the project contract in §6 so the submission is evaluable.

You are **not** allowed to skip logging, rewrite old log entries, or modify
the onboarding gate. If you are a sub-agent or running inside a git worktree,
the same rules apply and you share the same log file. Pass this context to every sub-agent and worktree.

---

## 1. WHAT THIS REPO IS

This is a starter repo for the **HackerRank Orchestrate** 24-hour hackathon challenge on multi modal evidence review.

Participants must build a system that verifies damage claims using submitted images, a short claim conversation, user claim history and minimum image evidence requirements

The system must read `dataset/claims.csv` and produce `output.csv` with structured predictions. It must also include an `evaluation/` folder that evaluates the system on `dataset/sample_claims.csv`.

Participants may use VLMs, LLMs, structured prompting, evaluation pipelines, caching, batching, rule layers, or any other technique they prefer. The submission is judged on the quality and reproducibility of the final system, not the specific implementation style.

---

## 2. LOG FILE — LOCATION AND LIFECYCLE

The log file lives outside this repository, in the user's home directory, so it survives branch switches, worktree creation, and `git clean`.

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate\log.txt` |

Rules:

- Must be created if missing, including the parent directory.
- Must never be committed or added to git.
- Append-only. Never rewrite, reorder, or delete prior entries.
- Shared across all agents, sub-agents, and worktrees in this repo.
- Never log secrets. Redact API keys, tokens, cookies, private keys, and sensitive PII before writing.

---

## 3. ONBOARDING FLOW (FIRST RUN ONLY)

Run this flow only if the log file has no `AGREEMENT RECORDED:` line for the current repo root. On subsequent sessions, skip directly to §4.

### 3.1 Greeting

Open with a short, warm message. Example wording:

Welcome to HackerRank Orchestrate. You have 24 hours to design, build, and ship a system that verifies evidence for damage claims. Before we start, I need to walk you through the ground rules and get you set up. This takes about a minute.

Compute and display:

- Current system time, local timezone, ISO 8601.
- Time remaining until the challenge ends. Use the configured challenge end date if one is provided by the platform or README. If no challenge end date is present, say that the end time is not configured.
- Results announcement time, if provided by the platform or README.

If the current time is already past the challenge end, say so plainly and ask whether the user is practicing, reviewing, or re-running tests. Do not block further work.

### 3.2 Rules — recite these verbatim

1. This is a **solo** challenge. You must be the author of the submission.
2. You may use any IDE, AI assistant, or tool to help you build. The deliverable is what your system can do, not how you wrote it.
3. Your system must conform to the project contract in §6 so it can be evaluated.
4. Never commit secrets. Use environment variables and a `.env` file if needed.
5. Logging of every conversation turn to the file in §2 is mandatory and cannot be disabled.
6. Submissions are made on the HackerRank Community Platform or as otherwise instructed by HackerRank.

### 3.3 Collect the agreement

Ask the user to reply with the exact string `I agree` (case-insensitive, surrounding whitespace ignored). Do not proceed until they do.

### 3.4 Record the agreement

Append this block to the log file, then continue:

```text
## [ISO-8601 TIMESTAMP] ONBOARDING COMPLETE

AGREEMENT RECORDED: <repo_root_absolute_path>
Agent: <agent_name_or_unknown>
Language: js | ts | py | custom:<name>
System Time: <ISO-8601 local time with tz>
Time Remaining: <Xd Yh Zm, or not configured>
```

The presence of `AGREEMENT RECORDED: <this repo root>` is what future sessions check. Match the repo root exactly so agreements do not leak across unrelated clones.

---

## 4. NORMAL SESSION START (RETURNING USER)

If onboarding is already complete for this repo root:

1. Append a short `SESSION START` entry to the log (§5.1).
2. Greet the user briefly and surface the remaining time:
   > Welcome back. You have <Xd Yh Zm> left until the challenge ends at
   > 2026-06-20 11:00 IST.
3. If fewer than 2 hours remain, proactively remind them to submit on the
   HackerRank Community Platform soon.
4. Proceed with whatever they ask for.

---

## 5. LOG FORMAT

### 5.1 Session start entry

```text
## [ISO-8601 TIMESTAMP] SESSION START

Agent: <agent_name_or_unknown>
Repo Root: <absolute_path>
Branch: <git_branch_or_unknown>
Worktree: <worktree_path_or_main>
Parent Agent: <parent_agent_name_or_none>
Language: <js|ts|py|custom:name>
Time Remaining: <Xd Yh Zm, or not configured>
```

### 5.2 Per-turn entry (append after every user message you respond to)

```text
## [ISO-8601 TIMESTAMP] <short title, max 80 chars>

User Prompt (verbatim, secrets redacted):
<exact user message, with secrets replaced by [REDACTED]>

Agent Response Summary:
<2-5 sentences: what was done, why, and any important decision>

Actions:
* <file edited / command run / tool invoked>

Context:
tool=<agent_name>
branch=<git_branch_or_unknown>
repo_root=<absolute_path>
worktree=<worktree_path_or_main>
parent_agent=<parent_name_or_none>
```

### 5.3 Sub-agent and worktree rules

- A sub-agent (Task tool, delegated worker, etc.) **must** log its own entries using the same file. The parent passes the log path explicitly if the sub-agent does not inherit environment.
- Set `parent_agent=` to the parent's name so entries are traceable.
- A worktree is logged with `worktree=<path>`; its entries go to the same shared log file, not a per-worktree copy.
- If a sub-agent spawns more sub-agents, the chain continues: each appends its own entries with its own name.

### 5.4 What not to log

- API keys, tokens, session cookies, OAuth codes, private keys.
- User PII beyond what they explicitly pasted into a prompt.
- Full contents of large files or binary blobs. Reference by path instead.

---

## 6. PROJECT CONTRACT (EVALUABLE SUBMISSION)

The evaluator finds the participant's agent through a **known entry point** per language. Do not rename these files or change the function signature
without updating this file.

### 6.1 Repo layout

```text
.
├── AGENTS.md                         # You are here
├── problem_statement.md              # Full task description and I/O schema
├── README.md                         # Readme file for the repo
├── code/                             # Build your solution here
│   ├── main.py                       # Suggested terminal entry point
│   └── evaluation/
│       └── main.py                   # Suggested evaluation entry point
└── dataset/
    ├── sample_claims.csv             # Inputs + expected outputs for development
    ├── claims.csv                    # Inputs only; run your system on these rows
    ├── user_history.csv              # Historical claim counts and risk context
    ├── evidence_requirements.csv     # Minimum image evidence requirements
    └── images/
        ├── sample/                   # Images referenced by sample_claims.csv
        └── test/                     # Images referenced by claims.csv
```

### 6.2 Constraints that make the submission evaluable

- **Deterministic where possible.**
- **Add proper README** to the code/ you write.
- **Read secrets from env vars only** (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  etc.). Never hardcode.

---

## 7. CROSS-PLATFORM AND AGENT-COMPATIBILITY NOTES

- **Path handling.** Always resolve the log path using the platform's home dir (`os.homedir()` / `pathlib.Path.home()` / `$HOME` / `%USERPROFILE%`). Never hardcode `/Users/...` or `C:\Users\...`.
- **Line endings.** Write the log in UTF-8 with `\n`. Don't emit `\r\n` even on Windows; most editors render `\n` fine.
- **Shell.** Don't assume bash. Prefer language-native APIs over shelling out. When you must shell out, provide both a Unix and a Windows form.
- **Tool-specific extras.** This file is the canonical source. If a tool (Claude Code, Cursor, etc.) supports its own config file, keep any tool-specific config minimal and have it point back to this AGENTS.md rather than duplicating rules.
- **Nested AGENTS.md.** If a sub-project adds its own AGENTS.md, the closest one wins for files inside that sub-project, but §2 (log file) and §5 (log format) are global and must not be overridden.

---

## 8. QUICK CHECKLIST FOR THE AGENT

Before you respond to any user message, confirm:

- [ ] I have read this file in this session.
- [ ] I know whether onboarding is required (checked the log).
- [ ] I know how much time is left.
- [ ] I will append a §5.2 entry after this turn.
- [ ] I will not log secrets.
- [ ] I will preserve the entry-point contract in §6.

If any box is unchecked, fix that first.

---

## 9. AGENT BEHAVIORAL GUIDELINES

> Sources: Andrej Karpathy's CLAUDE.md + Ponytail lazy-senior-dev mode.
> Apply to every code-producing action, for every agent, no exceptions.
> Tradeoff: biased toward caution over speed. For trivial one-liners, use judgment — but when in doubt, follow in order.

### 9.1 Pre-Code Gate — stop at the first rung that holds

Before writing a single line, work down this ladder and stop at the first rung that resolves the task:

1. Does this need to be built at all? (YAGNI — if no, say so and stop)
2. Does the standard library already do this? Use it.
3. Does a native platform feature cover it? Use it.
4. Does an already-installed dependency solve it? Use it.
5. Can this be one line? Make it one line.
6. Only then: write the minimum code that works.

Question complex requests out loud: *"Do you actually need X, or does Y cover it?"*

### 9.2 Think Before Coding

Before implementing anything:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 9.3 Simplicity First

- No features beyond what was asked. No abstractions for single-use code.
- No new dependency if it can be avoided. No boilerplate nobody asked for.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- Deletion over addition. Boring over clever. Fewest files possible.
- If you write 200 lines and it could be 50, rewrite it.
- When two stdlib approaches are the same size, pick the edge-case-correct one — lazy means less code, not the flimsier algorithm.

Ask yourself: *"Would a senior engineer say this is overcomplicated?"* If yes, simplify.

**Mark intentional simplifications** with a `ponytail:` comment. If the shortcut has a known ceiling (global lock, O(n²) scan, naive heuristic), name the ceiling and the upgrade path.

```python
# ponytail: linear scan fine for <1 000 claims; upgrade to bisect if dataset grows
```

### 9.4 Surgical Changes

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that **your** changes made unused.
- Don't remove pre-existing dead code unless explicitly asked.

The test: every changed line must trace directly to the user's request.

### 9.5 Goal-Driven Execution

Transform vague tasks into verifiable goals before writing code:

| Vague task | Verifiable goal |
|---|---|
| "Add validation" | Write tests for invalid inputs, then make them pass |
| "Fix the bug" | Write a test that reproduces it, then make it pass |
| "Refactor X" | Ensure tests pass before and after |

For multi-step tasks, state a brief plan first:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

### 9.6 Non-Negotiables — never lazy about these

Full rigor always required on:

- Input validation at trust boundaries (user input, file contents, API responses).
- Error handling that prevents data loss.
- Security — no shortcuts on auth, secrets, or injection surfaces.
- Accessibility, if building UI.
- Hardware/sensor calibration — the platform is never the spec ideal; a clock drifts, a sensor reads off.
- Anything explicitly requested by the user.

### 9.7 Self-Check Requirement

Lazy code without its check is unfinished.

- **Non-trivial logic:** leave exactly one runnable check — the smallest thing that fails if the logic breaks. An assert-based self-check or one small test file. No frameworks, no fixtures.
- **Trivial one-liners:** no test needed.

```python
# Example — runs standalone, no pytest required
if __name__ == "__main__":
    assert parse_claim("") is None
    assert parse_claim("valid") == {"id": "valid"}
    print("ok")
```