---
inclusion: always
---
# Cohesion Guardian

This project is monitored by Cohesion Guardian for architectural consistency.
The `cohesion-guardian` MCP server is connected to this IDE and exposes these tools:
`check_branch`, `list_issues`, `fix_issue`, `append_reasoning_log`.

## Repository identity

This workspace's Cohesion Guardian repo_id is: `Document-Classification-and-Information-Extraction-`

ALWAYS pass exactly this value as the `repo_id` argument to every cohesion-guardian MCP tool
(check_branch, list_issues, fix_issue, append_reasoning_log). Do NOT use the repo folder name or
guess it — the repo_id can differ from the folder name, and only this exact value matches the
Cohesion Guardian registry.

## Hook: Document Reasoning

If the "Document Reasoning" hook does NOT exist in this workspace, create it now with these settings:
- name: Document Reasoning
- event: postToolUse
- toolTypes: write
- action: askAgent
- prompt: "Use the cohesion-guardian MCP tool 'append_reasoning_log' to document the reasoning behind this code change. Pass the repo_id from the Repository identity section above, the current branch name, and: files_changed (comma-separated list), change_summary (one sentence), reasoning (1-2 sentences explaining WHY this change was made)."

This hook ensures that every code change made by a developer on any branch is recorded with its
reasoning, so the Issue Resolver can propose solutions that respect developer intent.

## MANDATORY TOOL USAGE — do NOT bypass these tools:

- User asks to CHECK / REVIEW / ANALYZE a branch for conflicts, cohesion, or consistency
  → you MUST call the `check_branch` MCP tool. Do NOT analyze the branch yourself.
- User asks to SEE / LIST / SHOW issues
  → you MUST call the `list_issues` MCP tool. Do NOT summarize from memory.
- User asks to FIX an issue ("fix this issue", "fix inc-XXXX", "apply solution N")
  → you MUST call the `fix_issue` MCP tool with repo_id, issue_id, and solution_number.
  Do NOT implement the fix yourself. Do NOT edit files directly. Do NOT create branches yourself.
  The Corrector Agent implements the fix through the `fix_issue` tool — your ONLY job is to call it.
  Fixing manually bypasses sandbox verification and the guardian workflow, and is NOT allowed.

These tools run the multi-agent Cohesion Guardian pipeline on AWS. Always prefer them over
doing the work locally.

## Guidelines
- Follow naming conventions in `.guardian/naming_conventions.md`
- Check architecture patterns in `.guardian/strategy.md`
- Your change reasoning is logged automatically via the Document Reasoning hook
