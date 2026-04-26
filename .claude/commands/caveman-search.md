Search the codebase for $ARGUMENTS using raw grep and find — no indexing, no fancy tooling, just brute-force shell commands.

Steps:
1. Run `grep -rn --include="*" "$ARGUMENTS" .` to find every line in every file that contains the search term (case-sensitive). Also run with `-i` flag for a case-insensitive pass.
2. Run `find . -name "*$ARGUMENTS*" -not -path "./.git/*"` to find files or directories whose names contain the term.
3. Summarise the results: list matching files with their line numbers and the matching lines, then list any filename matches. If there are no results, say so plainly.

Keep the output compact — file path + line number + matched line. No prose padding.
