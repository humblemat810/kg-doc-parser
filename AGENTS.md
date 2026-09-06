# Agent Notes

## Scope And Ownership

- This repo owns document parsing, OCR normalization, page-index parsing,
  workflow-ingest parser contracts, parser prompts, parser validation, and
  parser-specific ADRs/checklists.
- Put parser design docs under `doc/` in this repo.
- Keep host application wiring, VS Code demo launchers, long-run reporting, and
  promotion policy in `kogwistar-llm-wiki`.
- Keep reusable runtime, graph, workflow, budget, and substrate primitives in
  `kogwistar`.
- When parser code needs runtime behavior, prefer importing or extending
  `kogwistar` primitives instead of duplicating substrate logic here.

## Testing

- Prefer focused parser tests while iterating.
- Use `-p no:cacheprovider` on this Windows workspace when pytest appears to
  pass and then hang during shutdown.

