---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 2
entry_id: entry-0006
release_version: 0.1.1
kind: quality
summary:
  Improved release checks for documentation, typing, artifact contents, and
  clean-wheel installation
status: accepted
audience: null
scopes: []
source_refs: []
paths:
  - .github/workflows/test.yml
  - tools/check_release_artifacts.py
  - MANIFEST.in
issues: []
prs: []
sources:
  - tl:task-0003
contributors: []
breaking: false
internal: false
order: 6
---
