---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.1.9
kind: fixed
summary:
  Accepted punctuation-rich linguistic token context with the compatible Kitteng2p
  frontend
status: accepted
audience: null
scopes: []
source_refs:
  - tl:task-0050
paths:
  - pyproject.toml
  - tests/test_g2p_requests.py
  - kittensynth/voice.py
issues: []
prs: []
sources: []
contributors: []
breaking: false
internal: false
order: 1
---

The regular and bundled dependency floors require Kitteng2p 0.1.3, whose authoritative word segmentation safely projects compatible caller annotations.
