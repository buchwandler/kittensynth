---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.1.7
kind: added
summary:
  Added public ModelDownloadError reporting for managed model asset download
  and verification failures
status: accepted
audience: null
scopes: []
source_refs:
  - git:5f850c55cfbd9ed21fc8674332891b8319f2d158
paths:
  - kittensynth/__init__.py
  - kittensynth/_onnxvoice.py
  - kittensynth/errors.py
  - tests/test_errors.py
  - tests/test_managed_dependencies.py
  - tests/test_public_api.py
issues: []
prs: []
sources:
  - git:5f850c55cfbd9ed21fc8674332891b8319f2d158
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
