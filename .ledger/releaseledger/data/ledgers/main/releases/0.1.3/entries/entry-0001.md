---
schema_version: 2
object_type: release_entry
versioning:
  schema_version: 1
  revision: 1
entry_id: entry-0001
release_version: 0.1.3
kind: added
summary:
  Added catalog-only model discovery, deterministic runtime identity, request
  API capabilities, and typed public errors
status: accepted
audience: null
scopes: []
source_refs:
  - git:801a9ec4dd8df2c0a4284e7d8d171412592c492c
paths:
  - .github/workflows/python-publish.yml
  - .github/workflows/test.yml
  - README.md
  - docs/architecture.md
  - docs/onnxvoice-contract.md
  - kittensynth/__init__.py
  - kittensynth/api_contract.py
  - kittensynth/discovery.py
  - kittensynth/errors.py
  - kittensynth/identity.py
  - kittensynth/voice.py
  - tests/test_discovery.py
  - tests/test_errors.py
  - tests/test_identity.py
  - tests/test_public_api.py
  - tests/test_voice.py
issues: []
prs: []
sources:
  - git:801a9ec4dd8df2c0a4284e7d8d171412592c492c
contributors:
  - "@holgern"
breaking: false
internal: false
order: 1
---
