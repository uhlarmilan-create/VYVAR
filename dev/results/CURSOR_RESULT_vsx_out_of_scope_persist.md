CURSOR RESULT - 2026-08-11

What I did
Removed vsx_out_of_scope_types from DOC_CONFIG_FACTS so docs-sync no longer forces [] and
reverts Milan's ROT setting. Added config.json round-trip persistence test.

## Output / findings
- `dev/tools/docs_pdf/flow_doc_facts.py`: removed `"vsx_out_of_scope_types": []`
- `test_flow_doc_config_facts` no longer pins user runtime choice
- Skip semantics verified unchanged: Phase 0 sets skip_photometry=True; Phase 1 `continue`s
  on skip_reason=vsx_type_out_of_scope (full LC skip upstream)
- `test_vsx_out_of_scope_types_persists_in_config_json` PASS

## Errors (if any)
None.

## Files changed
- dev/tools/docs_pdf/flow_doc_facts.py
- dev/tests/test_vsx_out_of_scope_types.py
