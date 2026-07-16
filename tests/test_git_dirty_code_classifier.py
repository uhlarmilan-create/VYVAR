"""T3: git_dirty_code vs git_dirty_scratch classification."""

from __future__ import annotations

from photometry_core import classify_git_dirty_paths, _is_import_relevant_py_path


def test_import_relevant_root_py_only() -> None:
    assert _is_import_relevant_py_path("pipeline.py")
    assert not _is_import_relevant_py_path("scripts/foo.py")
    assert not _is_import_relevant_py_path("docs/VYVAR_STATE.md")
    assert not _is_import_relevant_py_path("tests/test_x.py")


def test_tracked_pipeline_mod_is_code() -> None:
    porcelain = " M pipeline.py\n"
    files = [{"path": "pipeline.py", "content_sha256": "abc"}]
    code_dirty, code_paths, scratch = classify_git_dirty_paths(porcelain, files)
    assert code_dirty is True
    assert code_paths == ["pipeline.py"]
    assert scratch == []


def test_untracked_root_py_is_code() -> None:
    porcelain = "?? new_module.py\n"
    files = [{"path": "new_module.py", "content_sha256": "def"}]
    code_dirty, code_paths, scratch = classify_git_dirty_paths(porcelain, files)
    assert code_dirty is True
    assert code_paths == ["new_module.py"]


def test_scratch_only_draft_434_class() -> None:
    """draft_434-style dirt: md/png/docs/scripts — not import-relevant code."""
    porcelain = (
        "?? CURSOR_RESULT_anchor_evidence.md\n"
        "?? docs/VYVAR_CODE_AUDIT.md\n"
        "?? scripts/forensic_disc_ui_match2.py\n"
    )
    files = [
        {"path": "CURSOR_RESULT_anchor_evidence.md", "content_sha256": "a"},
        {"path": "docs/VYVAR_CODE_AUDIT.md", "content_sha256": "b"},
        {"path": "scripts/forensic_disc_ui_match2.py", "content_sha256": "c"},
    ]
    code_dirty, code_paths, scratch = classify_git_dirty_paths(porcelain, files)
    assert code_dirty is False
    assert code_paths == []
    assert len(scratch) == 3


def test_tracked_md_is_scratch() -> None:
    porcelain = " M docs/VYVAR_STATE.md\n"
    files = [{"path": "docs/VYVAR_STATE.md", "content_sha256": "x"}]
    code_dirty, code_paths, scratch = classify_git_dirty_paths(porcelain, files)
    assert code_dirty is False
    assert scratch == ["docs/VYVAR_STATE.md"]
