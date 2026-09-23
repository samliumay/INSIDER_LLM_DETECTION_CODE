from insider_llm_detection import bench


def test_bundled_benchmark_matches_manifest():
    """The copy in benchmark/ is exactly what MANIFEST.json records."""
    assert bench.check() == []
    assert bench.version(bench.DEFAULT_BENCH).startswith("v0.2")


def test_check_reports_a_changed_file(tmp_path):
    import shutil
    shutil.copytree(bench.DEFAULT_BENCH, tmp_path / "b")
    (tmp_path / "b/prompts/variant_A.md").write_text("edited")
    assert bench.check(tmp_path / "b") == ["prompts/variant_A.md: changed since sync (edit the benchmark repo, then `ild sync-benchmark`)"]
