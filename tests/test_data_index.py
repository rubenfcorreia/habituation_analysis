from habituation_analysis import data


def test_refresh_index_excludes_test_experiments(tmp_path, monkeypatch):
    source_root = tmp_path / "source"
    (source_root / "M1" / "2026-01-01_01_M1").mkdir(parents=True)
    (source_root / "TEST" / "2026-01-01_01_TEST").mkdir(parents=True)
    monkeypatch.setattr(data, "ensure_output_dirs", lambda: None)
    monkeypatch.setattr(data, "INDEX_PATH", tmp_path / "dataset_index.json")

    index = data.HabituationStore(source_root=source_root, remote_root=tmp_path / "remote").refresh_index()

    assert [summary.exp_id for summary in index.sessions] == ["2026-01-01_01_M1"]
