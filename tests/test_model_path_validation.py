"""Tests for model weight path allowlisting (path traversal prevention)."""
import json
import os

import pytest

import model_registry


def _write_empty_registry_config(path):
    """Avoid ModelRegistry._initialize_default_config (real paths vs monkeypatched root)."""
    path.write_text(
        json.dumps(
            {
                "models": {"resnet": [], "yolo": []},
                "ab_test_enabled": False,
                "rollback_threshold": 0.7,
            }
        ),
        encoding="utf-8",
    )


def test_validate_model_path_accepts_under_models_dir(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(model_registry, "get_models_allowed_root", lambda: os.path.realpath(str(root)))
    target = root / "weights" / "m.pt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")

    ok, err = model_registry.validate_model_path(str(target))
    assert err is None
    assert ok == os.path.realpath(str(target))


def test_validate_model_path_rejects_parent_escape(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(model_registry, "get_models_allowed_root", lambda: os.path.realpath(str(root)))

    bad = os.path.join(str(root), "..", "secret.pt")
    ok, err = model_registry.validate_model_path(bad)
    assert ok is None
    assert err is not None
    assert "models directory" in err.lower()


def test_register_model_raises_on_traversal(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(model_registry, "get_models_allowed_root", lambda: os.path.realpath(str(root)))
    cfg = tmp_path / "reg.json"
    _write_empty_registry_config(cfg)
    reg = model_registry.ModelRegistry(config_path=str(cfg))

    bad_path = os.path.join(str(root), "..", "evil.pt")

    with pytest.raises(model_registry.ModelPathValidationError):
        reg.register_model(
            model_type="resnet",
            version="v9",
            path=bad_path,
        )


def test_register_model_stores_canonical_path(tmp_path, monkeypatch):
    root = tmp_path / "models"
    sub = root / "sub"
    sub.mkdir(parents=True)
    f = sub / "w.pt"
    f.write_bytes(b"x")
    monkeypatch.setattr(model_registry, "get_models_allowed_root", lambda: os.path.realpath(str(root)))
    cfg = tmp_path / "reg2.json"
    _write_empty_registry_config(cfg)
    reg = model_registry.ModelRegistry(config_path=str(cfg))

    reg.register_model(model_type="resnet", version="v1", path=str(f))
    meta = reg.get_model("resnet", "v1")
    assert meta is not None
    assert os.path.isabs(meta.path)
    assert os.path.normcase(meta.path) == os.path.normcase(os.path.realpath(str(f)))
