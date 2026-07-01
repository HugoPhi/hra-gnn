from hra_gnn.config import apply_overrides, load_config, validate_config


def test_config_inheritance_and_override() -> None:
    config = load_config("configs/synthetic.yaml")
    config = apply_overrides(config, ["model.hidden_dim=12", "ssl.enabled=false"])
    validate_config(config)
    assert config["dataset"]["kind"] == "synthetic"
    assert config["model"]["hidden_dim"] == 12
    assert config["ssl"]["enabled"] is False
