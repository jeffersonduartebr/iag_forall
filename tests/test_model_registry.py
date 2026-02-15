from app import model_registry as mr


def _fresh_registry():
    mr.ModelRegistry._instance = None
    return mr.ModelRegistry()


def test_model_config_properties_and_cost():
    cfg = mr.ModelConfig(
        name="my-model",
        provider=mr.Provider.OPENAI,
        capabilities={mr.Capability.TEXT, mr.Capability.VISION, mr.Capability.REASONING},
        cost_per_1k_input=0.5,
        cost_per_1k_output=1.0,
    )

    assert cfg.display_name == "my-model"
    assert cfg.full_name == "openai/my-model"
    assert cfg.supports_vision is True
    assert cfg.supports_streaming is False
    assert cfg.supports_reasoning is True
    assert cfg.calculate_cost(1000, 2000) == 2.5


def test_registry_get_list_and_filters():
    registry = _fresh_registry()

    by_short = registry.get("gpt-4o")
    by_full = registry.get("openai/gpt-4o")
    assert by_short is not None
    assert by_full is by_short

    all_models = registry.list_models()
    assert all_models
    assert all_models == sorted(all_models, key=lambda m: m.priority)

    vision_models = registry.get_vision_models()
    assert vision_models
    assert all(mr.Capability.VISION in m.capabilities for m in vision_models)

    local_models = registry.get_local_models()
    assert local_models
    assert all(m.provider == mr.Provider.OLLAMA for m in local_models)


def test_registry_get_or_default_and_fallback_chain():
    registry = _fresh_registry()

    known = registry.get_or_default("openai/gpt-4o")
    assert known.name == "gpt-4o"

    unknown_prefixed = registry.get_or_default("anthropic/custom-x")
    assert unknown_prefixed.provider == mr.Provider.ANTHROPIC
    assert unknown_prefixed.name == "custom-x"

    unknown_invalid = registry.get_or_default("custom/nope")
    assert unknown_invalid.provider == mr.Provider.OLLAMA
    assert unknown_invalid.name == "nope"

    a = mr.ModelConfig(
        name="chain-a",
        provider=mr.Provider.OPENAI,
        fallback_models=["openai/chain-b"],
    )
    b = mr.ModelConfig(
        name="chain-b",
        provider=mr.Provider.OPENAI,
        fallback_models=["openai/chain-a", "openai/chain-c"],
    )
    c = mr.ModelConfig(name="chain-c", provider=mr.Provider.OPENAI)
    registry.register(a)
    registry.register(b)
    registry.register(c)

    chain = registry.get_fallback_chain("openai/chain-a", max_depth=5)
    names = [m.full_name for m in chain]
    assert names == ["openai/chain-b", "openai/chain-c"]


def test_registry_cheapest_and_convenience_helpers():
    registry = _fresh_registry()

    cheapest_text = registry.get_cheapest_model(capability=mr.Capability.TEXT)
    assert cheapest_text is not None
    assert cheapest_text.cost_per_1k_output > 0

    assert registry.get_cheapest_model(capability=mr.Capability.FUNCTION_CALLING) is not None
    assert registry.get_cheapest_model(capability=mr.Capability("text")) is not None

    # Empty registry branch
    registry._models.clear()
    assert registry.get_cheapest_model() is None

    mr.ModelRegistry._instance = None
    cfg = mr.get_model_config("gpt-4o")
    assert cfg is not None
    assert cfg.full_name == "openai/gpt-4o"
