"""Registry factory for frozen-time native voice models."""

from tau2.agent.discrete_time_audio_native_agent import DiscreteTimeAudioNativeAgent
from tau2.data_model.simulation import OpenModelConfig
from tau2.voice.audio_native.open_model.adapter import OpenModelAdapter


def create_open_model_agent(tools, domain_policy, **kwargs):
    """Build an existing streaming participant around a native adapter."""
    if tools:
        raise ValueError("Open-model MVP does not support domain tools")
    options = {"model": kwargs.get("model", "minicpm")}
    if kwargs.get("tick_duration_seconds") is not None:
        options["tick_duration_seconds"] = kwargs["tick_duration_seconds"]
    config = kwargs.get("open_model_config") or OpenModelConfig(**options)
    adapter = OpenModelAdapter(
        config,
        runtime_config=kwargs.get("runtime_config"),
        transport=kwargs.get("transport"),
    )
    agent = DiscreteTimeAudioNativeAgent(
        tools=[],
        domain_policy=domain_policy,
        tick_duration_ms=config.tick_duration_ms,
        adapter=adapter,
        provider=config.model,
        model=config.model_identity,
        vad_config={},
        max_inactive_seconds=0,
    )
    agent._owns_adapter = True
    return agent
