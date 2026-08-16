"""
Misawantiga Shugen — bigger preset, training on the enhanced CCU corpus (100 MB).

Some Fry architecture (config-driven GPT), scaled up to exploit the larger dataset:
CCU_enchanced is ~xxB  train tokens / vocab 40000 vs the original CCU dataset. `model_name`
stays "fry — that is the architecture dispatch key in `MODEL_REGISTRY`; "shugen" is just a
larger configuration of it.
"""