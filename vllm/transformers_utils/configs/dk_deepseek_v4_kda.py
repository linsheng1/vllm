# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import Any

from vllm.transformers_utils.configs.dk_deepseek_v4_kda_utils import (
    get_dk_layers_block_type,
    normalize_dk_kda_layer_indices,
)
from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config


class DKDeepseekV4KDAConfig(DeepseekV4Config):
    model_type = "dk_deepseek_v4_kda"

    def __init__(
        self,
        dk_kda_layers: list[int] | tuple[int, ...] = (11, 21, 31),
        dk_kda_layers_are_1_based: bool = True,
        dk_kimi_linear_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        self.dk_kda_layers = tuple(dk_kda_layers)
        self.dk_kda_layers_are_1_based = dk_kda_layers_are_1_based
        self.dk_kimi_linear_config = dk_kimi_linear_config
        super().__init__(**kwargs)
        self.dk_kda_layer_indices = normalize_dk_kda_layer_indices(self)
        if hasattr(self, "num_hidden_layers"):
            self.layers_block_type = get_dk_layers_block_type(self)
        else:
            self.layers_block_type = []
