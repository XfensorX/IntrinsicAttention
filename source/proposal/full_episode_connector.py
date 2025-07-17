from typing import Any, Dict, List, Optional

import numpy as np
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType


# FIXME: This is boilerplate and completely wrong
class FullEpisodeConnector(ConnectorV2):
    """A connector piece that adds the episode id to the batch ."""

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        # Learner connector pipeline.

        for i, sa_episode in enumerate(
            self.single_agent_episode_iterator(episodes, agents_that_stepped_only=False)
        ):
            # Get all observations from the episode in one np array (except for
            # the very last one, which is the final observation not needed for
            # learning).
            i_batch = np.full((len(sa_episode)), i + 1)
            self.add_n_batch_items(
                batch=batch,
                column=Columns.EPS_ID,
                items_to_add=i_batch,
                num_items=len(sa_episode),
                single_agent_episode=sa_episode,
            )
        return batch
