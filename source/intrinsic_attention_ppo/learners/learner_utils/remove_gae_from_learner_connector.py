from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.learner import Learner


def remove_gae_from_learner_connectors(learner: Learner):
    """Remove GeneralAdvantageEstimation from the learner connectors.
    As the calculations are not in pytorch hand not differentiable."""
    learner._learner_connector.remove(GeneralAdvantageEstimation)
