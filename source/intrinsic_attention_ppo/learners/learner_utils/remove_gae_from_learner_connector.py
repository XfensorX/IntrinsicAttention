from ray.rllib.connectors.learner import GeneralAdvantageEstimation
from ray.rllib.core.learner import Learner


def remove_gae_from_learner_connectors(learner: Learner):
    learner._learner_connector.remove(GeneralAdvantageEstimation)
