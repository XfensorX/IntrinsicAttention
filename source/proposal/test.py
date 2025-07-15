import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import (
    FilterObservation,
)

ENV_NAME = "MiniGrid-DoorKey-16x16-v0"


def create_env():
    env = gym.make(ENV_NAME)
    env = FilterObservation(env, ["direction", "image"])
    return env


env = create_env()

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("Observation:", obs["image"].reshape(3, 7, 7))


plt.imshow(env.get_wrapper_attr("get_frame")(obs))
plt.title("Observation Image")

obs_, info = env.reset()
plt.imshow(env.get_wrapper_attr("get_frame")(obs))

plt.axis("off")
plt.show()
