import gymnasium as gym

# Create a simple environment perfect for beginners

env = gym.make("CarRacing-v3", render_mode="human")

observation, info = env.reset()
# The CartPole environment: balance a pole on a moving cart
# - Simple but not trivial
# - Fast training
# - Clear success/failure criteria
print(f"Starting observation: {observation}")
episode_over = False
total_reward = 0
while not episode_over:
    action = env.action_space.sample() 
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated
print(f"Episode finished! Total reward: {total_reward}")
env.close()