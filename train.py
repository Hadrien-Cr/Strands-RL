import torch
import torch.nn as nn

def train_agent(agent, optimizer, replay_buffer, batch_size, gamma):
    """
    Train the agent using the replay buffer.
    """

    # Sample a batch from the replay buffer
    sample_size = min(batch_size,replay_buffer.size())
    obs,  actions, next_obs,  dones, rewards = replay_buffer.sample(sample_size)
    # Compute the current Q values
    _,q_values = agent.get_action(obs)
    q_values = torch.gather(q_values,1, actions).squeeze(-1)
    # Compute the target Q values
    with torch.no_grad():
        _,next_q_values = agent.get_action(next_obs)
        max_next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards.squeeze(-1) + (gamma * max_next_q_values * (1 - dones.squeeze(-1)))

    # Compute the loss
    loss = nn.MSELoss()(q_values, target_q_values)
    # Optimize the network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

