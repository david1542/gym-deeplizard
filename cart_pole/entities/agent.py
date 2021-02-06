import numpy as np
import torch


MODES = {
    'LEARN': 'LEARN',
    'INFER': 'INFER'
}

class Agent:
    def __init__(self, strategy, num_actions, device, mode=MODES['LEARN']):
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device
        self.mode = mode
        self.current_step = 0
    
    def select_action(self, state, policy_net):
        if self.mode == MODES['LEARN']:
            exploration_rate = self.strategy.get_exploration_rate(self.current_step)
            self.current_step += 1

            # Exploration V.S Exploitation
            if np.random.rand() < exploration_rate:
                action = np.random.choice(self.num_actions)
                return torch.tensor([action]).to(self.device) # Explore
            else:
                with torch.no_grad():
                    return policy_net(state).argmax(dim=1) # Exploit
        else:
            with torch.no_grad():
                    return policy_net(state).argmax(dim=1) # Infer
