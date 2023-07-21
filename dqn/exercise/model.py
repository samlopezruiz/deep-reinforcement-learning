import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # hidden_size = state_size * 2
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action probabilities."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, hidden_size)

        # Advantage layer
        self.fc_adv = nn.Linear(hidden_size, hidden_size)
        self.adv_out = nn.Linear(hidden_size, action_size)

        # Value layer
        self.fc_val = nn.Linear(hidden_size, hidden_size)
        self.val_out = nn.Linear(hidden_size, 1)

    def forward(self, state):
        """Build a network that maps state -> action probabilities."""
        x = F.relu(self.fc1(state))

        adv = F.relu(self.fc_adv(x))
        adv = self.adv_out(adv)

        val = F.relu(self.fc_val(x))
        val = self.val_out(val)

        # Combine the value and advantage streams into the final output
        return val + adv - adv.mean()
