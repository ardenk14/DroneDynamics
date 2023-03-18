import torch.nn as nn

class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        output = model(state, action)
        loss = self.loss(output, target_state)

        return loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        last_state = state
        # Keep track of last action
        loss = 0.0

        for i in range(target_states.shape[1]): # Loop through each target state
            # Apply model to last predicted action and state
            new_state = model(last_state, actions[:, i, :])
            # Compare prediction to ground truth (add loss function with discount)
            loss += self.discount**i * self.loss(new_state, target_states[:, i, :])
            last_state = new_state

        return loss