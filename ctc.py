import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCLossManual(nn.Module):
    """
    Manual implementation of Connectionist Temporal Classification (CTC) Loss
    as described in the ICML 2006 paper by Alex Graves et al.
    I am assuming that we can use auto-differentiation for this.
    """

    def __init__(self, blank=0):
        super(CTCLossManual, self).__init__()
        self.blank = blank

    def _add_blanks(self, targets):
        """
        Insert blank tokens between labels and at the start/end.
        E.g., target = [A, B] => [blank, A, blank, B, blank]
        """
        B = torch.full((targets.size(0) * 2 + 1,), self.blank, dtype=targets.dtype, device=targets.device)
        B[1::2] = targets
        return B

    def _forward_variables(self, log_probs, targets):
        T = log_probs.size(0)
        S = targets.size(0) # S = number of labels (including blank)
        
        alpha = torch.full((T, S), -float('inf'), device=log_probs.device)
        alpha[0, 0] = log_probs[0, 0, self.blank]
        if S > 1:
            alpha[0, 1] = log_probs[0, 0, targets[1]] # this initializes the forward path probability for the first two states (start with either a blank or first label)

        for t in range(1, T):
            for s in range(S):
                current_label = targets[s].item()
                probs = [alpha[t - 1, s]]  # staying on the same label (probability we were on s and stayed there)

                if s - 1 >= 0:
                    probs.append(alpha[t - 1, s - 1]) # were on label s-1 and moved forward to s

                if s - 2 >= 0 and targets[s] != self.blank and targets[s] != targets[s - 2]:
                    probs.append(alpha[t - 1, s - 2]) # allow skip only if not landing on a blank or repeated label

                alpha[t, s] = torch.logsumexp(torch.stack(probs), dim=0) + log_probs[t, 0, current_label] # prob of all the previous nodes + current node

        return alpha

    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Compute the CTC loss.

        Arguments:
            logits: Tensor of shape (T = time steps, N = batch size, C = number of classes)
            targets: 1D Tensor of target labels
            input_lengths: Lengths of input sequences
            target_lengths: Lengths of target sequences
        """
        log_probs = F.log_softmax(logits, dim=2) 
        T, N, C = log_probs.shape
        assert N == 1, "Only batch size 1 is supported." # for simplicity

        target = targets[:target_lengths[0]]
        targets_with_blanks = self._add_blanks(target)

        alpha = self._forward_variables(log_probs, targets_with_blanks)

        final_alpha = torch.logsumexp(alpha[input_lengths[0] - 1, -2:], dim=0) # sum log-probs of final two CTC states (valid path endings) at last time step

        loss = -final_alpha 
        return loss

# Example usage (for testing, not training)
if __name__ == '__main__':
    T, C = 5, 3  # time steps, number of classes (0: blank, 1, 2)
    logits = torch.randn(T, 1, C, requires_grad=True)
    targets = torch.tensor([1, 2], dtype=torch.long)
    input_lengths = torch.tensor([T], dtype=torch.long)
    target_lengths = torch.tensor([2], dtype=torch.long)

    ctc = CTCLossManual(blank=0)
    loss = ctc(logits, targets, input_lengths, target_lengths)
    print("CTC Loss:", loss.item())

    # Backprop
    loss.backward()
    print("Gradients:", logits.grad)
