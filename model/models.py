import torch
import torch.nn as nn


class GRUCELLTaskSpecific(torch.nn.Module):
    """
    This class is our GRUCELL for the task specific layers, which includes the sharedGRU result in it`s computations.
    """

    def __init__(self, input_length=250, hidden_length=100):
        super(GRUCELLTaskSpecific, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # update gate components
        self.linear_w_z = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_u_z = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.linear_us_z = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.activation_z = nn.Sigmoid()

        # reset gate components
        self.linear_w_r = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_u_r = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.linear_us_r = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.activation_r = nn.Sigmoid()

        # new memory components
        self.linear_w_hn = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_u_hn = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.linear_us_hn = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.activation_hn = nn.Tanh()

    def update_gate(self, x, h_prev, h_shared_new):
        x_new = self.linear_w_z(x)
        h_new = self.linear_u_z(h_prev)
        hs_new = self.linear_us_z(h_shared_new)
        z = self.activation_z(x_new + h_new + hs_new)
        return z

    def reset_gate(self, x, h_prev, h_shared_new):
        x_new = self.linear_w_r(x)
        h_new = self.linear_u_r(h_prev)
        hs_new = self.linear_us_r(h_shared_new)
        r = self.activation_r(x_new + h_new + hs_new)
        return r

    def new_memory(self, x, h_prev, h_shared_new, r):
        x_new = self.linear_w_hn(x)
        h_new = r * self.linear_u_hn(h_prev)
        hs_new = self.linear_us_hn(h_shared_new)
        nm = self.activation_hn(x_new + h_new + hs_new)
        return nm

    def forward(self, x, h_prev, h_shared_new):
        # Equation 1: the update gate
        z = self.update_gate(x, h_prev, h_shared_new)

        # Equation 2. reset gate vector
        r = self.reset_gate(x, h_prev, h_shared_new)

        # Equation 3: The new memory component
        nm = self.new_memory(x, h_prev, h_shared_new, r)

        # Equation 4: the new hidden state
        h_new = (1 - z) * nm + z * h_prev

        return h_new


class GRUCELLShared(torch.nn.Module):
    """
    Our implementation of the GRUCELL for the shared layer.
    Used by each of the tasks.
    """



class GRUTaskSpecific(torch.nn.Module):
    """
    Our implementation of the GRU for the task specific layer.
    """
