import torch
import torch.nn as nn


class GRUCELLTaskSpecific(torch.nn.Module):
    """
    TODO: add description
    """
    def __init__(self, gru_shared, output_length, input_length=250, hidden_length=100):
        super(GRUCELLTaskSpecific, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
        self.gru_shared = gru_shared
        self.output_length = output_length  # as the amount of labels in the specific task

        # update gate components
        self.linear_w_z = nn.Linear(self.input_length, self.hidden_length, bias=False)
        self.linear_u_z = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_us_z = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_z = nn.Sigmoid()

        # reset gate components
        self.linear_w_r = nn.Linear(self.input_length, self.hidden_length, bias=False)
        self.linear_u_r = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_us_r = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_r = nn.Sigmoid()

        # new memory components
        self.linear_w_hn = nn.Linear(self.input_length, self.hidden_length, bias=False)
        self.linear_u_hn = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_us_hn = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_h = nn.Tanh()

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
        r = self.activation_1(x_new + h_new, hs_new)
        return r

    def new_memory(self, x, h_prev, h_shared_new, r):
        x_new = self.linear_w_hn(x)
        h_new = r * self.linear_u_hn(h_prev)
        hs_new = self.linear_us_hn(h_shared_new)
        nm = self.activation_3(x_new + h_new, hs_new)
        return nm

    def forward(self, x, h_prev, h_shared_prev):
        # call 'forward' of the the shared GRU
        h_shared_new = self.gru_shared(x, h_shared_prev)

        # Equation 1: the update gate
        z = self.update_gate(x, h_prev, h_shared_new)

        # Equation 2. reset gate vector
        r = self.reset_gate(x, h_prev, h_shared_new)

        # Equation 3: The new memory component
        nm = self.new_memory(x, h_prev, h_shared_new, r)

        # Equation 4: the new hidden state
        h_new = (1 - z) * nm + z * h_prev  # TODO: think about it (change as in the paper or not)

        return h_new


class GRUCELLShared(torch.nn.Module):
    """
    TODO: add description
    """
    def __init__(self, input_length=250, hidden_length=100):
        super(GRUCELLShared, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length

        # update gate components
        # weight matrices for task m=1
        self.linear_w_z_1 = nn.Linear(self.input_length, self.hidden_length, bias=False)
        self.linear_u_z_1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        # weight matrices for task m=2
        self.linear_w_z_2 = nn.Linear(self.input_length, self.hidden_length, bias=False)
        self.linear_u_z_2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_z = nn.Sigmoid()

        # reset gate components
        # weight matrices for task m=1
        self.linear_w_r_1 = nn.Linear(self.input_length, self.hidden_length, bias=False)
        self.linear_u_r_1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        # weight matrices for task m=2
        self.linear_w_r_2 = nn.Linear(self.input_length, self.hidden_length, bias=False)
        self.linear_u_r_2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_r = nn.Sigmoid()

        # new memory components
        # weight matrices for task m=1
        self.linear_w_hn_1 = nn.Linear(self.input_length, self.hidden_length, bias=False)
        self.linear_u_hn_1 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        # weight matrices for task m=2
        self.linear_w_hn_2 = nn.Linear(self.input_length, self.hidden_length, bias=False)
        self.linear_u_hn_2 = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_h = nn.Tanh()

    def update_gate(self, x, h_prev, linear_w, linear_u):
        x_new = linear_w(x)
        h_new = linear_u(h_prev)
        z = self.activation_z(x_new + h_new)
        return z

    def reset_gate(self, x, h_prev, linear_w, linear_u):
        x_new = linear_w(x)
        h_new = linear_u(h_prev)
        r = self.activation_1(x_new + h_new)
        return r

    def new_memory(self, x, h_prev, r, linear_w, linear_u):
        x_new = linear_w(x)
        h_new = r * linear_u(h_prev)
        nm = self.activation_3(x_new + h_new)
        return nm

    def forward(self, x, h_prev, m=1):
        if 1 == m:
            linear_w_z = self.linear_w_z_1
            linear_w_r = self.linear_w_r_1
            linear_w_hn = self.linear_w_hn_1
            linear_u_z = self.linear_u_z_1
            linear_u_r = self.linear_u_r_1
            linear_u_hn = self.linear_u_hn_1
        elif 2 == m:
            linear_w_z = self.linear_w_z_2
            linear_w_r = self.linear_w_r_2
            linear_w_hn = self.linear_w_hn_2
            linear_u_z = self.linear_u_z_2
            linear_u_r = self.linear_u_r_2
            linear_u_hn = self.linear_u_hn_2
        else:
            return

        # Equation 1: the update gate
        z = self.update_gate(x, h_prev, linear_w_z, linear_u_z)

        # Equation 2. reset gate vector
        r = self.reset_gate(x, h_prev, linear_w_r, linear_u_r)

        # Equation 3: The new memory component
        nm = self.new_memory(x, h_prev, r, linear_w_hn, linear_u_hn)

        # Equation 4: the new hidden state
        h_new = (1 - z) * nm + z * h_prev  # TODO: think about it (change as in the paper or not)

        return h_new


class GRUTaskSpecific(torch.nn.Module):
    """
    TODO: add description
    """
    def __init__(self, gru_shared, output_length, input_length=250, hidden_length=100):
        super(GRUTaskSpecific, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
        self.gru_shared = gru_shared
        self.output_length = output_length  # as the amount of labels in the specific task

        self.gru_cell_specific = GRUCELLTaskSpecific(gru_shared, output_length, input_length, hidden_length)

        # final classification
        self.linear_v = nn.Linear(self.hidden_length, self.output_length, bias=True)
        self.activation_y = nn.Softmax()

    def forward(self, input_batch):
        outputs = []
        h_t = torch.zeros(input_batch.size(0), self.hidden_length, dtype=torch.double)  # TODO: check if better to use 'randn'

        # for raw in input_batch:
        # h_t = self.gru_cell_specific(input_t, h_t)

        return outputs
