import torch
import torch.nn as nn
import defines as df


class GRUCellTaskSpecific(torch.nn.Module):
    """
    This class is our implementation of GRUCELL for the task specific layers,
    which includes the sharedGRU result in it`s computations.
    """
    def __init__(self, input_length=250, hidden_length=100, hidden_length_shared=100):
        super(GRUCellTaskSpecific, self).__init__()
        self.input_length = input_length
        self.hidden_length = hidden_length
        self.hidden_length_shared = hidden_length_shared

        # update gate components
        self.linear_w_z = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_u_z = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.linear_us_z = nn.Linear(self.hidden_length_shared, self.hidden_length, bias=True)
        self.activation_z = nn.Sigmoid()

        # reset gate components
        self.linear_w_r = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_u_r = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.linear_us_r = nn.Linear(self.hidden_length_shared, self.hidden_length, bias=True)
        self.activation_r = nn.Sigmoid()

        # new memory components
        self.linear_w_hn = nn.Linear(self.input_length, self.hidden_length, bias=True)
        self.linear_u_hn = nn.Linear(self.hidden_length, self.hidden_length, bias=True)
        self.linear_us_hn = nn.Linear(self.hidden_length_shared, self.hidden_length, bias=True)
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


class GRUMultiTask(torch.nn.Module):
    """
    Our implementation of the Multi-task GRU model, containing 3 GRU layers: 1 shared layer for both tasks
    and task specific layer for each task.
    """
    def __init__(self, input_length=250, hidden_length_rumors=100, hidden_length_stances=100, hidden_length_shared=100,
                 loss_func='CrossEntropyLoss', is_dropout=False, drop_prob=0):
        super(GRUMultiTask, self).__init__()
        self.input_length = input_length
        self.hidden_length_rumors = hidden_length_rumors
        self.hidden_length_stances = hidden_length_stances
        self.hidden_length_shared = hidden_length_shared
        self.is_dropout = is_dropout

        if loss_func != 'CrossEntropyLoss' and loss_func != 'BCELoss' and loss_func != 'L1Loss' \
                and loss_func != 'MSELoss':
            self.loss_func = 'CrossEntropyLoss'
        else:
            self.loss_func = loss_func

        self.gru_cell_rumors = GRUCellTaskSpecific(self.input_length, self.hidden_length_rumors,
                                                   self.hidden_length_shared)
        self.gru_cell_stances = GRUCellTaskSpecific(self.input_length, self.hidden_length_stances,
                                                    self.hidden_length_shared)
        self.gru_cell_shared = nn.GRUCell(self.input_length, self.hidden_length_shared, bias=True)

        # for final classification
        self.linear_v_rumors = nn.Linear(self.hidden_length_rumors, df.output_dim_rumors, bias=True)
        self.linear_v_stances = nn.Linear(self.hidden_length_stances, df.output_dim_stances, bias=True)

        if self.is_dropout:
            self.dropout = nn.Dropout(drop_prob)

        # if we use CrossEntropyLoss we don't need to apply softmax because
        # it (CrossEntropyLoss) apply a log_softmax layer after our final layer
        if self.loss_func != 'CrossEntropyLoss':
            self.activation_y_softmax_rumors = nn.Softmax(dim=0)
            self.activation_y_softmax_stances = nn.Softmax(dim=0)

    def forward(self, batch, h_prev_shared, m, h_prev_rumors=None, h_prev_stances=None):
        outputs = []
        for raw in batch:
            r = raw.view(1, self.input_length)
            h_s = h_prev_shared.view(1, self.hidden_length_shared)
            h_prev_shared = self.gru_cell_shared(r, h_s)
            h_prev_shared = h_prev_shared.view(self.hidden_length_shared)

            if m == df.task_rumors_no:
                h_prev_rumors = self.gru_cell_rumors(raw, h_prev_rumors, h_prev_shared)
                if self.is_dropout:
                    h_prev_rumors = self.dropout(h_prev_rumors)
                v = self.linear_v_rumors(h_prev_rumors)
                if self.loss_func != 'CrossEntropyLoss':
                    output = self.activation_y_softmax_rumors(v)
                else:
                    output = v
            else:  # m == df.task_stances_no
                h_prev_stances = self.gru_cell_stances(raw, h_prev_stances, h_prev_shared)
                if self.is_dropout:
                    h_prev_stances = self.dropout(h_prev_stances)
                v = self.linear_v_stances(h_prev_stances)
                if self.loss_func != 'CrossEntropyLoss':
                    output = self.activation_y_softmax_stances(v)
                else:
                    output = v

            outputs.append(output)
        if m == df.task_rumors_no:
            return torch.stack(outputs), h_prev_shared, h_prev_rumors
        else:  # m == df.task_stances_no
            return torch.stack(outputs), h_prev_shared, h_prev_stances

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.hidden_length_rumors).zero_(),   # for rumors
                  weight.new(self.hidden_length_stances).zero_(),  # for stances
                  weight.new(self.hidden_length_shared).zero_())   # for shared
        return hidden
