import numpy as np
import torch
import torch.nn as nn

w = torch.nn.Linear(5, 3)
print(w._parameters)
print(w.weight)

v = torch.tensor([-5.5, -7.3, 8.2, 8.7, 4.2])

n = w(v)

print(n)

print()


t = torch.tensor([-5.5, -7.3, 8.2])
g = nn.Tanh()
s = nn.Softmax(dim=0)
d = g(t)
d = s(d)
print(d)


# -------------------------------------------------------------------------------------

#               (250, 100, True)
rnn = nn.GRUCell(10, 20, True)
input = torch.randn(10)
input = input.view(1, 10)
hx = torch.randn(1, 20)
output = []
hx = rnn(input, hx)
hx = hx.view(20)
output.append(hx)
print(hx)
print(output)
print()


rnn = nn.GRUCell(10, 20)
input = torch.randn(6, 3, 10)
hx = torch.randn(3, 20)
output = []
for i in range(6):
    hx = rnn(input[i], hx)
    output.append(hx)

# -------------------------------------------------------------------------------------

# v = Variable(weight.new(1, 5, 10).zero_())

# ------------------------------ Working -------------------------------------------------------

# max_idx = torch.argmax(output, dim=0)  # dim=0 means: look in the first row
# one_hot = torch.nn.functional.one_hot(torch.tensor([max_idx]), self.output_length)
# one_hot = torch.squeeze(one_hot, 0)

# -------------------------------------------------------------------------------------

data = np.load('..\\data\\preprocessed data\\training\\rumors_labels.npy')

print(data)
print(type(data))
print(data.ndim)
print(len(data))



# -------------------------------------------------------------------------------------

list = []

l1 = torch.tensor([0.5, 0.2, 0.85, 0.05])
l2 = torch.tensor([0.3, 0.6, 0.75, 0.9])

list.append(l1)
list.append(l2)


c = torch.stack(list)

print(c)
print()

# -------------------------------------------------------------------------------------

l = torch.tensor([0.5, 0.2, 0.85, 0.05])
print(l)
print(type(l))

reverted = torch.argmax(l, dim=0)
print(reverted)


# One hot encoding buffer that you create out of the loop and just keep reusing
y_onehot = torch.zeros(4)

print(y_onehot)

one_hot = torch.nn.functional.one_hot(torch.tensor([reverted]), 4)
one_hot = torch.squeeze(one_hot, 0)
print(one_hot)


s =torch.stack([ torch.tensor([i]).repeat(15) for i in range(0,5)])
print(s)
print()



# -------------------------------------------------------------------------------------
from torch.utils.data import TensorDataset, DataLoader
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# if we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Preprocessed data paths
preprocessed_data_paths = {
    'training_rumors_tweets path':      '..\\data\\preprocessed data\\training\\rumors_tweets.npy',
    'training_rumors_labels path':      '..\\data\\preprocessed data\\training\\rumors_labels.npy',
    'training_stances_tweets path':     '..\\data\\preprocessed data\\training\\stances_tweets.npy',
    'training_stances_labels path':     '..\\data\\preprocessed data\\training\\stances_labels.npy',

}

train_data_rumors = TensorDataset(torch.from_numpy(np.load(preprocessed_data_paths['training_rumors_tweets path'])),
                                      torch.from_numpy(np.load(preprocessed_data_paths['training_rumors_labels path'])))

train_loader_rumors = DataLoader(train_data_rumors, shuffle=False, batch_size=28)


for inputs_rumors, labels_rumors in train_loader_rumors:
    n = torch.max(labels_rumors, 1)[1]
    print(n)

    for row in inputs_rumors:
        row_gpu = row.to(device)
        print(row)

# -------------------------------------------------------------------------------------




