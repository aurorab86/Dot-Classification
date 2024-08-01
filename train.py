import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from main import *

model = Model(num_neurons=15, num_classes=3)

optimizer = optim.SGD(model.parameters(), lr=0.1)

batch_size = 50

loss_values = []
acc_values = []

epoch = 1000
for epoch_cnt in range(epoch):
    idxs = np.arange(len(xtrain))
    np.random.shuffle(idxs)

    for batch_cnt in range(0, len(xtrain)//batch_size):
        batch_indices = idxs[batch_cnt * batch_size : (batch_cnt + 1) * batch_size]

        batch = xtrain[batch_indices]
        truth = ytrain[batch_indices]

        batch = torch.tensor(batch, dtype=torch.float32)
        truth = torch.tensor(truth, dtype=torch.long)

        pred = model.forward(batch)

        loss = F.cross_entropy(pred, truth)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        acc = accuracy(pred, truth)

    loss_values.append(loss.detach().clone().numpy())
    acc_values.append(acc)
    print(f"Epoch {epoch_cnt+1}----------------------------")
    print(f"loss: {loss.item():.6f}")
    print(f"acc: {acc:.6f}")




plt.figure()
plt.plot(loss_values, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(acc_values, label="accuracy")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.legend()
plt.grid()
plt.show()    


def dummy_function(x):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    return F.softmax(model(x), dim=1).detach().numpy()

fig, ax = data.visualize_model(dummy_function, entropy=False);