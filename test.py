import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.ReLU(),
    nn.Linear(20, 10)
)

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

input = torch.rand(10)
output = torch.rand(10)
my_target = torch.zeros(10)

def my_loss_fn(output, target):
    loss = torch.mean((output - target)**2)
    return loss 

for _ in range(10):
    model.train()

    y_pred = model(input)

    # loss = loss_fn(y_pred, output)
    loss = my_loss_fn(y_pred, my_target)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

# print('====== org input =====')
# print(input)
# print('====== org output =====')
# print(output)
# print('====== learnt output =====')
pred_out = model(input)
# print(pred_out.tolist())

torch.save(pred_out, 'models/test.pt')