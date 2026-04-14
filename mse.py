import numpy as np

y_pred = np.array([3, 5.0, 2.5, 7.0])
y = np.array([3.0, 4.5, 2.0, 8.0])

# Calculate MSE using NumPy
mse_numpy = np.mean((y_pred - y)**2)

# Create the MSELoss function in PyTorch
criterion = nn.MSELoss()

# Calculate MSE using PyTorch
mse_pytorch = criterion(torch.tensor(y_pred), torch.tensor(y))

print("MSE (NumPy):", mse_numpy)
print("MSE (PyTorch):", mse_pytorch)



## Training Loop

# Loop over the number of epochs and then the dataloader

# Loop over the number of epochs and the dataloader
# Loop over the number of epochs and the dataloader
for i in range(num_epochs):
  for data in dataloader:
    # Set the gradients to zero
    optimizer.zero_grad()
    # Run a forward pass
    feature, target = data
    prediction = model(feature)
    # Compute the loss
    loss = criterion(prediction, target)
    # Compute the gradients
    loss.backward()
    # Update the model's parameters
    optimizer.step()
show_results(model, dataloader)



# Do not use sigmoid and softmax activation in hidden layer coz, it output the values between 0 and 1. Assume
# sigmoid give 0.0000000352, it's too short number for gradient, which can cause the problem of vanishing gradient.
# So, to overcome this issue, use relu or leaky relu activation in hidden layer and sigmoid or softmax in output layer
