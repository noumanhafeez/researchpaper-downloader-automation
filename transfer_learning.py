# Transfer learning in pytorch.
# step 1: save weights and bias of any model
# step 2: load the weights
# step 3: freeze the layer. for example:
for name, param in model.named_parameters():

    # Check for first layer's weight
    if name == '0.weight':
        # Freeze this weight
        param.requires_grad = False

    # Check for second layer's weight
    if name == '1.weight':
        # Freeze this weight
        param.requires_grad = False


# use uniform initialization i.e convert weights to between 0 and 1 range
layer0 = nn.Linear(16, 32)
layer1 = nn.Linear(32, 64)

# Use uniform initialization for layer0 and layer1 weights
nn.init.uniform_(layer0.weight)
nn.init.uniform_(layer1.weight)

model = nn.Sequential(layer0, layer1)