# set the model to training mode
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    train_loss /= len(train_dataset)  # Average per sample [web:41][web:42]


# Set the model to evaluation mode
model.eval()
validation_loss = 0.0

with torch.no_grad():
    for features, labels in validationloader:
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Sum the current loss to the validation_loss variable
        validation_loss += loss.item()

# Set the model to evaluation mode
model.eval()
validation_loss = 0.0

with torch.no_grad():
    for features, labels in validationloader:
        outputs = model(features)
        loss = criterion(outputs, labels)
        # Sum the current loss to the validation_loss variable
        validation_loss += loss.item()

# Calculate the mean loss value
validation_loss_epoch = validation_loss / len(validationloader)
print(validation_loss_epoch)

# Set the model back to training mode
model.train()


# all these things can be replace by torchmetrics
from torchmetrics.classification import Accuracy

metric = Accuracy(task="multiclass", num_classes=10)
# In loop:
preds = torch.argmax(outputs, dim=1)
metric.update(preds, labels)
print(metric.compute())  # tensor(0.9234)
metric.reset()  # For next epoch
