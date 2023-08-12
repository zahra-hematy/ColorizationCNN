import torch
import torchvision
from PIL import Image
from model import *
from Dataset import * 
from torch.utils.data import Dataset
import torch.nn.functional as F           # layers, activations and more

loss_function = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, )
losses = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
num_epochs = 100
print('train')
for epoch in range(num_epochs):
    # Train:
    model.train()
    train_loss = 0.0
    for batch in loader:

        Ls, abs_ = batch['L'], batch['ab']
        Ls = Ls.to(device)
        abs_ = abs_.to(device)

        optimizer.zero_grad()
        # Output of Autoencoder
        outputs = model(Ls).to(device)


        # Resize the output to match the size of the target tensor (abs_)
        outputs = F.interpolate(outputs, size=abs_.size()[2:], mode='bilinear', align_corners=False)

        loss = loss_function(outputs, abs_)  # Slice the first two channels for the 'ab' components
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataset)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:  # Use test_loader for validation loop

            Ls, abs_ = batch['L'], batch['ab']
            Ls = Ls.to(device)
            abs_ = abs_.to(device)

            outputs = model(Ls)
            outputs = F.interpolate(outputs, size=abs_.size()[2:], mode='bilinear', align_corners=False)
            loss = loss_function(outputs, abs_)  
            val_loss += loss.item()

    val_loss /= len(test_dataset)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

print("Training completed.")

# Save the model
torch.save(model.state_dict(), "colorization_model.pt")
