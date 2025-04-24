import matplotlib.pyplot as plt

def plot_loss(train_losses, val_losses):
    """
    Plot training and validation loss over epochs.
    
    Parameters:
    - train_losses: List of training losses per epoch.
    - val_losses: List of validation losses per epoch.
    """

    # Check if the lengths of the lists are equal
    if len(train_losses) != len(val_losses):
        raise ValueError("The lengths of train_losses and val_losses must be equal.")

    # Plotting the losses
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
