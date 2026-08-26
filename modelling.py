import matplotlib.pyplot as plt

def display_loss(loss_history: list, plot_name: str) -> None:

    # Plot the loss over epochs to visualise how well the model is doing
    epochs = list(range(1, len(loss_history)+1))

    plt.figure()
    plt.plot(epochs, loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(plot_name)
