import matplotlib.pyplot as plt


def visualize(input_data, n_plots=None, image_shape=None):
    """
    Visualize images and weight matrices
    Args:
        input_data (nd.array): flatten input data
        n_plots (tuple or list): number plots in rows and cols
        image_shape (tuple or list): original shape of the image
    """
    print("Plotting results...")
    plt.figure()
    row, col = n_plots
    reshaped_data = [i.reshape(image_shape) for i in input_data]
    for i in range(1, len(reshaped_data) + 1):
        plt.subplot(row, col, i)
        plt.matshow(reshaped_data[i - 1], fignum=False)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.show()
