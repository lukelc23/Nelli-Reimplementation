3000 training steps, 7 items_n, 20 output weights

h1s = np.zeros((n, self.h1_size)) = np.zeros((7,20))

all_h1s shape: (3000, 7, 20) - tensor

def extract_h1s(self):
    """Calculate network hidden state representations for all input items

    :return: Numpy array of size (items_n, h1_size) containing all hidden states
    """
    with torch.no_grad():
        n = self.items_n
        h1s = np.zeros((n, self.h1_size))
        for i in range(n):
            with torch.no_grad():
                h1s[i, :] = self.layer_1.weight[:,i].detach().numpy().copy()
        return h1



results["train"]["h1s"][seed, training_step] = model.extract_h1s()

all_h1s = np.mean(results["train"]["h1s"], axis=0)

distance = euclidean_distances(all_h1s[int(positions[i] * steps_after_training)])


