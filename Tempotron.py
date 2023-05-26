import time
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import os
from pathlib import Path

# device: Device for running PyTorch computations (Optional: "cpu","cuda","mps")
# Warning!!! The "mps" device of Apple Silicon is currently unstable in Pytorch 2.0.0.
# You may encounter various bugs/errors with the "mps" device. Please use it with caution.
device = th.device("cuda:0")

class Tempotron:
    """
    A class representing a tempotron, as described in Gutig & Sompolinsky (2006).

    The (subthreshold) membrane voltage of the tempotron is a weighted sum from all incoming
    spikes and the resting potential of the neuron. The contribution of each spike decays
    exponentially with time, the speed of this decay is determined by two parameters tau and
    tau_s, denoting the decay time constants of membrane integration and synaptic currents,
    respectively.

    Note that we add some custom parameters to the model:
    - echo is used to control display output
    - dendrites_num is used to set the number of encoded neurons
    - A is used to control the range of Gaussian receiving field encoding
    """

    def __init__(self, V_rest, tau, tau_s, synaptic_efficacies, A, dendrites_num, echo=1.0, threshold=1.0, ):
        """
        Initializes the Tempotron object with the given parameters

        Parameters:
        - V_rest: The resting membrane potential
        - threshold: Neuron firing threshold (default: 1.0)
        - tau: Membrane integration time constant
        - tau_s: Synaptic current time constant
        - synaptic_efficacies: Synaptic efficacy values
        - A: Gaussian receiving field encoding range
        - dendrites_num: Number of a Tempotron neuron's dendrites
        - echo: Controls display output (default: 1.0)
        """
        self.V_rest = V_rest
        self.tau = float(tau)
        self.tau_s = float(tau_s)
        self.log_tts = np.log(self.tau / self.tau_s)
        self.threshold = threshold
        self.efficacies = synaptic_efficacies.to(device)
        self.A = A
        self.dendrites_num = dendrites_num

        # spike integration time, compute this with formula
        self.t_spi = 10

        # Initialize the weight update momentum with zeros
        self.dw_momentum = th.zeros(dendrites_num, ).to(device)

        # Compute normalisation factor V_0
        self.V_norm = self.compute_norm_factor(tau, tau_s)
        self.echo = echo

    def compute_norm_factor(self, tau, tau_s):
        """
        Compute and return the normalisation factor:

        V_0 = (tau * tau_s * log(tau/tau_s)) / (tau - tau_s)

        That normalises the method:
                
        K(t-t_i) = V_0 (exp(-(t-t_i)/tau) - exp(-(t-t_i)/tau_s)

        Such that it amplitude is 1 and the unitary PSP
        amplitudes are given by the synaptic efficacies.
        """
        tmax = (tau * tau_s * np.log(tau / tau_s)) / (tau - tau_s)
        if tmax < 0:
            v_max = 0
        else:
            v_max = 1 * (np.exp(-(tmax - 0) / self.tau) - np.exp(-(tmax - 0) / self.tau_s))
        V_0 = 1 / v_max
        return V_0

    def train(self, epoch, batchsize, learning_rate, momentum, noise_key,
              Gaussian_sigma, jitter_sigma, adding_missing_prob):
        """
        Train the tempotron on the given input-output pairs,
        applying gradient decscend to adapt the weights.

        :param epoch: the maximum number of training steps
        :param batchsize: The size of the training batch
        :param learning_rate: The learning rate of the gradient descent, given in the form of an interval
               The learning rate will decrease from the upper bound to the lower bound of the interval during training
        :param momentum: The momentum switch for weight updates
        :param noise_key: The key used to select the noise type to be added during training
               (optional: Gaussian, jitter, adding&missing)
        :param Gaussian_sigma: The standard deviation of Gaussian noise
        :param jitter_sigma: The standard deviation of jitter noise
        :param adding_missing_prob: The probability of adding or missing spikes
        """

        epoch_count = []
        test_loss_values = []
        train_loss_values = []

        # Encode program initialization and get encoded data
        train_data, test_data = self.generate_trian_data(noise_key, Gaussian_sigma, jitter_sigma, adding_missing_prob)
        train_spike_time = train_data[0]
        train_data_labels = train_data[1].flatten()
        [dendrites_num, Number, sample] = train_spike_time.shape
        random_indices = th.randperm(Number)
        train_data_labels = train_data_labels[random_indices]
        train_spike_time = train_spike_time[:, random_indices, :]

        print("Start training")
        train_start = time.time()
        for i in range(epoch):
            # Variable learning rate
            if i % round(epoch / 20) == 0 and i != 0:
                learning_rate[1] = (learning_rate[1] - learning_rate[0]) * 0.5 + learning_rate[0]

            # Adapt weights through the training set in random order
            epoch_start = time.time()
            for idx in range(int(Number / batchsize)):
                self.adapt_weights(train_spike_time[:, idx * batchsize: (idx + 1) * batchsize, :],
                                   train_data_labels[idx * batchsize:(idx + 1) * batchsize],
                                   batchsize=batchsize, learning_rate=learning_rate[1],
                                   momentum=momentum)
            epoch_end = time.time()

            # Test the training results
            if self.echo == 1:
                test_spike_time = test_data[0]
                test_data_labels = test_data[1].flatten()
                test_error_roit = self.test_batch(batchsize=10, user_defined=(test_spike_time, test_data_labels))
                train_error_roit = self.test_batch(batchsize=10, user_defined=(train_data[0], train_data[1]))
                epoch_count.append(i)
                train_loss_values.append(train_error_roit)
                test_loss_values.append(test_error_roit)
                # train_now = time.time()
                # print(f"Epoch: {i} completed | Total Time up to now: {train_now - train_start} s ")
                print(f"Epoch: {i} | Train Loss: {train_loss_values[i]:.4f} | Test Loss: {test_loss_values[i]:.4f} | "
                      f"Time: {epoch_end - epoch_start:.2f} s ")

                # Save updated synaptic efficacies
                EFFICACIES_PATH = Path("Efficacies")
                EFFICACIES_PATH.mkdir(parents=True, exist_ok=True)
                EFFICACIES_NAME = 'Epoch' + str(i) + '_efficacies.pt'
                EFFICACIES_SAVE_PATH = EFFICACIES_PATH / EFFICACIES_NAME
                # print(f"Saving efficacies to: {EFFICACIES_PATH}")
                th.save(obj=self.efficacies, f=EFFICACIES_SAVE_PATH)

        # Save training and testing loss values
        LOSS_PATH = Path("Loss")
        LOSS_PATH.mkdir(parents=True, exist_ok=True)
        train_end = time.time()
        print(f"Epoch: {i} completed | Total Training Time: {train_end - train_start:.2f} s ")
        print(f"Saving loss to: {LOSS_PATH}")
        th.save(obj=train_loss_values, f=LOSS_PATH / 'train_loss.pt')
        th.save(obj=test_loss_values, f=LOSS_PATH / 'test_loss.pt')

        # Plot and save the loss curves
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.plot(epoch_count, train_loss_values, label="Train loss")
        ax.plot(epoch_count, test_loss_values, label="Test loss")
        ax.set_title("Training and testing loss curves")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")
        ax.legend()
        fig.savefig(LOSS_PATH / "loss_curves.png", dpi=300)
        plt.show()
        return

    def generate_trian_data(self, noise, Gaussian_sigma, jitter_sigma, adding_missing_prob):
        """
        The method generates training data.

        It calls the 'noise_data_encode' method,
        which generates data with noise based on the noise type passed in and encodes it.

        Next, the method extracts the training data and labels,
        sends them to the 'process_code_list' method,
        and finally returns a list of the pulse-coding of the training data,
        the training data, and the test data.

        This method also takes an optional 'noise' parameter for generating data with specific types of noise.
        """
        train_data, test_data = self.noise_data_encode(noise, Gaussian_sigma, jitter_sigma, adding_missing_prob)
        return train_data, test_data

    def adapt_weights(self, spike_times, data_labels, batchsize, learning_rate, momentum):
        """
        Modify the synaptic efficacies such that the learns to classify the input pattern correctly.
        Whenever the Tempotron's spike output is different from the desired output,
        the following update is computed:

        dw = lambda sum_{ti} K(t_max, ti)

        The synaptic efficacies are increased by this weight if the tempotron did erroneously
        not elecit an output spike, and decreased if it erroneously did.

        :param spike_times: a matrix with encoded spike times for every signal and every dendrite (also called afferent)
        :param data_labels: the classification of the input pattern
        :param batchsize: The size of the training batch
        :param learning_rate: The learning rate of the gradient descent, given in the form of an interval
               The learning rate will decrease from the upper bound to the lower bound of the interval during training
        :param momentum: The momentum switch for weight updates
        """

        # Calculate the tmax
        tmax = self.compute_tmax(spike_times)

        # Extract the dimensions of the spike_times tensor
        [neuron_dim, signal_dim, sample] = spike_times.shape

        # Reshape tmax to prepare for the upcoming calculation
        tmax = tmax.view(signal_dim, 1, 1)
        tmax = tmax.repeat(1, 1, neuron_dim * sample).view(signal_dim, 1, neuron_dim, sample)

        # Transform spike_times dimensions for the upcoming calculation
        t_i = spike_times.transpose(0, 1)
        t_i = t_i.unsqueeze(1).repeat(1, 1, 1, 1)

        # Calculate the vmax and output the result
        vmax = self.compute_membrane_potential(tmax, t_i)
        vmax = vmax.squeeze(1)
        vmax_bool = (vmax >= self.threshold).to(th.int)
        data_labels = th.from_numpy(data_labels).to(device)

        # Check if the Tempotron correctly classified the input pattern
        if th.all(vmax_bool.eq(data_labels)):
            # No need for update
            return
        elif not th.any(vmax_bool.eq(data_labels)):
            # All signals in the current batch need update
            pass
        else:
            # Prepare wrongly classified signals for upcoming update
            adjust_signal_idx = (vmax_bool != data_labels).to(th.int)
            mask = adjust_signal_idx == 1
            tmax = tmax[mask]
            data_labels = data_labels[mask]
            t_i = t_i[mask]

        # Compute the synaptic efficacies update
        dw = (self.dw(learning_rate, tmax, t_i))
        dw[data_labels == 0, :] *= -1
        dw = dw.sum(dim=0) / batchsize

        # Apply momentum to speed up training
        if momentum == "on":
            momentum_factor = 0.1
            self.dw_momentum = momentum_factor * self.dw_momentum + (1 - momentum_factor) * dw
            dw = self.dw_momentum

        # Update the synaptic efficacies
        self.efficacies += dw

    def dw(self, learning_rate, tmax, spike_times):
        """
        Compute the update for synaptic efficacies, according to the following gradient-based learning rule:

        dwi = lambda sum_{ti} K(t_max, ti)

        where lambda is the learning rate and t_max denotes the time at which the postsynaptic potential V(t)
        reached its maximal value.

        For multiple signal inputs in a batch, the dwi of each signal is calculated separately.
        """
        # Compute the contributions of the individual spikes at time tmax
        spike_contribs = self.compute_spike_contributions(tmax, spike_times)

        # Multiply with learning rate to get updates
        update = (learning_rate * spike_contribs).squeeze(1)

        return update

    def test_batch(self, batchsize, user_defined=None):
        """
        This method tests the model's accuracy on the validation dataset.
        It takes the batch size (batchsize) and an optional user-defined dataset (user_defined), and returns the error
        rate. If user-defined dataset is not given, it will automatically read the dataset in the "validation_dataset".
        
        This method  returns the classification error rate (also called loss) of the given dataset.
        """
        result = []
        correct_number = 0
        efficacies_len = len(self.efficacies)
        assert self.dendrites_num == efficacies_len, "The number of efficacies does not match the number of " \
                                                     "Tempotron's dendrites (afferents), and therefore synapses " \
                                                     "cannot be formed. "

        if user_defined is None:
            data, data_labels = self.read_file(pattern="validation")
            [Number, sample] = data.shape
            spike_time = self.encode_data(data)
            [dendrites_num, Number, sample] = spike_time.shape

        else:
            spike_time = user_defined[0]
            data_labels = user_defined[1].flatten()
            [dendrites_num, Number, sample] = spike_time.shape

        for idx in range(int(Number / batchsize)):
            # print("Running: ", idx)
            if idx == 0:
                result, N = self.test(spike_time[:, idx * batchsize: (idx + 1) * batchsize, :],
                                      data_labels[idx * batchsize:(idx + 1) * batchsize])
                correct_number += N
            else:
                x_result, N = self.test(spike_time[:, idx * batchsize: (idx + 1) * batchsize, :],
                                        data_labels[idx * batchsize: (idx + 1) * batchsize])
                result = th.cat((result, x_result), dim=0)
                correct_number += N
        Correct_roit = (np.sum(correct_number) / Number) * 100
        return 1 - (Correct_roit / 100)

    def test(self, spike_times, data_labels):
        """
        This method is used to compute the membrane potential of neurons based on the input of neuron spiking times
        and convert the membrane potential to binary prediction results using a pre-defined threshold.

        The core part of the method involves reshaping and dimension transformation of data to compute the membrane
        potential of neurons. The membrane potential is then converted to binary results using a pre-defined threshold.
        Finally, the method compares the predicted results with the labeled data to compute the classification accuracy
        and returns the predicted results as a tensor and the number of correctly classified samples.
        """

        # Calculate the tmax
        tmax = self.compute_tmax(spike_times)

        # Extract the dimensions of the spike_times tensor
        [neuron_dim, signal_dim, sample] = spike_times.shape

        # Reshape tmax to prepare for the upcoming calculation
        tmax = tmax.view(signal_dim, 1, 1)
        tmax = tmax.repeat(1, 1, neuron_dim * sample).view(signal_dim, 1, neuron_dim, sample)

        # Transform spike_times dimensions for the upcoming calculation
        t_i = spike_times.transpose(0, 1)
        t_i = t_i.unsqueeze(1).repeat(1, 1, 1, 1)

        # Calculate the vmax
        vmax = self.compute_membrane_potential(tmax, t_i)

        # Gamma is True, Neutron is False
        Result = (vmax > self.threshold).cpu().numpy().astype(int)

        # Calculate the number of correctly classified samples
        correct_number = np.count_nonzero(Result == (np.expand_dims(data_labels, axis=1)))

        return th.tensor(Result), correct_number

    def K(self, V_0, t, t_i):
        """
        This method calculates the contribution of each neuron to neural signals.
        It computes a value based on input parameters V_0, t, t_i, and tau_s.

        It masks NaN values in t_i and values with t less than t_i,
        then assigns 0 to the corresponding elements of the masked values array.

        The NaN denotes a spiking encoding window that did not encode a spike.

        Finally, it computes the contribution of each neuron to neural signals
        and returns the result.
        """
        # Calculate K value
        value = (V_0 * (th.exp(-(t - t_i) / self.tau) - th.exp(-(t - t_i) / self.tau_s))).clone().detach().to(device)

        # Mask NaN values in t_i and values with t less than t_i
        mask_t_i = th.isnan(t_i) | (t < t_i)

        # Mask out the value of 0 in t
        mask_t = th.eq(t, 0)
        value[mask_t_i] = 0
        value[mask_t] = 0

        # Calculate the contribution of each neuron to each signal (by a summation process)
        value = th.as_tensor(value).clone().detach().to(device)
        spike_contribs = th.sum(value, dim=value.dim() - 1)

        return spike_contribs

    def compute_membrane_potential(self, t, t_i):
        """
        This method calculates the membrane potential of neurons.

        It first obtains the signal dimension, maximum time dimension, neuron
        dimension, and number of samples by getting the shape information of the t_i matrix.

        Then, it calls the compute_spike_contributions method to calculate the contribution
        of neurons to neural signals. The spike_contribs returned are multiplied with synaptic
        efficacies and expanded in signal, maximum time, and neuron dimensions. The resulting value represents
        the total incoming potential for neurons between each time period.

        Finally, the sum of all total_incoming is taken over neuron dimension and added to the resting
        potential value (V_rest) to calculate the membrane potential (V) for each neuron.

        The method returns the calculated membrane potential values.
        """
        [signal_dim, t_max_dim, neuron_dim, sample] = t_i.shape

        spike_contribs = self.compute_spike_contributions(t, t_i)

        # Multiply with the synaptic efficacies
        efficacies = th.as_tensor(self.efficacies, device=device).clone().detach().view(1, neuron_dim)
        efficacies = efficacies.repeat(signal_dim, t_max_dim, 1)

        total_incoming = spike_contribs * efficacies

        # Add the sum of total_incoming and V_rest to get the membrane potential
        V = th.sum(total_incoming, dim=2) + th.tensor(self.V_rest, device=device)

        return V

    def compute_spike_contributions(self, t, t_i):
        """
        Calculate the contribution of neurons.
        """
        # Convert self.V_norm to a PyTorch tensor
        V_norm = th.tensor(self.V_norm, device=device)

        # Compute spike_contribs
        spike_contribs = self.K(V_norm, t, t_i)

        return spike_contribs

    def compute_tmax(self, spike_times):
        """
        This method calculates the tmax, the time at which the membrane potential of a neuron
        reaches its maximum value.

        It first extracts the dimensions of neurons and signals from the input variable spike_times.
        Then, it converts each signal into a row and stores the output weights of each signal, as well
        as the positions of NaN values in the variables weights and mask. Next, it sorts the signals in
        each row and calculates the corresponding τ, τ_s, and div for the neuron's output weights.

        The NaN denotes a spiking encoding window that did not encode a spike.

        It also identifies the locations where div values are less than or equal to 0 and sets the tmax
        at these locations to the time values in the signal. Finally, it computes the time of the maximum
        membrane potential generated by each triggering signal on the neuron, and returns it as tmax.
        """
        spike_times2 = spike_times
        [neuron_dim, signal_dim, sample] = spike_times.shape

        # Convert each signal into a row
        times = spike_times.transpose(0, 1).flatten(start_dim=1).clone().detach().to(device, dtype=th.float32)

        # Replicate efficacies to match the neuron and signal dimensions
        weights = self.efficacies.repeat(sample).view(1, -1).repeat(signal_dim, 1).clone().detach().to(device,
                                                                                                       dtype=th.float32)

        # Record NaN value positions in the mask matrix
        mask = ~th.isnan(times)
        # Convert bool values to 0 or 1
        mask = th.where(mask, th.tensor([1], device=device), th.tensor([0], device=device))
        # Replace NaN values in the time matrix with 0 for subsequent calculations
        times = th.nan_to_num(times, nan=0.0)

        # Sort each row in times in descending order
        times, sorted_indices = th.sort(times, dim=1)
        weights = weights.gather(1, sorted_indices)
        mask = mask.gather(1, sorted_indices)

        # Truncate unnecessary signals to speed up the calculation
        truncate_location = th.sum(times == 0, axis=1)
        times = times[:, truncate_location[th.argmin(truncate_location, dim=0)]:]
        weights = weights[:, truncate_location[th.argmin(truncate_location, dim=0)]:]

        # Mask also needs to be intercepted
        mask = mask[:, truncate_location[th.argmin(truncate_location, dim=0)]:]

        # Compute τ and τ_s and div; set div to 0 if it's NaN
        tau = (weights * th.exp(times / self.tau)) * mask
        tau_s = (weights * th.exp(times / self.tau_s)) * mask

        sum_tau = th.cumsum(tau, dim=1)
        sum_tau_s = th.cumsum(tau_s, dim=1)

        div = th.nan_to_num(sum_tau_s / sum_tau, nan=0.0)

        # Handle boundary cases where div is less than or equal to 0
        boundary_cases = (div <= 0)
        div[boundary_cases] = 10

        tmax_list = (
                    self.tau * self.tau_s * (self.log_tts + th.log(div)) / (self.tau - self.tau_s)).clone().detach().to(
            device=device, dtype=th.float32)

        tmax_list[boundary_cases] = times[boundary_cases]

        # Back up a variable temp that will later be used to index the maximum value
        temp = tmax_list

        # Transform the tmax_list to calculate vmax
        [signal_dim, n] = tmax_list.shape

        tmax_list = tmax_list.view(signal_dim, n, 1)
        tmax_list = tmax_list.repeat(1, 1, neuron_dim * sample).view(signal_dim, n, neuron_dim, sample)

        t_i = spike_times2.transpose(0, 1)
        t_i = t_i.unsqueeze(1).repeat(1, n, 1, 1)

        vmax_list = self.compute_membrane_potential(tmax_list, t_i)

        # Find the index with the maximum value along the second axis
        max_values, max_indices = th.max(vmax_list, dim=1)

        # Extract the corresponding element in the temp tensor using the maximum value indices
        tmax = temp.gather(1, max_indices.unsqueeze(1)).squeeze(1)
        tmax = tmax.view(signal_dim, 1)

        return tmax

    @staticmethod
    def read_file(pattern=None):
        """
        This static method reads data files. If the parameter "pattern" is "validation",
        it reads the validation dataset file; otherwise, it reads the training dataset file.

        The dataset txt file is read using the loadtxt method of the NumPy library and stored
        in variables "data" and "data_labels". These two variables are combined into a matrix "Data",
        which is shuffled using a random permutation of its rows. The first (sample-1) columns
        of the matrix are stored in the variable "data", while the last column is stored in the variable "data_labels".

        Finally, the method returns the variables "data" and "data_labels".

        This method is a static method, meaning it can be called using the class name without initialization.
        """

        if pattern == "validation":
            data = np.loadtxt(
                os.path.abspath(".") + os.sep +
                "Dataset" + os.sep + "dataset" + os.sep + "validation_dataset" + os.sep + "validation_data_normalized.txt",
                dtype=np.float32,
                delimiter=',')

            data_labels = np.loadtxt(
                os.path.abspath(
                    ".") + os.sep + "Dataset" + os.sep + "dataset" + os.sep + "validation_dataset" + os.sep + "test_labels.txt",
                dtype=np.float32,
                delimiter=','
            )
        else:
            data = np.loadtxt(
                os.path.abspath(".") + os.sep +
                "Dataset" + os.sep + "dataset" + os.sep + "training_dataset" + os.sep + "training_data_normalized.txt",
                dtype=np.float32,
                delimiter=',')

            data_labels = np.loadtxt(
                os.path.abspath(
                    ".") + os.sep + "Dataset" + os.sep + "dataset" + os.sep + "training_dataset" + os.sep + "train_labels.txt",
                dtype=np.float32,
                delimiter=','
            )

        return data, data_labels

    def noise_data_encode(self, noise, Gaussian_sigma, jitter_sigma, adding_missing_prob):
        """
        This method reads data from a file,
        splits it into training and testing sets, adds various types of noise to the data (if required),
        encodes the data using latency encoding and then Gaussian encoding on GPU,
        and returns the final spike-encoded train and test data.

        The method takes 'noise' as a parameter,
        which can be a combination of 'Gaussian', 'jitter', 'adding&missing' (noise types to be added to data).

        It is worth noting that all spike encoding windows that contain no spike are encoded to NaN.
        """

        # Reading data from file and obtaining data dimension
        orignal_data, orignal_data_labels = self.read_file()

        [signal_num_all, sample_all] = orignal_data.shape
        orignal_data_labels = orignal_data_labels.reshape(signal_num_all, 1)

        # Splice into a matrix for easy scrambling
        Data = np.concatenate((orignal_data, orignal_data_labels), axis=1)

        # scrambled row
        Data = Data[np.random.permutation(Data.shape[0])]

        # Separate data and data labels from Data
        data = Data[:, 0:sample_all - 1]
        data_labels = Data[:, sample_all:sample_all + 1]
        data_labels = data_labels.flatten()

        # Splitting data into train and test sets
        train_num = round(signal_num_all * 0.8)
        train_data = data[0:train_num, :]
        train_data_temp = train_data
        train_data_labels = data_labels[0:train_num]
        train_data_labels_temp = train_data_labels
        test_data = data[train_num:signal_num_all, :]
        test_data_labels = data_labels[train_num:signal_num_all]

        # Defining number of neurons and obtaining data dimensions
        neuron = self.dendrites_num
        [signal_num, sample] = train_data.shape

        # Encoding data using latency encoding and then guassian encoding on GPU
        Gaussian_spike_time = self.encode_data(train_data)

        # Setting NaN values to spike times equal to zero
        NaN_matrix = th.ones(Gaussian_spike_time.size(), dtype=th.float16).to(device) * float('nan')
        Gaussian_spike_time = th.where(th.Tensor.eq(Gaussian_spike_time, 0), NaN_matrix, Gaussian_spike_time)

        Gaussian_spike_time_temp = Gaussian_spike_time

        # Adding guassian noise if required
        if "Gaussian" in noise:
            mu = 0
            sigma = Gaussian_sigma

            # Adding noise to data and encoding it again
            gaussian_noise_data = np.random.normal(mu, sigma, (signal_num, sample)).reshape(
                (signal_num, sample)) + train_data_temp
            Gaussian_spike_time_gaussian = self.encode_data(gaussian_noise_data)
            NaN_matrix_gaussian = th.ones(Gaussian_spike_time_gaussian.size(), dtype=th.float16).to(device) * float(
                'nan')
            Gaussian_spike_time_gaussian = th.where(th.Tensor.eq(Gaussian_spike_time_gaussian, 0), NaN_matrix_gaussian,
                                                    Gaussian_spike_time_gaussian)

            # Concatenating the original and noise-affected data
            Gaussian_spike_time = th.cat((Gaussian_spike_time, Gaussian_spike_time_gaussian), dim=1)
            train_data_labels = np.concatenate((train_data_labels, train_data_labels_temp), axis=0)
            print("Gaussian noise has been added")

        # Adding jitter noise if required
        if "jitter" in noise:
            mu = 0
            sigma = jitter_sigma

            # Adding random noise to spiked data and encoding it again
            Gaussian_spike_time_jetter = th.tensor(np.random.normal(mu, sigma, (neuron, signal_num, sample)),
                                                   device=device) + Gaussian_spike_time_temp
            Gaussian_spike_time = th.cat((Gaussian_spike_time, Gaussian_spike_time_jetter), dim=1)
            train_data_labels = np.concatenate((train_data_labels, train_data_labels_temp), axis=0)
            print("jitter noise has been added")

        # Adding adding&missing noise if required
        if "adding&missing" in noise:
            p = adding_missing_prob
            time_window = 279
            A1 = self.A * time_window

            # Adding random noise to unspiked data
            random_matrix = np.random.rand(neuron, signal_num, sample)
            adding_m = random_matrix < p
            mask_value = th.isnan(Gaussian_spike_time_temp)
            adding_spike = th.tensor(np.random.rand(neuron, signal_num, sample) * A1 * adding_m.astype(int),
                                     device=device) * mask_value
            Gaussian_spike_time_adding = th.where(th.isnan(Gaussian_spike_time_temp),
                                                  th.zeros_like(Gaussian_spike_time_temp), Gaussian_spike_time_temp)
            Gaussian_spike_time_adding = Gaussian_spike_time_adding + adding_spike
            Gaussian_spike_time_adding = th.where(Gaussian_spike_time_adding == 0, th.tensor(np.nan, device=device),
                                                  Gaussian_spike_time_adding)

            # Concatenating the original and noise-affected data
            Gaussian_spike_time = th.cat((Gaussian_spike_time, Gaussian_spike_time_adding), dim=1)
            train_data_labels = np.concatenate((train_data_labels, train_data_labels_temp), axis=0)

            # Making some elements in spiked data equal to zero
            random_matrix = np.random.rand(neuron, signal_num, sample)
            missing_m = (random_matrix < p).reshape(signal_num, neuron * sample)
            Gaussian_spike_time_missing = Gaussian_spike_time_temp.reshape(signal_num, neuron * sample)
            Gaussian_spike_time_missing[missing_m] = 0
            Gaussian_spike_time_missing = Gaussian_spike_time_missing.reshape(neuron, signal_num, sample)
            Gaussian_spike_time_missing = th.where(Gaussian_spike_time_missing == 0, th.tensor(np.nan, device=device),
                                                   Gaussian_spike_time_missing)

            # Concatenating the original and noise-affected data
            Gaussian_spike_time = th.cat((Gaussian_spike_time, Gaussian_spike_time_missing), dim=1)
            train_data_labels = np.concatenate((train_data_labels, train_data_labels_temp), axis=0)
            print("missing&adding noise has been added")

        # Obtaining the final spike-encoded train and test data
        train_data_spike = Gaussian_spike_time
        train_data_labels = train_data_labels
        Gaussian_test_spike_time = self.encode_data(test_data)
        test_data_spike = Gaussian_test_spike_time

        # Returning the final train and test data
        return (train_data_spike, train_data_labels), (test_data_spike, test_data_labels)

    def encode_data(self, data):
        """
        The method encodes data and returns the Gaussian pulse code.
        It takes the 'data' parameter, performs latency encoding on the data using the 'Latency_encoding' method,
        and then performs Gaussian encoding on the encoded data using the 'Gaussian_encoding' method.
        Finally, it returns the Gaussian pulse code.

        It is worth noting that all spike encoding windows that contain no spike are encoded to NaN.
        """
        Latency_spike_time, Latency_labels = self.Latency_encoding(data)
        Gaussian_spike_time = self.Gaussian_encoding(Latency_spike_time, Latency_labels)

        return Gaussian_spike_time

    def gaussian(self, x, mu, sig):
        """
        This method calculates the Gaussian value given input x, mean(mu), and standard deviation(sig).
        It returns the computed Gaussian value multiplied by a constant factor A (self.A).
        """
        return th.exp(-th.pow(x - mu, 2.) / (2 * th.pow(sig, 2.))) * self.A

    @staticmethod
    def Latency_encoding(all_data):
        """
        This method performs Latency encoding on data. First, all data is transferred to the
        GPU (assuming it is available) and normalized.

        Then, thresholds (T_1 and T_2) and a time window are determined, and the time window is used to encode the
        occurrence time of triggering signals. The range of this time window is from 1 to the length of the signal,
        and it is subtracted from the normalized data to obtain Latency_spike_time.

        Next, Latency_spike_time between the threshold values of T_1 and T_2 is assigned a value of 1,
        while in other cases it is assigned a value of 0. These labels are stored in Latency_labels.

        After obtaining Latency_labels, it is multiplied with Latency_spike_time to obtain the occurrence time of
        the neuron's excitation signal within the time window.

        Finally, any occurrence time that is not recorded in Latency_spike_time is returned as NaN,
        and Latency_spike_time and Latency_labels are returned as the result.
        """
        # Transfer the data to the GPU, data-type: Tensor
        Data = th.tensor(all_data, device=device)

        # m represents the number of signals
        # n represents the length of signals
        Min = Data.min(dim=1)[0].unsqueeze(1)
        Max = Data.max(dim=1)[0].unsqueeze(1)
        # Normalize data
        normalized_DATA = (Data - Min) / (Max - Min)

        # When the stimulus intensity was small enough, no coding was performed
        T_1 = 0.0001  # Threshold
        T_2 = 0.999

        # A time window was generated for encoding
        time_window = th.arange(Data.shape[1], device=device)

        # The time window is transformed to 1 to signal length
        Latency_spike_time = (time_window + 1 - normalized_DATA)

        Latency_labels_1 = Latency_spike_time - time_window > T_1
        Latency_labels_2 = Latency_spike_time - time_window < T_2
        Latency_labels = Latency_labels_1 * Latency_labels_2
        Latency_labels = Latency_labels.to(dtype=th.float)

        Latency_spike_time = Latency_spike_time * Latency_labels
        Latency_spike_time[Latency_spike_time == 0] = float('nan')

        return Latency_spike_time, Latency_labels

    def Gaussian_encoding(self, Latency_spike_time: th.Tensor, Latency_labels: th.Tensor) -> th.Tensor:
        """
        This method is used for Gaussian encoding.

        The method takes two tensors Latency_spike_time and Latency_labels as parameters.

        The operation of the method can be divided into the following main parts:

            Parameter initialization: Initialize some variables including beta, Imax, Imin, m, n, mu, j and sig among others.

            Build mu matrix: Compute two different time Imin and Imax's mu values that will be used to construct
            Gaussian distribution and used for encoding.

            Building Gaussian method for encoding: After initializing the user's weight vector,
            create Gaussian_spike_time, whose value will be a Gaussian distribution computed in the
            connection matrix (weight matrix).
            The vector will be three-dimensional (n, m, dendrites_num) and will be used to compute the total
            input of neurons. That is, Latency_spike_time and mu matrix are iteratively added to Gaussian_spike_time.

            Threshold processing: To produce a sparse neural activity pattern,
            the output y value is truncated and transformed to the (0,1) range.
            Two thresholds T_1 and T_2 are set to reduce quantization noise and remove noise.
            After this, binary matrix processing is performed based on the threshold,
            so that the output is generated only when the y value is higher than T_1 and lower than T_2.
            Negative information is removed from it.

            Elimination of uncoded information: Convert Latency_labels into a time window in the Latency_labels * (0,1)
            range that will be used to move time. Subtract the time window from the spike time Gaussian_spike_time.
            Gaussian_spike_time is set based on Gaussian distribution calculated from Gaussian_spike_time, mu and sig.
            After calculation, extra values are removed from the matrix to remove uncoded information.

            Replace the 0 values in Gaussian_spike_time with NaN.

            Return the method with Gaussian_spike_time as the output.

        The method also calls a method named gaussian, which will be used to compute the Gaussian distribution.
        """
        # parameter initialization
        beta = 2
        Imax = 1
        Imin = 0
        m, n = Latency_spike_time.size()
        mu = th.zeros(self.dendrites_num, m, n, device=device)
        j = th.arange(self.dendrites_num, device=device)
        sig = th.tensor(1 / beta * (Imax - Imin) / (self.dendrites_num - 2), device=device)

        # Building mu matrix
        mu_dimension_one = (Imin + (2 * j - 3) / 2 * (Imax - Imin) / (self.dendrites_num - 2))
        mu[:] = mu_dimension_one.reshape(-1, 1, 1)

        # Gaussian method for encoding
        Gaussian_spike_time = th.zeros(self.dendrites_num, m, n, device=device)

        # Building 3-dimension GRF_encoding matrix
        Gaussian_spike_time[:] = Latency_spike_time.unsqueeze(0).expand_as(Gaussian_spike_time)

        # Threshold
        T_1 = 0.0001
        T_2 = 0.9

        output = th.zeros(self.dendrites_num, m, n, device=device)
        time_window = th.arange(n, device=device).unsqueeze(0) * Latency_labels

        # Transfer time window
        Gaussian_spike_time = (Gaussian_spike_time - time_window)

        # Gaussian encode
        y = th.round(self.gaussian(Gaussian_spike_time, mu, sig) * 100) / 100

        y_ = y.view(1, -1)
        GRF_labels_1 = y_.clone().gt(T_1)
        GRF_labels_2 = y_.clone().lt(T_2)
        GRF_labels_1.mul_(GRF_labels_2)
        y_.mul_(GRF_labels_1)

        # Uncoded information is removed
        time_window = time_window.unsqueeze(0).expand_as(Gaussian_spike_time).reshape(1, -1)
        time_window.mul_(GRF_labels_1).mul_(GRF_labels_2)
        time_window = time_window.view(self.dendrites_num, m, n)
        Gaussian_spike_time = y_.view(self.dendrites_num, m, n) + time_window

        Gaussian_spike_time[th.eq(Gaussian_spike_time, 0)] = float('nan')

        return Gaussian_spike_time

    #########################################################PLOT#######################################################
    """
    The following methods are used to plot the internal activity of a Tempotron neuron, which run on the CPU.
    These methods draw on many techniques from the Tempotron project "https://github.com/dieuwkehupkes/Tempotron".
    One can utilize the following methods to research the behavior of Tempotron.
    """

    def K_plot(self, V_0, t, t_i):
        """
        Compute the method

        K(t-t_i) = V_0 (exp(-(t-t_i)/tau) - exp(-(t-t_i)/tau_s)
        """
        if t < t_i:
            value = 0
        else:
            value = V_0 * (np.exp(-(t - t_i) / self.tau) - np.exp(-(t - t_i) / self.tau_s))
        return value

    def compute_spike_contributions_plot(self, t, spike_times):
        """
        Compute the decayed contribution of the incoming spikes.
        """
        # nr of synapses
        N_synapse = len(spike_times)
        # loop over spike times to compute the contributions
        # of individual spikes
        spike_contribs = np.zeros(N_synapse)
        for neuron_pos in range(N_synapse):
            for spike_time in spike_times[neuron_pos]:
                # print self.K(self.V_rest, t, spike_time)
                spike_contribs[neuron_pos] += self.K_plot(self.V_norm, t, spike_time)
        return spike_contribs

    def compute_membrane_potential_plot(self, t, spike_times):
        """
        Compute the membrane potential of the neuron given
        by the method:

        V(t) = sum_i w_i sum_{t_i} K(t-t_i) + V_rest

        Where w_i denote the synaptic efficacies and t_i denote
        ith dendrite.

        :param spike_times: an array with at position i the spike times of
                            the ith dendrite
        :type spike_times: numpy.ndarray
        """
        # create an array with the contributions of the
        # spikes for each synaps
        spike_contribs = self.compute_spike_contributions_plot(t, spike_times)

        # multiply with the synaptic efficacies
        total_incoming = spike_contribs * self.efficacies

        # add sum and add V_rest to get membrane potential
        V = total_incoming.sum() + self.V_rest

        return V

    def get_membrane_potentials(self, t_start, t_end, spike_times, interval=0.1):
        """
        Get a list of membrane potentials from t_start to t_end
        as a result of the inputted spike times.
        """
        # vector  # create ised version of membrane potential method
        potential_vect = np.vectorize(self.compute_membrane_potential_plot)
        # exclude spike times from being vectorised
        potential_vect.excluded.add(1)

        # compute membrane potentials
        t = np.arange(t_start, t_end, interval)
        membrane_potentials = potential_vect(t, spike_times)

        return t, membrane_potentials

    def get_derivatives(self, t_start, t_end, spike_times, interval=0.1):
        """
        Get a list of the derivative of the membrane potentials from
        t_start to t_end as a result of the inputted spike times.
        """
        # create a vectorised version of derivative method
        deriv_vect = np.vectorize(self.compute_derivative)
        # exclude spike times from being vectorised
        deriv_vect.excluded.add(1)

        # compute derivatives
        t = np.arange(t_start, t_end, interval)
        derivatives = deriv_vect(t, spike_times)

        return t, derivatives

    def plot_membrane_potential(self, t_start, t_end, spike_times, interval=0.1):
        """
        Plot the membrane potential between t_start and t_end as
        a result of the input spike times.
        :param t_start: start time in ms
        :param t_end: end time in ms
        :param interval: time step at which membrane potential is computed
        """
        # compute membrane_potential
        t, membrane_potentials = self.get_membrane_potentials(t_start, t_end, spike_times, interval)

        # format axes
        plt.xlabel('Time (ms)')
        plt.ylabel('V(t)')

        ymax = max(membrane_potentials.max() + 0.1, self.threshold + 0.1)
        ymin = min(membrane_potentials.min() - 0.1, -self.threshold - 0.1)
        plt.ylim(ymax=ymax, ymin=ymin)
        plt.axhline(y=self.threshold, linestyle='--', color='k')

        # plot membrane potential
        plot = plt.plot(t, membrane_potentials)

        # return plot
        return membrane_potentials
        # plt.show()

    def plot_potential_and_derivative(self, t_start, t_end, spike_times, interval=0.1):
        """
        Plot the membrane potential and the derivative of the membrane
        potential as a result of the input spikes between t_start and
        t_end.
        :param t_start: start time in ms
        :param t_end: end time in ms
        """
        # compute membrane potentials
        t, membrane_potentials = self.get_membrane_potentials(t_start, t_end, spike_times, interval)

        # compute derivatives
        t, derivatives = self.get_derivatives(t_start, t_end, spike_times, interval)

        # format axes
        plt.xlabel('Time(ms)')
        # ylabel???

        ymax = max(membrane_potentials.max() + 0.1, self.threshold + 0.1)
        ymin = min(membrane_potentials.min() - 0.1, -self.threshold - 0.1)
        plt.ylim(ymax=ymax, ymin=ymin)

        plt.axhline(y=self.threshold, linestyle='--', color='k')
        plt.axhline(y=0.0, linestyle='--', color='r')
        plt.axvline(x=16.5, color='b')

        # plot
        plt.plot(t, membrane_potentials, label='Membrane potential')
        plt.plot(t, derivatives, label='Derivative')
        plt.show()

    def compute_derivative(self, t, spike_times):
        """
        Compute the derivative of the membrane potential
        of the neuron at time t.
        This derivative is given by:

        V'(t) = V_0 sum_i w_i sum_{t_n} (exp(-(t-t_n)/tau_s)/tau_s - exp(-(t-t_n)/tau)/tau)

        for t_n < t
        """
        # sort spikes in chronological order
        spikes_chron = [(time, synapse) for synapse in range(len(spike_times)) for time in spike_times[synapse]]
        spikes_chron.sort()

        # Make a list of spike times and their corresponding weights
        spikes = [(s[0], self.efficacies[s[1]]) for s in spikes_chron]

        # At time t we want to incorporate all the spikes for which
        # t_spike < t
        sum_tau = np.array([spike[1] * np.exp(spike[0] / self.tau) for spike in spikes if spike[0] <= t]).sum()
        sum_tau_s = np.array([spike[1] * np.exp(spike[0] / self.tau_s) for spike in spikes if spike[0] <= t]).sum()

        factor_tau = np.exp(-t / self.tau) / self.tau
        factor_tau_s = np.exp(-t / self.tau_s) / self.tau_s

        deriv = self.V_norm * (factor_tau_s * sum_tau_s - factor_tau * sum_tau)

        return deriv

    @staticmethod
    def remove_nan(p, data):
        """
        The method removes NaN values from data.
        It replaces NaN values with -1, converts the data to a list,
        and iterates through each element of the list. Within each element,
        if there is a -1 value, it removes it and converts it to a set. Finally,
        the data is converted back to a numpy array and returned.
        """
        import numpy as np
        data = np.nan_to_num(data, nan=-1)
        data = data.tolist()
        for k in range(0, p):
            while -1 in data[k]:
                data[k].remove(-1)
                data[k] = list(set(data[k]))
        data = np.array(data, dtype=object)
        return data


if __name__ == '__main__':
    import Tempotron as Tp

    device = th.device("cuda:0")

    # Parameter setting
    tau = 8.4
    tau_s = 2.1

    # Load efficacies
    EFFICACIES_PATH = Path("Efficacies")
    EFFICACIES_PATH.mkdir(parents=True, exist_ok=True)
    EFFICACIES_NAME = "Epoch299_efficacies.pt"
    EFFICACIES_SAVE_PATH = EFFICACIES_PATH / EFFICACIES_NAME
    efficacies = th.load(EFFICACIES_SAVE_PATH)

    # Initialize Tempotron
    tempotron = Tp.Tempotron(0, tau, tau_s, efficacies, A=1, dendrites_num=25, echo=1, threshold=1.0)

    # Start validation
    start = time.time()
    loss = tempotron.test_batch(batchsize=10)
    end = time.time()

    Correct_roit = (1 - loss) * 100
    print(
        f"Validation completed | Classification Accuracy: {Correct_roit:.4f} % | Total Training Time: {end - start:.2f} s ")
