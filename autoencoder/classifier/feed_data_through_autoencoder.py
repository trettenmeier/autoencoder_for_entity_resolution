import torch
import numpy as np
from tqdm import tqdm


def feed_data_through_trained_autoencoder(model, train_loader, test_loader):
    output_training_data = get_model_output(model, train_loader)
    output_test_data = get_model_output(model, test_loader)

    return output_training_data, output_test_data


def get_model_output(model, train_loader):
    output_a = []
    output_b = []

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)

    for input_a, input_b in tqdm(train_loader):
        model.eval()
        with torch.no_grad():
            input_a = input_a.to(device)
            input_b = input_b.to(device)

            input_a = model.reshape_input(input_a)
            input_b = model.reshape_input(input_b)

            a = model.encoder(input_a)
            b = model.encoder(input_b)

            # iterate over first dimension (the batch dimension):
            for i in range(len(a)):
                output_a.append(a[i])
                output_b.append(b[i])

    output_a_list = []
    for tens in output_a:
        output_a_list.append(tens.cpu().detach().numpy())

    output_b_list = []
    for tens in output_b:
        output_b_list.append(tens.cpu().detach().numpy())

    output_concat_list = []

    for i in range(len(output_b_list)):
        temp_1 = output_a_list[i].reshape(1, -1)
        temp_2 = output_b_list[i].reshape(1, -1)
        output_concat_list.append(np.concatenate((temp_1, temp_2), axis=1))

    output_concat_list = np.array(output_concat_list)
    output_concat_list = np.squeeze(output_concat_list)

    return output_concat_list
