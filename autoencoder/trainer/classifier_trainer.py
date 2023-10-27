import logging
import os
import mlflow
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..utils.get_resources import get_config, get_path_to_working_dir

logger = logging.getLogger(__name__)


def neural_net_classifier_trainer(model, train_loader, val_loader):
    config = get_config()
    num_epochs = config["classifier_neural_net"]["num_epochs"]

    mlflow.log_params({
        "classifier_type": "neural net",
        "classifier_neural_net_num_epochs": num_epochs,
    })

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    logger.info(model)

    try:
        mlflow.log_text(str(model), "classifier_model.txt")
    except:
        logger.warning("Cant write model architecture to disk.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.4, verbose=True)

    min_val_loss = np.inf

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # log statistics
            running_loss += loss.item()
            if i % 200 == 199:  # log every 200 mini-batches
                logger.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                mlflow.log_metric("classifier_train_loss", running_loss / 200, epoch)
                running_loss = 0.0

        val_loss = validate(model, val_loader, criterion, device)
        mlflow.log_metric("classifier_val_loss", val_loss, epoch)
        logger.info(val_loss)

        if val_loss < min_val_loss:
            val_loss = min_val_loss
            save_best_model(model)

        # reduce lr on plateau
        scheduler.step(val_loss)

    logger.info('Finished Training')
    return load_best_model(model, device)


def validate(model, val_loader, criterion, device):
    valid_loss = 0.0
    model.eval()

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()

        return valid_loss / 200  # batch_size


def save_best_model(model):
    file_location = os.path.join(
        get_path_to_working_dir(),
        'temp_file_autoencoder_best_val_loss.pth'
    )
    torch.save(model.state_dict(), file_location)


def load_best_model(model, device):
    logger.info("Loading best model...")
    file_location = os.path.join(
        get_path_to_working_dir(),
        "temp_file_autoencoder_best_val_loss.pth"
    )
    model.load_state_dict(
        torch.load(file_location, map_location=torch.device(device))
    )
    logger.info("Returning best model...")
    return model
