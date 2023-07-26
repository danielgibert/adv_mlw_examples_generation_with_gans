import sys
import argparse
import os
import torch
sys.path.append("../../../")
from src.ml_models.torch_models.inception_detector.iat_net import IATNetwork
from src.ml_models.torch_models.inception_detector.byte_histogram_net import ByteHistogramNetwork
from src.gan_implementations.utils import  load_json
from src.ml_models.torch_models.inception_detector.numpy_feature_dataset import NumpyFeatureDataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inception Network Training Script')
    parser.add_argument("--training_filepath",
                        type=str,
                        help="Filepath to the training features",
                        default="/mnt/hdd2/inception/imports/2017/training/")
    parser.add_argument("--testing_filepath",
                        type=str,
                        help="Filepath to the testing features",
                        default="/mnt/hdd2/inception/imports/2017/testing/")
    parser.add_argument("--network_parameters",
                        type=str,
                        help="Network parameters")
    parser.add_argument("--training_parameters",
                        type=str,
                        help="Training parameters")
    parser.add_argument("--output_filepath",
                        type=str,
                        help="Output filepath",
                        default="models/")

    parser.add_argument('--iat', dest='iat', action='store_true')
    parser.add_argument('--no-iat', dest='iat', action='store_false')
    parser.set_defaults(iat=False)

    parser.add_argument('--byte', dest='byte', action='store_true')
    parser.add_argument('--no-byte', dest='byte', action='store_false')
    parser.set_defaults(byte=False)

    parser.add_argument("--cuda",
                        type=int,
                        help="The GPU to use, i.e cuda:0, cuda:1",
                        default=None)
    args = parser.parse_args()

    # Use GPU if available

    if args.cuda is None:
        dev = "cpu"
        cuda = False
    else:
        dev = "cuda:{}".format(args.cuda)
        cuda = True

    print("Device: {}".format(dev))
    device = torch.device(dev)

    # Create output filepath
    try:
        os.makedirs(args.output_filepath)
        os.makedirs(os.path.join(args.output_filepath, "best"))
        os.makedirs(os.path.join(args.output_filepath, "checkpoints"))
    except FileExistsError as fee:
        print(fee)

    # Initialize Inception model
    network_parameters = load_json(args.network_parameters)
    training_parameters = load_json(args.training_parameters)

    if args.iat:
        features_size = 1280
        model = IATNetwork(network_parameters)
    elif args.byte:
        features_size = 256
        model = ByteHistogramNetwork(network_parameters)
    else:
        raise Exception("Please choose one of the following features:\n1.-iat\n2.-byte")
    model.to(device)

    # Create Datasets and DataLoaders
    training_dataset = NumpyFeatureDataset(
        args.training_filepath
    )
    # Create generator dataset
    testing_dataset = NumpyFeatureDataset(
        args.testing_filepath
    )


    criterion = torch.nn.BCELoss()

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=training_parameters["lr"],
                                   betas=(training_parameters["beta1"], training_parameters["beta2"]))
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    training_losses = []
    best_loss = sys.maxsize

    for epoch in range(training_parameters["num_epochs"]):

        model.train()

        print("Epoch: {}".format(epoch))
        # Freeze the generator and train the discriminator
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=training_parameters["batch_size"],
            shuffle=True,
            drop_last=True
        )
        i = 1
        for (X, Y) in training_dataloader:
            optimizer.zero_grad()
            X = X.to(device)
            Y = Y.to(device)

            y_pred = model(X).view(-1)
            # Calculate loss on all-real batch
            err = criterion(y_pred, Y)
            # Calculate gradients in a  backward pass
            err.backward()
            # Update D
            optimizer.step()  # Discriminator Parameter Update

            # Output training stats
            if i % training_parameters["log_interval"] == 0:
                accuracy = torch.sum(torch.round(y_pred) == Y)
                print("Step: {}; Accuracy: {}/{}; Loss: {}".format(i, accuracy, training_parameters["batch_size"], err))
                print(Y)
                print(y_pred)
            i += 1

        if epoch % training_parameters["eval_interval"] == 0:
            testing_dataloader = DataLoader(
                testing_dataset,
                batch_size=training_parameters["batch_size"],
                shuffle=True,
                drop_last=True
            )
            model.eval()
            j = 0

            validation_loss = []
            validation_accuracy = 0
            for (X, Y) in testing_dataloader:
                # Benign samples
                X = X.to(device)
                Y = Y.to(device)

                y_pred = model(X).view(-1)
                err = criterion(y_pred, Y)

                accuracy = torch.sum(torch.round(y_pred) == Y)

                validation_accuracy += accuracy
                validation_loss.append(err)
                j += 1
            print("Validation accuracy: {}/{}={}".format(validation_accuracy, len(testing_dataset), validation_accuracy / len(testing_dataset)))
            print("Validation loss: {}".format(sum(validation_loss) / j))

            if sum(validation_loss) / j <= best_loss:
                best_loss = sum(validation_loss) / j
                torch.save(model.state_dict(),
                           os.path.join(args.output_filepath, "best/best_model.pt"))
            model.train()

            if epoch % training_parameters["save_interval"]  == 0:
                torch.save(model.state_dict(),
                           os.path.join(args.output_filepath, "checkpoints/model_checkoint_{}.pt".format(epoch)))

    # Get the best checkpoint and evaluate on validation data
    model.load_state_dict(torch.load(os.path.join(args.output_filepath, "best/best_model.pt")))
    model.eval()

    testing_dataloader = DataLoader(
        testing_dataset,
        batch_size=training_parameters["batch_size"],
        shuffle=True,
        drop_last=True
    )

    j = 0
    validation_loss = []
    validation_accuracy = 0
    for (X, Y) in testing_dataloader:
        # Benign samples
        X = X.to(device)
        Y = Y.to(device)

        y_pred = model(X).view(-1)
        err = criterion(y_pred, Y)

        accuracy = torch.sum(torch.round(y_pred) == Y)

        validation_accuracy += accuracy
        validation_loss.append(err)
        j += 1
    print("Validation accuracy: {}/{}={}".format(validation_accuracy, len(testing_dataset),
                                                 validation_accuracy / len(testing_dataset)))
    print("Validation loss: {}".format(sum(validation_loss) / j))

