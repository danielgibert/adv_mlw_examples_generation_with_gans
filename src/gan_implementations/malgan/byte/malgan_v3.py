import torch
import sys
from torch.utils.data import DataLoader
from torch.autograd import Variable
from src.gan_implementations.byte_histogram_gan_skeleton import ByteHistogramGAN
from src.gan_implementations.utils import load_json
from src.gan_implementations.malgan.byte.generator_network import GeneratorNetwork
from src.gan_implementations.malgan.byte.discriminator_network import DiscriminatorNetwork
from src.gan_implementations.metrics.frechlet_inception_distance import calculate_fid
from scipy.spatial.distance import pdist
from src.gan_implementations.metrics.jensen_shannon_distance import jensen_shannon_distance_np
import os
import numpy as np
from abc import abstractmethod
import wandb


#os.environ["WANDB_API_KEY"] = "d85e2b9ca580c73318bb485a0c6b24d5cd99bf41"
#os.environ["WANDB_MODE"] = "offline"
class SkeletonMalGAN(ByteHistogramGAN):

    @abstractmethod
    def predict_labels_with_blackbox_model(self, features:dict):
        pass

    def initialize_optimizers(self, training_parameters:dict):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=training_parameters["lr"]["generator"],
                                       betas=(training_parameters["beta1"], training_parameters["beta2"]))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=training_parameters["lr"]["discriminator"],
                                       betas=(training_parameters["beta1"], training_parameters["beta2"]))
        optimizers = {"G":optimizer_G,
                      "D":optimizer_D}
        return optimizers

    def initialize_criterions(self):
        criterions = {
            "BCELoss": torch.nn.BCELoss()
        }
        return criterions

    def run_training_epoch(self, criterions:dict, optimizers:dict, info:dict, training_parameters:dict, epoch:int):
        print("Epoch: {}".format(epoch))
        self.logger.info("\nEpoch: {}".format(epoch))

        self.generator.train()
        self.discriminator.train()

        # Freeze the generator and train the discriminator
        discriminator_dataloader = DataLoader(
            self.training_discriminator_dataset,
            batch_size=training_parameters["batch_size"],
            shuffle=True,
            drop_last=True
        )
        generator_dataloader = DataLoader(
            self.training_generator_dataset,
            batch_size=training_parameters["batch_size"],
            shuffle=True,
            drop_last=True
        )
        generator_iterator = iter(generator_dataloader)
        discriminator_iterator = iter(discriminator_dataloader)

        expected_labels = Variable(self.Tensor(training_parameters["batch_size"]).fill_(0.0), requires_grad=False).to(
            self.device).to(self.device)

        try:
            while True:
                try:
                    i = 1
                    while True and i < training_parameters["num_steps"]:
                        d_features, d_ember_features_v1, d_ember_features_v2, d_raw_paths, d_raw_npz_paths, d_y = next(
                            discriminator_iterator)
                        g_features, g_ember_features_v1, g_ember_features_v2, g_raw_paths, g_raw_npz_paths, g_y = next(
                            generator_iterator)
                        ############################
                        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                        ###########################
                        # Train with all real-batch
                        optimizers["D"].zero_grad()
                        d_features = d_features.to(self.device)  # Move to GPU if available

                        # Generate batch of real data.
                        d_output_real = self.discriminator(d_features).view(-1)

                        d_features_np = d_features.cpu().detach().numpy()
                        d_ember_features_v1_np = d_ember_features_v1.cpu().detach().numpy()
                        d_ember_features_v2_np = d_ember_features_v2.cpu().detach().numpy()

                        features_dict = {
                            "byte_histogram": d_features_np,
                            "ember_v1": d_ember_features_v1_np,
                            "ember_v2": d_ember_features_v2_np
                        }
                        blackbox_benign_labels = Variable(
                            self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                            requires_grad=False).to(self.device)

                        # Calculate loss on all-real batch
                        errD_real = criterions["BCELoss"](d_output_real, blackbox_benign_labels)
                        # Calculate gradients for D in backward pass
                        # errD_real.backward()
                        D_x = d_output_real.mean().item()

                        ## Train with all-fake batch
                        # Generate a batch of noise to generate the fake samples
                        g_features = g_features.to(self.device)  # Move to GPU if available

                        noise = torch.randn(training_parameters["batch_size"], self.generator_parameters["z_size"]).to(
                            self.device)
                        fake_samples = self.generator([g_features, noise])

                        # Classify all fake batch with D
                        d_output_fake = self.discriminator(fake_samples.detach()).view(-1)
                        # print("Output discriminator: ", d_output_fake)
                        # print("Target: ", fake_labels)

                        fake_features_np = fake_samples.cpu().detach().numpy()
                        # Replace EMBER features
                        fake_ember_features_v1 = np.array([
                            self.ember_feature_extractor_v1.replace_byte_histogram_features(np.squeeze(g_ember_features_v1[i]),
                                                                                            fake_features_np[i])
                            for i in range(fake_features_np.shape[0])
                        ])

                        fake_ember_features_v2 = np.array([
                            self.ember_feature_extractor_v2.replace_byte_histogram_features(np.squeeze(g_ember_features_v2[i]),
                                                                                            fake_features_np[i])
                            for i in range(fake_features_np.shape[0])
                        ])

                        features_dict = {
                            "byte_histogram": fake_features_np,
                            "ember_v1": fake_ember_features_v1,
                            "ember_v2": fake_ember_features_v2
                        }
                        blackbox_fake_labels = Variable(
                            self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                            requires_grad=False).to(self.device)

                        # Calculate D's loss on the all-fake batch
                        errD_fake = criterions["BCELoss"](d_output_fake, blackbox_fake_labels)
                        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                        # errD_fake.backward()
                        D_G_z1 = d_output_fake.mean().item()
                        # Compute error of D as sum over the fake and the real batches
                        errD = errD_real + errD_fake  # Add up the average of the two losses
                        errD.backward()
                        # Update D
                        optimizers["D"].step()  # Discriminator Parameter Update
                        info["losses"]["train"]["D"].append(errD.item())

                        accuracyD_real = torch.sum(torch.round(d_output_real) == blackbox_benign_labels)
                        accuracyD_fake = torch.sum(torch.round(d_output_fake) == blackbox_fake_labels)
                        num_samples_evaded_blackbox = torch.sum(torch.round(blackbox_fake_labels) == expected_labels)
                        accuracyD = (accuracyD_real + accuracyD_fake) / (training_parameters["batch_size"] * 2)
                        print(
                            "Discriminator; Epoch: {}; Step: {}; Discriminator accuracy: {}; Accuracy on real samples: {}; Accuracy on fake samples: {}".format(
                                epoch, i, accuracyD, accuracyD_real, accuracyD_fake))
                        print(
                            "Discriminator; Number of samples that evaded the black-box detector: {}/{}".format(
                                num_samples_evaded_blackbox,
                                training_parameters[
                                    "batch_size"]))

                        self.logger.info(
                            "Epoch: {}; Step: {}; Discriminator accuracy: {}; Accuracy on real samples: {}; Accuracy on fake samples: {}".format(
                                epoch, i, accuracyD, accuracyD_real, accuracyD_fake))
                        self.logger.info(
                            "Number of samples that evaded the black-box detector: {}/{}".format(
                                num_samples_evaded_blackbox,
                                training_parameters[
                                    "batch_size"]))
                        if self.is_wandb == True:
                            wandb.log(
                                {
                                    "Training D loss": errD.item(),
                                    "Training discriminator accuracy on benign samples": accuracyD_real / training_parameters[
                                        "batch_size"],
                                    "Training discriminator accuracy on fake samples": accuracyD_fake / training_parameters[
                                        "batch_size"]
                                },
                            )

                        i += 1
                except StopIteration as e:
                    raise e

                generator_dataloader = DataLoader(
                    self.training_generator_dataset,
                    batch_size=training_parameters["batch_size"],
                    shuffle=True,
                    drop_last=True
                )
                generator_iterator = iter(generator_dataloader)

                try:
                    i = 1
                    while True and i < training_parameters["num_steps"]:
                        g_features, g_ember_features_v1, g_ember_features_v2, g_raw_paths, g_raw_npz_paths, g_y = next(
                            generator_iterator)
                        g_features = g_features.to(self.device)  # Move to GPU if available

                        noise = torch.randn(training_parameters["batch_size"], self.generator_parameters["z_size"]).to(
                            self.device)
                        fake_samples = self.generator([g_features, noise])

                        # print("Output discriminator: ", d_output_fake)
                        # print("Target: ", fake_labels)

                        fake_features_np = fake_samples.cpu().detach().numpy()
                        # Replace EMBER features
                        fake_ember_features_v1 = np.array([
                            self.ember_feature_extractor_v1.replace_byte_histogram_features(np.squeeze(g_ember_features_v1[i]),
                                                                                            fake_features_np[i])
                            for i in range(fake_features_np.shape[0])
                        ])

                        fake_ember_features_v2 = np.array([
                            self.ember_feature_extractor_v2.replace_byte_histogram_features(np.squeeze(g_ember_features_v2[i]),
                                                                                            fake_features_np[i])
                            for i in range(fake_features_np.shape[0])
                        ])

                        features_dict = {
                            "byte_histogram": fake_features_np,
                            "ember_v1": fake_ember_features_v1,
                            "ember_v2": fake_ember_features_v2
                        }

                        optimizers["G"].zero_grad()  # Generator Parameter Gradient Zero
                        # Since we just updated D, perform another forward pass of all-fake batch through D
                        d_output = self.discriminator(fake_samples).view(-1)

                        blackbox_fake_labels = Variable(
                            self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                            requires_grad=False).to(self.device)

                        # Calculate G's loss based on this output
                        errG = criterions["BCELoss"](d_output, expected_labels)
                        # Calculate gradients for G
                        errG.backward()
                        D_G_z2 = d_output.mean().item()
                        # Update G
                        optimizers["G"].step()
                        # Save Losses for plotting later
                        info["losses"]["train"]["G"].append(errG.item())
                        accuracyD_fake = torch.sum(torch.round(d_output) == blackbox_fake_labels)
                        num_samples_evaded_blackbox = torch.sum(blackbox_fake_labels == expected_labels)

                        print("Generator; Epoch: {}; Step: {}; G loss: {}; Training D accuracy on fake samples: {}/{}".format(
                            epoch,
                            i,
                            errG.item(),
                            accuracyD_fake,
                            training_parameters["batch_size"]
                        ))
                        print(
                            "Generator; Number of samples that evaded the black-box detector: {}/{}".format(
                                num_samples_evaded_blackbox,
                                training_parameters[
                                    "batch_size"]))

                        self.logger.info(
                            "Epoch: {}; Step: {}; Discriminator accuracy on fake samples: {}".format(
                                epoch, i, accuracyD_fake))
                        self.logger.info(
                            "Number of samples that evaded the black-box detector: {}/{}".format(
                                num_samples_evaded_blackbox,
                                training_parameters[
                                    "batch_size"]))

                        if self.is_wandb == True:
                            wandb.log(
                                {
                                    "Training G loss": errG.item(),
                                    "Training discriminator accuracy on fake samples": accuracyD_fake / training_parameters[
                                        "batch_size"]
                                },
                            )
                        i += 1
                except StopIteration as e:
                    raise e
        except StopIteration:
            pass
        return optimizers, info

    def evaluate(self, epoch: int, i: int, criterions: dict, training_parameters: dict):
        pass

    def build_generator_network(self, generator_parameters_filepath:str):
        self.generator_parameters = load_json(generator_parameters_filepath)
        self.generator = GeneratorNetwork(self.generator_parameters)
        self.generator.to(self.device)

    def build_discriminator_network(self, discriminator_parameters_filepath:str):
        self.discriminator_parameters = load_json(discriminator_parameters_filepath)
        self.discriminator = DiscriminatorNetwork(self.discriminator_parameters)
        self.discriminator.to(self.device)