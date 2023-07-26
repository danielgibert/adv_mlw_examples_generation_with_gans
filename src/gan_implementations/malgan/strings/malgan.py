import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from src.gan_implementations.strings_gan_skeleton import StringsGAN
from src.gan_implementations.utils import load_json
from src.gan_implementations.malgan.strings.generator_network import GeneratorNetwork
from src.gan_implementations.malgan.strings.discriminator_network import DiscriminatorNetwork
from src.gan_implementations.metrics.frechlet_inception_distance import calculate_fid
from src.feature_extractors.utils import load_all_strings
import os
import numpy as np
from abc import abstractmethod
import wandb

class SkeletonMalGAN(StringsGAN):
    @abstractmethod
    def predict_labels_with_blackbox_model(self, features: dict):
        pass

    def initialize_optimizers(self, training_parameters: dict):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=training_parameters["lr"]["generator"],
                                       betas=(training_parameters["beta1"], training_parameters["beta2"]))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=training_parameters["lr"]["discriminator"],
                                       betas=(training_parameters["beta1"], training_parameters["beta2"]))
        optimizers = {"G": optimizer_G,
                      "D": optimizer_D}
        return optimizers

    def initialize_criterions(self):
        criterions = {
            "BCELoss": torch.nn.BCELoss()
        }
        return criterions

    def run_training_epoch(self, criterions: dict, optimizers: dict, info: dict, training_parameters:dict, epoch: int):
        print("Epoch: {}".format(epoch))
        self.logger.info("\nEpoch: {}".format(epoch))

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

        i = 1

        try:
            while True:
                d_features, d_hashed_features, d_all_strings_features_filepath, d_ember_features_v1, d_ember_features_v2, d_raw_paths, d_y = next(discriminator_iterator)
                g_features, g_hashed_features, g_all_strings_features_filepath, g_ember_features_v1, g_ember_features_v2, g_raw_paths, g_y = next(generator_iterator)

                ############################
                # (1) Update D network
                ###########################
                optimizers["D"].zero_grad()  # Discrim
                d_features = d_features.to(self.device)  # Move to GPU if available
                # Generate batch of real data. It is provided by d_features, d_hashed_features, d_y
                d_output_real = self.discriminator(d_features).view(-1)

                d_hashed_features_np = d_hashed_features.cpu().detach().numpy()
                d_ember_features_v1_np = d_ember_features_v1.cpu().detach().numpy()
                d_ember_features_v2_np = d_ember_features_v2.cpu().detach().numpy()

                features_dict = {
                    "hashed": d_hashed_features_np,
                    "ember_v1": d_ember_features_v1_np,
                    "ember_v2": d_ember_features_v2_np
                }

                benign_labels = Variable(
                    self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                    requires_grad=False).to(self.device)

                errD_real = criterions["BCELoss"](d_output_real, benign_labels)
                D_x = d_output_real.mean().item()


                g_features = g_features.to(self.device)  # Move to GPU if available
                noise = torch.randn(training_parameters["batch_size"], self.generator_parameters["z_size"]).to(
                    self.device)
                fake_samples = self.generator([g_features, noise])
                d_output_fake = self.discriminator(fake_samples.detach()).view(-1)

                fake_features_np = fake_samples.cpu().detach().numpy()
                fake_features_np = fake_features_np.round().astype(int)
                g_features_np = g_features.cpu().detach().numpy().astype(int)

                g_all_strings_features = []
                for j in range(len(g_all_strings_features_filepath)):
                    g_all_strings_features.append(load_all_strings(g_all_strings_features_filepath[j]))

                fake_hashed_features = np.array([self.feature_extractor.update_hashed_features(g_features_np[j],
                                                                                               fake_features_np[j],
                                                                                               g_all_strings_features[j],
                                                                                               g_hashed_features[
                                                                                                   j]) for j in
                                                 range(fake_features_np.shape[0])])
                fake_ember_features_v1 = np.array(
                    [self.ember_feature_extractor_v1.replace_hashed_strings_features(g_ember_features_v1[j], fake_hashed_features[j]) for j in
                     range(fake_features_np.shape[0])])
                fake_ember_features_v2 = np.array(
                    [self.ember_feature_extractor_v2.replace_hashed_strings_features(g_ember_features_v2[j], fake_hashed_features[j]) for i in
                     range(fake_features_np.shape[0])])

                features_dict = {
                    "hashed": fake_hashed_features,
                    "ember_v1": fake_ember_features_v1,
                    "ember_v2": fake_ember_features_v2
                }

                fake_labels = Variable(
                    self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                    requires_grad=False).to(self.device)

                errD_fake = criterions["BCELoss"](d_output_fake, fake_labels)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                # errD_fake.backward()
                D_G_z1 = d_output_fake.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake  # Add up the average of the two losses
                errD.backward()
                # Update D
                optimizers["D"].step()  # Discriminator Parameter Update

                accuracyD_real = torch.sum(torch.round(d_output_real) == benign_labels)
                accuracyD_fake = torch.sum(torch.round(d_output_fake) == fake_labels)
                num_samples_evaded_blackbox = torch.sum(torch.round(fake_labels) == expected_labels)
                accuracyD = (accuracyD_real + accuracyD_fake) / (training_parameters["batch_size"] * 2)
                print(
                    "Epoch: {}; Step: {}; Discriminator accuracy: {}; Accuracy on real samples: {}; Accuracy on fake samples: {}".format(
                        epoch, i, accuracyD, accuracyD_real, accuracyD_fake))
                print("Number of samples that evaded the black-box detector: {} out of {}".format(
                    num_samples_evaded_blackbox, training_parameters["batch_size"]))

                self.logger.info(
                    "Epoch: {}; Step: {}; Discriminator accuracy: {}; Accuracy on real samples: {}; Accuracy on fake samples: {}".format(
                        epoch, i, accuracyD, accuracyD_real, accuracyD_fake))
                self.logger.info(
                    "Number of samples that evaded the black-box detector: {} out of {}".format(
                        num_samples_evaded_blackbox,
                        training_parameters[
                            "batch_size"]))

                ############################
                # (2) Update G network
                ###########################
                optimizers["G"].zero_grad()  # Generator Parameter Gradient Zero
                # Since we just updated D, perform another forward pass of all-fake batch through D
                d_output_fake = self.discriminator(fake_samples).view(-1)

                fake_features_np = fake_samples.cpu().detach().numpy()
                fake_features_np = fake_features_np.round().astype(int)
                g_features_np = g_features.cpu().detach().numpy().astype(int)

                g_all_strings_features = []
                for j in range(len(g_all_strings_features_filepath)):
                    g_all_strings_features.append(load_all_strings(g_all_strings_features_filepath[j]))

                fake_hashed_features = np.array([self.feature_extractor.update_hashed_features(g_features_np[j],
                                                                                               fake_features_np[j],
                                                                                               g_all_strings_features[
                                                                                                   j],
                                                                                               g_hashed_features[
                                                                                                   j]) for j in
                                                 range(fake_features_np.shape[0])])
                fake_ember_features_v1 = np.array(
                    [self.ember_feature_extractor_v1.replace_hashed_strings_features(g_ember_features_v1[j], fake_hashed_features[j]) for j in
                     range(fake_features_np.shape[0])])
                fake_ember_features_v2 = np.array(
                    [self.ember_feature_extractor_v2.replace_hashed_strings_features(g_ember_features_v2[j], fake_hashed_features[j]) for j in
                     range(fake_features_np.shape[0])])

                features_dict = {
                    "hashed": fake_hashed_features,
                    "ember_v1": fake_ember_features_v1,
                    "ember_v2": fake_ember_features_v2
                }

                fake_labels = Variable(
                    self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                    requires_grad=False).to(self.device)

                errG = criterions["BCELoss"](d_output_fake, expected_labels)

                errG.backward()
                D_G_z2 = d_output_fake.mean().item()
                # Update G
                optimizers["G"].step()  # Generator Parameter Update


                # Output training stats
                if i % training_parameters["log_interval"] == 0:
                    print(
                        '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                        % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                           errD.item(), errG.item()))
                    self.logger.info(
                        '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                        % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                           errD.item(), errG.item()))
                    info = self.log(
                        d_features,
                        fake_samples,
                        g_features,
                        g_hashed_features,
                        g_all_strings_features_filepath,
                        g_ember_features_v1,
                        g_ember_features_v2,
                        g_raw_paths,
                        training_parameters["batch_size"],
                        info
                    )
                # Save Losses for plotting later
                info["losses"]["train"]["G"].append(errG.item())
                info["losses"]["train"]["D"].append(errD.item())
                if self.is_wandb == True:
                    wandb.log(
                        {
                            "Training D loss": errD.item(),
                            "Training G loss": errG.item(),
                            "Training discriminator accuracy on benign samples": accuracyD_real / training_parameters["batch_size"],
                            "Training discriminator accuracy on fake samples": accuracyD_fake / training_parameters["batch_size"]
                        },
                    )

                if i % training_parameters["eval_interval"] == 0:
                    # Evaluate on a subset of the validation data
                    G_val_loss, D_val_loss, evasion_rates, evaluation_info = self.evaluate(
                        epoch, i, criterions, training_parameters["batch_size"])

                    info["total_strings"]["original"].append(
                        sum(evaluation_info["total_strings"]["original"]) / len(
                            evaluation_info["total_strings"]["original"]))
                    info["total_strings"]["fake"].append(sum(evaluation_info["total_strings"]["fake"]) / len(
                        evaluation_info["total_strings"]["fake"]))

                    # Add distance metrics
                    for metric in evaluation_info["distance_metrics"]:
                        info["distance_metrics"]["validation"][metric]["between_original_and_fake"].append(
                            sum(evaluation_info["distance_metrics"][metric]["between_original_and_fake"]) / len(
                                evaluation_info["distance_metrics"][metric]["between_original_and_fake"]))
                        info["distance_metrics"]["validation"][metric]["between_fake"].append(
                            sum(evaluation_info["distance_metrics"][metric]["between_fake"]) / len(
                                evaluation_info["distance_metrics"][metric]["between_fake"]))

                    if G_val_loss < info["losses"]["best"]["G"]:
                        info["losses"]["best"]["G"] = G_val_loss
                        torch.save(self.generator,
                                   os.path.join(self.output_filepath, "generator/generator_best.pt"))
                        if self.is_wandb == True:
                            wandb.run.summary["best_loss_G"] = G_val_loss
                    if D_val_loss < info["losses"]["best"]["D"]:
                        info["losses"]["best"]["D"] = D_val_loss
                        torch.save(self.discriminator, os.path.join(self.output_filepath,
                                                                    "discriminator/discriminator_best.pt"))
                        if self.is_wandb == True:
                            wandb.run.summary["best_loss_D"] = D_val_loss
                    info = self.save_models(info, evasion_rates)
                i += 1
        except StopIteration:
            pass
        return optimizers, info

    def evaluate(self, epoch: int, i: int, criterions: dict, batch_size: int):
        discriminator_dataloader = DataLoader(
            self.validation_discriminator_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        generator_dataloader = DataLoader(
            self.validation_generator_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        generator_iterator = iter(generator_dataloader)
        discriminator_iterator = iter(discriminator_dataloader)

        self.generator.eval()
        self.discriminator.eval()

        evaluation_info = self.initialize_evaluation_info_dictionary()

        G_val_losses = []
        D_val_losses = []

        j = 0

        total_samples = len(self.validation_discriminator_dataset)

        expected_labels = Variable(self.Tensor(batch_size).fill_(0.0), requires_grad=False).to(
            self.device).to(self.device)

        try:
            while True:
                d_features, d_hashed_features, d_all_strings_features_filepath, d_ember_features_v1, d_ember_features_v2, d_raw_paths, d_y = next(discriminator_iterator)
                g_features, g_hashed_features, g_all_strings_features_filepath, g_ember_features_v1, g_ember_features_v2, g_raw_paths, g_y = next(
                    generator_iterator)

                # Benign samples
                d_features = d_features.to(self.device)
                d_output_real = self.discriminator(d_features.detach()).view(-1)

                d_hashed_features_np = d_hashed_features.cpu().detach().numpy()
                d_ember_features_v1_np = d_ember_features_v1.cpu().detach().numpy()
                d_ember_features_v2_np = d_ember_features_v2.cpu().detach().numpy()

                features_dict = {
                    "hashed": d_hashed_features_np,
                    "ember_v1": d_ember_features_v1_np,
                    "ember_v2": d_ember_features_v2_np
                }

                benign_labels = Variable(
                    self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                    requires_grad=False).to(self.device)
                errD_real = criterions["BCELoss"](d_output_real, benign_labels)

                g_features = g_features.to(self.device)  # Move to GPU if available

                noise = torch.randn(batch_size, self.generator_parameters["z_size"]).to(
                    self.device)
                fake_samples = self.generator([g_features, noise])

                # Classify all fake batch with D
                d_output_fake = self.discriminator(fake_samples.detach()).view(-1)

                fake_features_np = fake_samples.cpu().detach().numpy()
                fake_features_np = fake_features_np.round().astype(int)
                g_features_np = g_features.cpu().detach().numpy().astype(int)

                g_all_strings_features = []
                for j in range(len(g_all_strings_features_filepath)):
                    g_all_strings_features.append(load_all_strings(g_all_strings_features_filepath[j]))

                fake_hashed_features = np.array([self.feature_extractor.update_hashed_features(g_features_np[j],
                                                                                               fake_features_np[j],
                                                                                               g_all_strings_features[
                                                                                                   j],
                                                                                               g_hashed_features[
                                                                                                   j]) for j in
                                                 range(fake_features_np.shape[0])])
                fake_ember_features_v1 = np.array(
                    [self.ember_feature_extractor_v1.replace_hashed_strings_features(g_ember_features_v1[j], fake_hashed_features[j]) for j in
                     range(fake_features_np.shape[0])])
                fake_ember_features_v2 = np.array(
                    [self.ember_feature_extractor_v2.replace_hashed_strings_features(g_ember_features_v2[j],fake_hashed_features[j]) for j in
                     range(fake_features_np.shape[0])])

                features_dict = {
                    "hashed": fake_hashed_features,
                    "ember_v1": fake_ember_features_v1,
                    "ember_v2": fake_ember_features_v2
                }

                fake_labels = Variable(
                    self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                    requires_grad=False).to(self.device)
                errD_fake = criterions["BCELoss"](d_output_fake, fake_labels)

                errD = errD_real + errD_fake
                errG = criterions["BCELoss"](d_output_fake, expected_labels)

                G_val_losses.append(errG.item())
                D_val_losses.append(errD.item())

                evaluation_info = self.evaluate_against_ml_models(evaluation_info, fake_samples, g_features,
                                                                  g_hashed_features, g_all_strings_features_filepath, g_ember_features_v1,
                                                                  g_ember_features_v2, g_raw_paths)

                j += 1
        except StopIteration:
            pass

        print(
            "Epoch: {}; Training Step: {}; Validation set\n Generator error: {}\n Discriminator error: {}".format(epoch,
                                                                                                                  i,
                                                                                                                  sum(
                                                                                                                      G_val_losses) / len(
                                                                                                                      G_val_losses),
                                                                                                                  sum(
                                                                                                                      D_val_losses) / len(
                                                                                                                      D_val_losses)))
        self.logger.info(
            "Epoch: {}; Training Step: {}; Validation set\n Generator error: {}\n Discriminator error: {}".format(epoch,
                                                                                                                  i,
                                                                                                                  sum(
                                                                                                                      G_val_losses) / len(
                                                                                                                      G_val_losses),
                                                                                                                  sum(
                                                                                                                      D_val_losses) / len(
                                                                                                                      D_val_losses)))
        if self.is_wandb == True:
            wandb.log(
                {
                    "Evaluation D loss": sum(D_val_losses) / len( D_val_losses),
                    "Evaluation G loss": sum(G_val_losses) / len(G_val_losses)
                }
            )
        self.print_info(evaluation_info, total_samples)
        self.log_info(evaluation_info, total_samples)
        self.log_to_wandb(evaluation_info, total_samples, "Evaluation")
        output_filepath = os.path.join(self.output_filepath,
                                       "validation_results/validation_{}epoch_{}step.txt".format(epoch, i))
        self.write_to_file(evaluation_info, output_filepath, total_samples)

        self.generator.train()
        self.discriminator.train()

        evasion_rates = {
            "hashed_model_2017": evaluation_info["detections"]["fake"][
                                     "hashed_model_2017"] / total_samples if self.features_model_2017 else 1.0,
            "hashed_model_2018": evaluation_info["detections"]["fake"][
                                     "hashed_model_2018"] / total_samples if self.features_model_2018 else 1.0,
            "ember_model_2017": evaluation_info["detections"]["fake"][
                                    "ember_model_2017"] / total_samples if self.ember_model_2017 else 1.0,
            "ember_model_2018": evaluation_info["detections"]["fake"][
                                    "hashed_model_2018"] / total_samples if self.ember_model_2018 else 1.0,
            "sorel20m_lightgbm_model": evaluation_info["detections"]["fake"][
                                           "sorel20m_lightgbm_model"] / total_samples if self.sorel20m_lightgbm_model else 1.0,
            "sorel20m_ffnn_model": evaluation_info["detections"]["fake"][
                                       "sorel20m_ffnn_model"] / total_samples if self.sorel20m_ffnn_model else 1.0,
            "malconv_model": evaluation_info["detections"]["fake"][
                                 "malconv_model"] / total_samples if self.malconv_model else 1.0,
            "nonnegative_malconv_model": evaluation_info["detections"]["fake"][
                                             "nonneg_malconv_model"] / total_samples if self.malconv_model else 1.0
        }

        return sum(G_val_losses) / len(G_val_losses), sum(D_val_losses) / len(
            D_val_losses), evasion_rates, evaluation_info


    def build_generator_network(self, generator_parameters_filepath: str):
        self.generator_parameters = load_json(generator_parameters_filepath)
        self.generator = GeneratorNetwork(self.generator_parameters)
        self.generator.to(self.device)

    def build_discriminator_network(self, discriminator_parameters_filepath: str):
        self.discriminator_parameters = load_json(discriminator_parameters_filepath)
        self.discriminator = DiscriminatorNetwork(self.discriminator_parameters)
        self.discriminator.to(self.device)





