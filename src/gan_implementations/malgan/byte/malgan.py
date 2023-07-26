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

        i = 1
        try:
            while True:
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
                benign_labels = Variable(
                    self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                    requires_grad=False).to(self.device)

                # Calculate loss on all-real batch
                errD_real = criterions["BCELoss"](d_output_real, benign_labels)
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
                fake_labels = Variable(
                    self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                    requires_grad=False).to(self.device)


                # Calculate D's loss on the all-fake batch
                errD_fake = criterions["BCELoss"](d_output_fake, fake_labels)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                # errD_fake.backward()
                D_G_z1 = d_output_fake.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake  # Add up the average of the two losses
                errD.backward()
                # Update D
                optimizers["D"].step()  # Discriminator Parameter Update
                info["losses"]["train"]["D"].append(errD.item())

                accuracyD_real = torch.sum(torch.round(d_output_real) == benign_labels)
                accuracyD_fake = torch.sum(torch.round(d_output_fake) == fake_labels)
                num_samples_evaded_blackbox = torch.sum(torch.round(fake_labels) == expected_labels)
                accuracyD = (accuracyD_real + accuracyD_fake) / (training_parameters["batch_size"] * 2)
                print(
                    "Epoch: {}; Step: {}; Discriminator accuracy: {}; Accuracy on real samples: {}; Accuracy on fake samples: {}".format(
                        epoch, i, accuracyD, accuracyD_real, accuracyD_fake))
                print(
                    "Number of samples that evaded the black-box detector: {} out of {}".format(num_samples_evaded_blackbox,
                                                                                                training_parameters[
                                                                                                    "batch_size"]))

                self.logger.info(
                    "Epoch: {}; Step: {}; Discriminator accuracy: {}; Accuracy on real samples: {}; Accuracy on fake samples: {}".format(
                        epoch, i, accuracyD, accuracyD_real, accuracyD_fake))
                self.logger.info(
                    "Number of samples that evaded the black-box detector: {} out of {}".format(num_samples_evaded_blackbox,
                                                                                                training_parameters[
                                                                                                    "batch_size"]))

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                if i % training_parameters["train_every_X_steps"] == 0:
                    optimizers["G"].zero_grad()  # Generator Parameter Gradient Zero
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    d_output = self.discriminator(fake_samples).view(-1)

                    fake_labels = Variable(
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
                    try:
                        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f = %.4f'
                              % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, D_G_z1 / D_G_z2))
                    except ZeroDivisionError:
                        print(
                            '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f = ZeroDivisionError'
                            % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                # Output training stats
                if i % training_parameters["log_interval"] == 0:
                    try:
                        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f = %.4f'
                              % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, D_G_z1 / D_G_z2))
                        self.logger.info(
                            '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f = %.4f'
                            % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, D_G_z1 / D_G_z2))
                    except ZeroDivisionError:
                        print(
                            '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f = ZeroDivisionError'
                            % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                        self.logger.info(
                            '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f = ZeroDivisionError'
                            % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                        continue
                    info = self.log(
                        d_features,
                        fake_samples,
                        g_features,
                        g_ember_features_v1,
                        g_ember_features_v2,
                        g_raw_paths,
                        g_raw_npz_paths,
                        training_parameters["batch_size"],
                        info
                    )

                if self.is_wandb == True:
                    wandb.log(
                        {
                            "Training D loss": errD.item(),
                            "Training G loss": errG.item(),
                            "Training discriminator accuracy on benign samples": accuracyD_real / training_parameters[
                                "batch_size"],
                            "Training discriminator accuracy on fake samples": accuracyD_fake / training_parameters[
                                "batch_size"]
                        },
                    )

                if i % training_parameters["eval_interval"] == 0:
                    # Evaluate on a subset of the validation data
                    G_val_loss, D_val_loss, evasion_rates, evaluation_info = self.evaluate(
                        epoch,
                        i,
                        criterions,
                        training_parameters
                    )
                    info["fid"]["scores"].append(evaluation_info["fid"])
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

                    if evaluation_info["fid"] < info["fid"]["best"]:
                        info["fid"]["best"] = evaluation_info["fid"]
                        torch.save(self.generator,
                                   os.path.join(self.output_filepath, "generator/generator_best_fid.pt"))
                        torch.save(self.discriminator, os.path.join(self.output_filepath,
                                                                    "discriminator/discriminator_best_fid.pt"))
                        if self.is_wandb == True:
                            wandb.run.summary["best_fid"] = evaluation_info["fid"]

                    info = self.save_models(info, evasion_rates)
                i += 1
        except StopIteration:
            pass
        return optimizers, info

    def evaluate(self, epoch:int, i:int, criterions:dict, training_parameters:dict):
        discriminator_dataloader = DataLoader(
            self.validation_discriminator_dataset,
            batch_size=training_parameters["validation_batch_size"],
            shuffle=True,
            drop_last=True
        )
        generator_dataloader = DataLoader(
            self.validation_generator_dataset,
            batch_size=training_parameters["validation_batch_size"],
            shuffle=True,
            drop_last=True
        )

        discriminator_iterator = iter(discriminator_dataloader)
        generator_iterator = iter(generator_dataloader)

        self.generator.eval()
        self.discriminator.eval()

        evaluation_info = self.initialize_evaluation_info_dictionary()

        G_val_losses = []
        D_val_losses = []

        j = 0

        total_samples = training_parameters["evaluation_samples"] * training_parameters["validation_batch_size"]
        expected_labels = Variable(self.Tensor(training_parameters["validation_batch_size"]).fill_(0.0), requires_grad=False).to(
            self.device).to(self.device)

        fake_features_list = []
        original_features_list = []

        for k in range(training_parameters["evaluation_samples"]):
            d_features, d_ember_features_v1, d_ember_features_v2, d_raw_paths, d_raw_npz_paths, d_y = next(discriminator_iterator)
            # Benign samples
            d_features_dev = d_features.to(self.device)
            d_output_real = self.discriminator(d_features_dev.detach()).view(-1)

            d_features_np = d_features.cpu().detach().numpy()
            d_ember_features_v1_np = d_ember_features_v1.cpu().detach().numpy()
            d_ember_features_v2_np = d_ember_features_v2.cpu().detach().numpy()

            features_dict = {
                "byte_histogram": d_features_np,
                "ember_v1": d_ember_features_v1_np,
                "ember_v2": d_ember_features_v2_np
            }
            benign_labels = Variable(
                self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                requires_grad=False).to(self.device)

            errD_real = criterions["BCELoss"](d_output_real, benign_labels)

            # Malicious samples
            # Sample a random batch of malware samples
            g_features, g_ember_features_v1, g_ember_features_v2, g_raw_paths, g_raw_paths_npz, g_y = next(generator_iterator)
            g_features = g_features.to(self.device)  # Move to GPU if available

            noise = torch.randn(training_parameters["validation_batch_size"], self.generator_parameters["z_size"]).to(
                self.device)
            fake_samples = self.generator([g_features, noise])
            # Classify all fake batch with D
            d_output_fake = self.discriminator(fake_samples.detach()).view(-1)

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
            fake_labels = Variable(
                self.Tensor(self.predict_labels_with_blackbox_model(features_dict).astype(float)),
                requires_grad=False).to(self.device)


            # Calculate D's loss on the all-fake batch
            errD_fake = criterions["BCELoss"](d_output_fake, fake_labels)

            errD = errD_real + errD_fake
            errG = criterions["BCELoss"](d_output_fake, expected_labels)

            G_val_losses.append(errG.item())
            D_val_losses.append(errD.item())

            # Store intermediate features from both benign and fake samples
            act2 = self.inception_model.retrieve_features(d_features)
            act2 = act2.cpu().detach().numpy()
            evaluation_info["intermediate_features"]["benign"].append(act2)

            original_features_list.append(g_features.cpu().detach().numpy())
            fake_features_list.append(fake_samples.cpu().detach().numpy())

            evaluation_info = self.evaluate_against_ml_models(
                evaluation_info,
                fake_samples,
                g_features,
                g_ember_features_v1,
                g_ember_features_v2,
                g_raw_paths,
                g_raw_paths_npz,
                replace=True
            )
            j += 1

        original_features_arr = np.array(original_features_list)
        original_features_arr = np.squeeze(original_features_arr)
        fake_features_arr = np.array(fake_features_list)
        fake_features_arr = np.squeeze(fake_features_arr)

        evaluation_info["distance_metrics"]["jensenshannon"]["between_original_and_fake"].append(
            jensen_shannon_distance_np(original_features_arr, fake_features_arr))
        evaluation_info["distance_metrics"]["jensenshannon"]["between_fake"].append(
            pdist(fake_features_arr, metric="jensenshannon"))

        fid_score = calculate_fid(np.concatenate(evaluation_info["intermediate_features"]["fake"]),
                                  np.concatenate(evaluation_info["intermediate_features"]["benign"]))
        evaluation_info["fid"] = fid_score
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
                    "Evaluation D loss": sum(D_val_losses) / len(D_val_losses),
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
            "byte_histogram_model_2017": evaluation_info["detections"]["fake"][
                                     "byte_histogram_model_2017"] / total_samples if self.features_model_2017 else 1.0,
            "byte_histogram_model_2018": evaluation_info["detections"]["fake"][
                                     "byte_histogram_model_2018"] / total_samples if self.features_model_2018 else 1.0,
            "ember_model_2017": evaluation_info["detections"]["fake"][
                                    "ember_model_2017"] / total_samples if self.ember_model_2017 else 1.0,
            "ember_model_2018": evaluation_info["detections"]["fake"][
                                    "ember_model_2018"] / total_samples if self.ember_model_2018 else 1.0,
            "sorel20m_lightgbm_model": evaluation_info["detections"]["fake"][
                                           "sorel20m_lightgbm_model"] / total_samples if self.sorel20m_lightgbm_model else 1.0,
            "sorel20m_ffnn_model": evaluation_info["detections"]["fake"][
                                       "sorel20m_ffnn_model"] / total_samples if self.sorel20m_ffnn_model else 1.0,
            "malconv_model": evaluation_info["detections"]["fake"][
                                 "malconv_model"] / total_samples if self.malconv_model else 1.0,
            "nonnegative_malconv_model": evaluation_info["detections"]["fake"][
                                             "nonneg_malconv_model"] / total_samples if self.nonneg_malconv_model else 1.0,
            "malconvgct_model": evaluation_info["detections"]["fake"][
                                 "malconvgct_model"] / total_samples if self.malconvgct_model else 1.0,
        }

        return sum(G_val_losses) / len(G_val_losses), sum(D_val_losses) / len(D_val_losses), evasion_rates, evaluation_info

    def build_generator_network(self, generator_parameters_filepath:str):
        self.generator_parameters = load_json(generator_parameters_filepath)
        self.generator = GeneratorNetwork(self.generator_parameters)
        self.generator.to(self.device)

    def build_discriminator_network(self, discriminator_parameters_filepath:str):
        self.discriminator_parameters = load_json(discriminator_parameters_filepath)
        self.discriminator = DiscriminatorNetwork(self.discriminator_parameters)
        self.discriminator.to(self.device)