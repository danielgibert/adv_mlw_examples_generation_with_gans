import torch
import sys
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
sys.path.append("../../../../")
from src.gan_implementations.byte_histogram_gan_skeleton import ByteHistogramGAN
from src.gan_implementations.utils import load_json
from src.gan_implementations.wassertein_gan_gp.byte.generator_network import GeneratorNetwork
from src.gan_implementations.wassertein_gan_gp.byte.discriminator_network import DiscriminatorNetwork
from src.gan_implementations.metrics.frechlet_inception_distance import calculate_fid
from src.gan_implementations.byte_histogram_dataset import ByteHistogramDataset
from scipy.spatial.distance import pdist
from src.gan_implementations.metrics.jensen_shannon_distance import jensen_shannon_distance_np
import os
import numpy as np
import wandb
import torch.autograd as autograd
#os.environ["WANDB_API_KEY"] = "d85e2b9ca580c73318bb485a0c6b24d5cd99bf41"
#os.environ["WANDB_MODE"] = "offline"



class WasserteinGANGradientPenalty(ByteHistogramGAN):
    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = self.Tensor(np.random.random((real_samples.size(0), 1,)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(self.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def initialize_optimizers(self, training_parameters:dict):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=training_parameters["lr"]["generator"],
                                       betas=(training_parameters["beta1"], training_parameters["beta2"]))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=training_parameters["lr"]["discriminator"],
                                       betas=(training_parameters["beta1"], training_parameters["beta2"]))
        optimizers = {"G":optimizer_G,
                      "D":optimizer_D}
        return optimizers

    def initialize_criterions(self):
        return None

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

        valid_labels = Variable(self.Tensor(training_parameters["batch_size"]).fill_(-1.0), requires_grad=False).to(
            self.device)
        fake_labels = Variable(self.Tensor(training_parameters["batch_size"]).fill_(1.0), requires_grad=False).to(
            self.device)

        i = 1

        try:
            while True:
                d_features, d_ember_features_v1, d_ember_features_v2, d_raw_paths, d_raw_npz_paths, d_y = next(discriminator_iterator)
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
                D_x = d_output_real.mean().item()

                ## Train with all-fake batch
                # Generate a batch of noise to generate the fake samples
                g_features = g_features.to(self.device)  # Move to GPU if available

                noise = torch.randn(training_parameters["batch_size"], self.generator_parameters["z_size"]).to(
                    self.device)
                fake_samples = self.generator([g_features, noise])
                # Classify all fake batch with D
                d_output_fake = self.discriminator(fake_samples.detach()).view(-1)
                D_G_z1 = d_output_fake.mean().item()
                # Calculate D loss
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(self.discriminator, d_features.data, fake_samples.data)
                # Adversarial loss
                errD = -torch.mean(d_output_real) + torch.mean(d_output_fake) + training_parameters["lambda_gp"] * gradient_penalty

                errD.backward()
                # Update D
                optimizers["D"].step()  # Discriminator Parameter Update
                info["losses"]["train"]["D"].append(errD.item())

                if self.is_wandb == True:
                    wandb.log(
                        {
                            "Training D loss": errD.item()
                        },
                    )
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                if i % training_parameters["train_every_X_steps"] == 0:
                    optimizers["G"].zero_grad()  # Generator Parameter Gradient Zero
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    fake_samples = self.generator([g_features, noise])
                    d_output_fake = self.discriminator(fake_samples).view(-1)

                    # Calculate G's loss based on this output
                    errG = -torch.mean(d_output_fake)
                    D_G_z2 = d_output_fake.mean().item()
                    errG.backward()
                    # Update G
                    optimizers["G"].step()
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD_x:%.6f\tD_G_z1:%.6f/D_G_z2:%.6f'
                          % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                    info["losses"]["train"]["G"].append(errG.item())
                    if self.is_wandb == True:
                        wandb.log(
                            {
                                "Training G loss": errG.item()
                            },
                        )
                # Output training stats
                if i % training_parameters["log_interval"] == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                          % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                             errD.item(), errG.item()))
                    self.logger.info(
                        '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                        % (epoch, training_parameters["num_epochs"], i, len(discriminator_dataloader),
                           errD.item(), errG.item()))

                    self.logger.info("Benign features: {}".format(d_features))
                    self.logger.info("Malicious features: {}".format(g_features))
                    self.logger.info("Fake features: {}".format(fake_samples))
                    self.logger.info("Benign labels: {}".format(d_output_real))
                    self.logger.info("Fake labels: {}".format(d_output_fake))

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

                if i % training_parameters["eval_interval"] == 0:
                    # Evaluate on a subset of the validation data
                    G_val_loss, D_val_loss, evasion_rates, evaluation_info = self.evaluate(
                        epoch,
                        i,
                        criterions,
                        valid_labels,
                        fake_labels,
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

    def evaluate(self, epoch:int, i:int, criterions:dict, valid_labels:Variable, fake_labels:Variable, training_parameters:dict):
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
        # Create again the valid labels and fake labels with the correct batch size

        fake_features_list = []
        original_features_list = []

        for k in range(training_parameters["evaluation_samples"]):
            d_features, d_ember_features_v1, d_ember_features_v2, d_raw_paths, d_raw_npz_paths, d_y = next(discriminator_iterator)
            # Benign samples
            d_features_dev = d_features.to(self.device)
            d_output_real = self.discriminator(d_features_dev.detach()).view(-1)

            # Malicious samples
            # Sample a random batch of malware samples
            g_features, g_ember_features_v1, g_ember_features_v2, g_raw_paths, g_raw_paths_npz, g_y = next(generator_iterator)
            g_features = g_features.to(self.device)  # Move to GPU if available

            noise = torch.randn(training_parameters["validation_batch_size"], self.generator_parameters["z_size"]).to(
                self.device)
            fake_samples = self.generator([g_features, noise])
            # Classify all fake batch with D
            d_output_fake = self.discriminator(fake_samples.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD = -torch.mean(d_output_real) + torch.mean(d_output_fake)
            errG = -torch.mean(d_output_fake)

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
                                             "nonneg_malconv_model"] / total_samples if self.malconv_model else 1.0
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

    def validate(self, epoch: int, batch_size: int = 32):
        benign_dataloader = self.initialize_dataloader(self.validation_discriminator_dataset, batch_size, shuffle=False,
                                                       drop_last=True)
        malicious_dataloader = self.initialize_dataloader(self.validation_generator_dataset, batch_size, shuffle=False,
                                                          drop_last=True)

        output_filepath = os.path.join(self.output_filepath,
                                       "validation_results/validation_{}epoch.txt".format(epoch))
        evaluation_info = self.check_evasion_rate_with_losses(
            self.validation_discriminator_dataset,
            benign_dataloader,
            self.validation_generator_dataset,
            malicious_dataloader,
            output_filepath)
        self.log_to_wandb(evaluation_info, len(self.validation_generator_dataset), "Validation")
        return evaluation_info["losses"]["G"], evaluation_info["losses"]["D"]

    def check_evasion_rate_with_losses(self, benign_dataset: torch.utils.data.Dataset, benign_dataloader: DataLoader, malicious_dataset: torch.utils.data.Dataset, malicious_dataloader: DataLoader, output_filepath:str, replace:bool = True, approach:str = "exact"):
        self.generator.eval()
        self.discriminator.eval()

        evaluation_info = self.initialize_evaluation_info_dictionary()

        benign_features_list = []
        fake_features_list = []
        original_features_list = []

        d_output_real_list = []
        d_output_fake_list = []
        benign_iterator = iter(benign_dataloader)
        malicious_iterator = iter(malicious_dataloader)

        try:
            j = 0
            while True:
                d_features, d_ember_features_v1, d_ember_features_v2, d_raw_paths, d_raw_npz_paths, d_y = next(benign_iterator)
                g_features, g_ember_features_v1, g_ember_features_v2, g_raw_paths, g_raw_npz_paths, g_y = next(malicious_iterator)

                d_features = d_features.to(self.device)
                g_features = g_features.to(self.device)
                d_output_real = self.discriminator(d_features).view(-1)

                noise = torch.randn(g_features.shape[0], self.generator_parameters["z_size"]).to(
                    self.device)
                fake_samples = self.generator([g_features, noise])
                d_output_fake = self.discriminator(fake_samples).view(-1)

                #errD = -torch.mean(d_output_real) + torch.mean(d_output_fake)
                #errG = -torch.mean(d_output_fake)

                d_output_real_list.append(d_output_real)
                d_output_fake_list.append(d_output_fake)

                act1 = self.inception_model.retrieve_features(fake_samples.cpu().detach())
                act1 = act1.cpu().detach().numpy()
                evaluation_info["intermediate_features"]["fake"].append(act1)

                fake_features = fake_samples.cpu().detach().numpy()

                benign_features_list.append(d_features.cpu().detach().numpy())
                original_features_list.append(g_features.cpu().detach().numpy())
                fake_features_list.append(fake_features)

                evaluation_info = self.evaluate_against_ml_models(
                    evaluation_info,
                    fake_samples,
                    g_features,
                    g_ember_features_v1,
                    g_ember_features_v2,
                    g_raw_paths,
                    g_raw_npz_paths,
                    replace=replace,
                    approach=approach
                )
                j += 1
        except StopIteration:
            pass

        original_features_arr = np.array(original_features_list)
        original_features_arr = np.squeeze(original_features_arr, axis=1)
        fake_features_arr = np.array(fake_features_list)
        fake_features_arr = np.squeeze(fake_features_arr, axis=1)
        benign_features_arr = np.array(benign_features_list)
        benign_features_arr = np.squeeze(benign_features_arr, axis=1)

        evaluation_info["distance_metrics"]["jensenshannon"]["between_original_and_fake"].append(
            jensen_shannon_distance_np(original_features_arr, fake_features_arr))
        evaluation_info["distance_metrics"]["jensenshannon"]["between_fake_and_benign"].append(
            jensen_shannon_distance_np(fake_features_arr, benign_features_arr))
        evaluation_info["distance_metrics"]["jensenshannon"]["between_fake"].append(
            pdist(fake_features_arr, metric="jensenshannon"))
        if self.is_wandb:
            wandb.log({
                "Validation: Jensen-Shannon distance between fake and benign samples".format(): sum(
                            evaluation_info["distance_metrics"]["jensenshannon"]["between_fake_and_benign"]) / len(
                            evaluation_info["distance_metrics"]["jensenshannon"]["between_fake_and_benign"])
            })

        fid_score = calculate_fid(np.concatenate(evaluation_info["intermediate_features"]["fake"]),
                                  self.benign_intermediate_features)
        evaluation_info["fid"] = fid_score

        d_output_real = torch.cat(d_output_real_list, 0)
        d_output_fake = torch.cat(d_output_fake_list, 0)

        errD = -torch.mean(d_output_real) + torch.mean(d_output_fake)
        errG = -torch.mean(d_output_fake)

        if self.is_wandb:
            wandb.log({
                "Validation: D loss": errD.item(),
                "Validation: G loss": errG.item()
            })

        evaluation_info["losses"]["D"] = errD.item()
        evaluation_info["losses"]["G"] = errG.item()

        self.print_info(evaluation_info, len(malicious_dataset))
        self.log_info(evaluation_info, len(malicious_dataset))
        self.write_to_file(evaluation_info, output_filepath, len(malicious_dataset))

        self.generator.train()
        self.discriminator.train()
        return evaluation_info

    def train(self, training_parameters:dict, patience:int=5):
        criterions = self.initialize_criterions()
        optimizers = self.initialize_optimizers(training_parameters)
        self.watch_models(log="gradients", log_freq=training_parameters["eval_interval"])

        train_D_losses = []
        val_D_losses = []

        train_G_losses = []
        val_G_losses = []

        best_val_D_loss = sys.maxsize
        best_val_G_loss = sys.maxsize

        info = self.initialize_training_info_dictionary()
        for epoch in range(training_parameters["num_epochs"]):
            optimizers, info = self.run_training_epoch(criterions, optimizers, info, training_parameters, epoch)
            train_D_losses.append(sum(info["losses"]["train"]["D"]) / len(info["losses"]["train"]["D"]))
            train_G_losses.append(sum(info["losses"]["train"]["D"]) / len(info["losses"]["train"]["D"]))
            if epoch % training_parameters["validation_interval"] == 0:
                # Validate on the whole validation data
                val_D, val_G = self.validate(epoch, training_parameters["validation_batch_size"])
                val_D_losses.append(val_D)
                val_G_losses.append(val_G)
                if abs(val_D) <= best_val_D_loss:
                    best_val_D_loss = abs(val_D)
                    torch.save(self.generator,
                               os.path.join(self.output_filepath, "generator/generator_best_val_D.pt"))
                    torch.save(self.discriminator, os.path.join(self.output_filepath,
                                                                "discriminator/discriminator_best_val_D.pt"))
                if abs(val_G) <= best_val_G_loss:
                    best_val_G_loss = abs(val_G)
                    torch.save(self.generator,
                               os.path.join(self.output_filepath, "generator/generator_best_val_G.pt"))
                    torch.save(self.discriminator, os.path.join(self.output_filepath,
                                                                "discriminator/discriminator_best_val_G.pt"))

            if epoch % training_parameters["save_interval"] == 0:
                torch.save(self.generator,
                           os.path.join(self.output_filepath, "generator/generator_checkoint_{}.pt".format(epoch)))
                torch.save(self.discriminator, os.path.join(self.output_filepath,
                                                            "discriminator/discriminator_checkpoint_{}.pt".format(
                                                                epoch)))
            # Early-stopping


        # Save generator and discriminator models (last epoch)
        torch.save(self.generator, os.path.join(self.output_filepath, "generator/generator.pt"))
        torch.save(self.discriminator, os.path.join(self.output_filepath, "discriminator/discriminator.pt"))
        self.plot_results(info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wassertein GAN training')
    parser.add_argument("--malware_histogram_features_filepath",
                        type=str,
                        help="Filepath to maliciouss import features",
                        default="../../../npz/BODMAS/histogram_features/BODMAS/")
    parser.add_argument("--malware_ember_features_filepath_version1",
                        type=str,
                        help="Filepath to malicious EMBER features (Version 1)",
                        default="../../../npz/BODMAS/ember_features/2017/BODMAS/")
    parser.add_argument("--malware_ember_features_filepath_version2",
                        type=str,
                        help="Filepath to malicious EMBER features (Version 2)",
                        default="../../../npz/BODMAS/ember_features/2018/BODMAS/")
    parser.add_argument("--malware_raw_executables_filepath",
                        type=str,
                        help="Filepath to goodware raw executables",
                        default="../../../npz/BODMAS/raw/BODMAS/")
    parser.add_argument("--malware_raw_npz_executables_filepath",
                        type=str,
                        help="Filepath to goodware raw executables",
                        default="../../../npz/BODMAS/raw_npz/BODMAS/")
    parser.add_argument("--training_malware_annotations_filepath",
                        type=str,
                        help="Filepath to malicious annotations (training)",
                        default="../../../annotations/byte_histogram/data/bodmas_train.csv")
    parser.add_argument("--validation_malware_annotations_filepath",
                        type=str,
                        help="Filepath to malicious annotations (validation)",
                        default="../../../annotations/byte_histogram/data/bodmas_validation.csv")
    parser.add_argument("--goodware_histogram_features_filepath",
                        type=str,
                        help="Filepath to benign import features",
                        default="../../../npz/BODMAS/histogram_features/benign/")
    parser.add_argument("--goodware_ember_features_filepath_version1",
                        type=str,
                        help="Filepath to goodware EMBER features (Version 1)",
                        default="../../../npz/BODMAS/ember_features/2017/benign/")
    parser.add_argument("--goodware_ember_features_filepath_version2",
                        type=str,
                        help="Filepath to goodware EMBER features (Version 2)",
                        default="../../../npz/BODMAS/ember_features/2018/benign/")
    parser.add_argument("--goodware_raw_executables_filepath",
                        type=str,
                        help="Filepath to goodware raw executables",
                        default="../../../npz/BODMAS/raw/benign/")
    parser.add_argument("--goodware_raw_npz_executables_filepath",
                        type=str,
                        help="Filepath to goodware raw executables",
                        default="../../../npz/BODMAS/raw_npz/benign/")
    parser.add_argument("--training_goodware_annotations_filepath",
                        type=str,
                        help="Filepath to benign annotations (training)",
                        default="../../../annotations/byte_histogram/data/benign_train.csv")
    parser.add_argument("--validation_goodware_annotations_filepath",
                        type=str,
                        help="Filepath to benign annotations (validation)",
                        default="../../../annotations/byte_histogram/data/benign_validation.csv")
    parser.add_argument("--output_filepath",
                        type=str,
                        help="Generator and discriminator torch models filepaths",
                        default="models/Conditional_GAN_byte_features/")
    parser.add_argument("--generator_parameters_filepath",
                        type=str,
                        help="Filepath of the generator parameters",
                        default="hyperparameters/generator_network/baseline_parameters.json")
    parser.add_argument("--discriminator_parameters_filepath",
                        type=str,
                        help="Filepath of the discriminator parameters",
                        default="hyperparameters/discriminator_network/baseline_parameters.json")
    parser.add_argument("--training_parameters_filepath",
                        type=str,
                        help="Training parameters filepath",
                        default="training_parameters/training_parameters.json")
    parser.add_argument("--test_malware_annotations_filepath",
                        type=str,
                        help="Filepath to malware annotations (test)",
                        default="../../../annotations/byte_histogram/data/bodmas_test.csv")
    parser.add_argument("--test_goodware_annotations_filepath",
                        type=str,
                        help="Filepath to benign annotations (test)",
                        default="../../../annotations/byte_histogram/data/benign_test.csv")
    parser.add_argument("--feature_version",
                        type=int,
                        help="EMBER feature version",
                        default=2)

    parser.add_argument("--histogram_lightgbm_model_filepath_version1",
                        type=str,
                        help="Byte histogram black-box detector (Version 1)",
                        default="../../../ml_models/lightgbm_models/byte_histogram_detector/byte_histogram_subset_model_2017.txt")
    parser.add_argument("--histogram_lightgbm_model_filepath_version2",
                        type=str,
                        help="Byte histogram black-box detector (Version 2)",
                        default="../../../ml_models/lightgbm_models/byte_histogram_detector/byte_histogram_subset_model_2018.txt")
    parser.add_argument('--histogram-version1', dest='histogram_version1', action='store_true')
    parser.add_argument('--no-histogram-version1', dest='histogram_version1', action='store_false')
    parser.set_defaults(histogram_version1=False)
    parser.add_argument('--histogram-version2', dest='histogram_version2', action='store_true')
    parser.add_argument('--no-histogram-version2', dest='histogram_version2', action='store_false')
    parser.set_defaults(histogramversion2=False)

    parser.add_argument("--ember_model_filepath_version1",
                        type=str,
                        help="EMBER 2017 Black-box detector (Version 1)",
                        default="../../../ml_models/lightgbm_models/ember_detector/ember_model_2017.txt")
    parser.add_argument("--ember_model_filepath_version2",
                        type=str,
                        help="EMBER 2018 Black-box detector (Version 2)",
                        default="../../../ml_models/lightgbm_models/ember_detector/ember_model_2018.txt")
    parser.add_argument('--ember-version1', dest='ember_version1', action='store_true')
    parser.add_argument('--no-ember-version1', dest='ember_version1', action='store_false')
    parser.set_defaults(ember_version1=False)
    parser.add_argument('--ember-version2', dest='ember_version2', action='store_true')
    parser.add_argument('--no-ember-version2', dest='ember_version2', action='store_false')
    parser.set_defaults(ember_version2=False)

    parser.add_argument("--sorel20m_lightgbm_model_filepath",
                        type=str,
                        help="SOREL-20M lightGBM black-box detector (Version 2)",
                        default="../../../ml_models/lightgbm_models/sorel20m_detector/seed0/lightgbm.model")
    parser.add_argument("--sorel20m_ffnn_model_filepath",
                        type=str,
                        help="SOREL-20M FFNN black-box detector (Version 2)",
                        default="../../../ml_models/torch_models/sorel20m_detector/seed0/epoch_10.pt")
    parser.add_argument('--sorel20m-lightgbm', dest='sorel20m_lightgbm', action='store_true')
    parser.add_argument('--no-sorel20m-lightgbm', dest='sorel20m_lightgbm', action='store_false')
    parser.set_defaults(sorel20m_lightgbm=False)
    parser.add_argument('--sorel20m-ffnn', dest='sorel20m_ffnn', action='store_true')
    parser.add_argument('--no-sorel20m-ffnn', dest='sorel20m_ffnn', action='store_false')
    parser.set_defaults(sorel20m_ffnn=False)

    parser.add_argument("--malconv_model_filepath",
                        type=str,
                        help="MalConv black-box detector (Version 2)",
                        default="../../../ml_models/torch_models/malconv_detector/models/malconv.h5")
    parser.add_argument('--malconv', dest='malconv', action='store_true')
    parser.add_argument('--no-malconv', dest='malconv', action='store_false')
    parser.set_defaults(malconv=False)

    parser.add_argument("--nonneg_malconv_model_filepath",
                        type=str,
                        help="Non-negative MalConv black-box detector (Version 2)",
                        default="../../../ml_models/torch_models/nonnegative_malconv_detector/models/nonneg.checkpoint")
    parser.add_argument('--nonneg-malconv', dest='nonneg_malconv', action='store_true')
    parser.add_argument('--no-nonneg-malconv', dest='nonneg_malconv', action='store_false')
    parser.set_defaults(nonneg_malconv=False)

    parser.add_argument("--cuda_device",
                        type=int,
                        help="Cuda device",
                        default=None)

    parser.add_argument("--inception_parameters_filepath",
                        type=str,
                        help="Inception parameters",
                        default="../../../ml_models/torch_models/inception_detector/network_parameters/byte_histogram_net_params.json")
    parser.add_argument("--inception_checkpoint",
                        type=str,
                        help="Inception checkpoint",
                        default="../../../ml_models/torch_models/inception_detector/models/byte_histogram_model_2017/best/best_model.pt")
    parser.add_argument("--inception_features_filepath",
                        type=str,
                        help="Inception benign intermediate features",
                        default="../../../ml_models/torch_models/inception_detector/intermediate_features/byte_histogram_2017.npz")

    parser.add_argument('--wandb', dest='wandb', action='store_true')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false')
    parser.set_defaults(wandb=True)
    args = parser.parse_args()



    wassertein_gan = WasserteinGANGradientPenalty()
    wassertein_gan.create_logger(args.output_filepath)
    wassertein_gan.init_directories(args.output_filepath)
    wassertein_gan.init_cuda(args.cuda_device)
    wassertein_gan.initialize_feature_extractor()
    wassertein_gan.initialize_ember_feature_extractors()

    wassertein_gan.initialize_training_and_validation_generator_datasets(
        args.malware_histogram_features_filepath,
        args.malware_ember_features_filepath_version1,
        args.malware_ember_features_filepath_version2,
        args.malware_raw_executables_filepath,
        args.malware_raw_npz_executables_filepath,
        args.training_malware_annotations_filepath,
        args.validation_malware_annotations_filepath
    )
    wassertein_gan.initialize_training_and_validation_discriminator_datasets(
        args.goodware_histogram_features_filepath,
        args.goodware_ember_features_filepath_version1,
        args.goodware_ember_features_filepath_version2,
        args.goodware_raw_executables_filepath,
        args.goodware_raw_npz_executables_filepath,
        args.training_goodware_annotations_filepath,
        args.validation_goodware_annotations_filepath
    )

    wassertein_gan.build_generator_network(args.generator_parameters_filepath)
    wassertein_gan.build_discriminator_network(args.discriminator_parameters_filepath)

    wassertein_gan.build_blackbox_detectors(
        args.histogram_lightgbm_model_filepath_version1,
        args.histogram_version1,
        args.histogram_lightgbm_model_filepath_version2,
        args.histogram_version2,
        args.ember_model_filepath_version1,
        args.ember_version1,
        args.ember_model_filepath_version2,
        args.ember_version2,
        args.sorel20m_lightgbm_model_filepath,
        args.sorel20m_lightgbm,
        args.sorel20m_ffnn_model_filepath,
        args.sorel20m_ffnn,
        args.malconv_model_filepath,
        args.malconv,
        args.nonneg_malconv_model_filepath,
        args.nonneg_malconv
    )

    inception_parameters = load_json(args.inception_parameters_filepath)
    wassertein_gan.load_inception_model(inception_parameters, args.inception_checkpoint)
    wassertein_gan.load_inception_benign_intermediate_features(args.inception_features_filepath)

    training_parameters = load_json(args.training_parameters_filepath)
    wassertein_gan.init_wandb(training_parameters, args.wandb)
    wassertein_gan.train(training_parameters)

    if args.test_malware_annotations_filepath is not None and args.test_goodware_annotations_filepath is not None:
        test_discriminator_dataset = ByteHistogramDataset(
            args.goodware_histogram_features_filepath,
            args.goodware_ember_features_filepath_version1,
            args.goodware_ember_features_filepath_version2,
            args.goodware_raw_executables_filepath,
            args.goodware_raw_npz_executables_filepath,
            args.test_goodware_annotations_filepath
        )
        test_generator_dataset = ByteHistogramDataset(
            args.malware_histogram_features_filepath,
            args.malware_ember_features_filepath_version1,
            args.malware_ember_features_filepath_version2,
            args.malware_raw_executables_filepath,
            args.malware_raw_npz_executables_filepath,
            args.test_malware_annotations_filepath
        )
        wassertein_gan.generator = torch.load(os.path.join(args.output_filepath, "generator/generator_best_fid.pt"), map_location=wassertein_gan.device)

        wassertein_gan.test(test_generator_dataset, training_parameters["validation_batch_size"])
