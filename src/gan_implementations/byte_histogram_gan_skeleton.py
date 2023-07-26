import torch
from src.gan_implementations.gan_skeleton import SkeletonGAN
from src.feature_extractors.bytes_histogram_extractor import ByteHistogramExtractor
from src.pe_modifier import PEModifier
from src.gan_implementations.solver.utils import calculate_histogram, heuristic_approach, append_bytes_given_target_histogram
from src.gan_implementations.utils import plot_distance_metric, plot_evasion_rate, plot_generator_and_discriminator_training_loss, plot_generator_and_discriminator_validation_loss, plot_frechlet_inception_distance
from src.gan_implementations.metrics.frechlet_inception_distance import calculate_fid
from src.gan_implementations.byte_histogram_dataset import ByteHistogramDataset
from src.ml_models.torch_models.inception_detector.byte_histogram_net import ByteHistogramNetwork
from src.gan_implementations.metrics.jensen_shannon_distance import jensen_shannon_distance_torch, jensen_shannon_distance_np
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist
import copy
import sys
import numpy as np
from abc import abstractmethod
import os
import wandb


class ByteHistogramGAN(SkeletonGAN):

    def initialize_feature_extractor(self):
        self.feature_extractor = ByteHistogramExtractor()

    def initialize_training_and_validation_discriminator_datasets(
            self,
            goodware_histogram_features_filepath,
            goodware_ember_features_filepath_version1,
            goodware_ember_features_filepath_version2,
            goodware_raw_executables_filepath,
            goodware_raw_npz_executables_filepath,
            training_goodware_annotations_filepath,
            validation_goodware_annotations_filepath
    ):
        # Create discriminator's training and validation dataset
        self.training_discriminator_dataset = ByteHistogramDataset(
            goodware_histogram_features_filepath,
            goodware_ember_features_filepath_version1,
            goodware_ember_features_filepath_version2,
            goodware_raw_executables_filepath,
            goodware_raw_npz_executables_filepath,
            training_goodware_annotations_filepath
        )
        self.validation_discriminator_dataset = ByteHistogramDataset(
            goodware_histogram_features_filepath,
            goodware_ember_features_filepath_version1,
            goodware_ember_features_filepath_version2,
            goodware_raw_executables_filepath,
            goodware_raw_npz_executables_filepath,
            validation_goodware_annotations_filepath
        )

    def initialize_training_and_validation_generator_datasets(
            self,
            malware_histogram_features_filepath,
            malware_ember_features_filepath_version1,
            malware_ember_features_filepath_version2,
            malware_raw_executables_filepath,
            malware_raw_npz_executables_filepath,
            training_malware_annotations_filepath,
            validation_malware_annotations_filepath
    ):
        self.training_generator_dataset = ByteHistogramDataset(
            malware_histogram_features_filepath,
            malware_ember_features_filepath_version1,
            malware_ember_features_filepath_version2,
            malware_raw_executables_filepath,
            malware_raw_npz_executables_filepath,
            training_malware_annotations_filepath
        )
        self.validation_generator_dataset = ByteHistogramDataset(
            malware_histogram_features_filepath,
            malware_ember_features_filepath_version1,
            malware_ember_features_filepath_version2,
            malware_raw_executables_filepath,
            malware_raw_npz_executables_filepath,
            validation_malware_annotations_filepath
        )

    def load_inception_model(self, inception_parameters, inception_checkpoint):
        self.inception_model = ByteHistogramNetwork(inception_parameters)
        self.inception_model.load_state_dict(torch.load(inception_checkpoint, map_location=torch.device('cpu')))
        self.inception_model.eval()
        #self.inception_model.to(torch.device("cpu"))
        return self.inception_model

    def load_inception_benign_intermediate_features(self, inception_features_filepath: str):
        self.benign_intermediate_features = np.load(inception_features_filepath, allow_pickle=True)["arr_0"]
        return self.benign_intermediate_features

    def evaluate_on_histogram_model_2017(self, histogram_features):
        if self.features_model_2017 is not None:
            model_output = self.features_model_2017.predict(histogram_features)
            detections = model_output.round().astype(int)
            detected = np.sum(detections)
            return detected
        return None

    def evaluate_on_histogram_model_2018(self, histogram_features):
        if self.features_model_2018 is not None:
            model_output = self.features_model_2018.predict(histogram_features)
            detections = model_output.round().astype(int)
            detected = np.sum(detections)
            return detected
        return None

    @abstractmethod
    def run_training_epoch(self, criterions: dict, optimizers: dict, info: dict, training_parameters: dict, epoch: int):
        pass

    def initialize_training_info_dictionary(self):
        return {
            "losses": {
                "train":{
                    "G": [],
                    "D": []
                },
                "validation": {
                    "G": [],
                    "D": []
                },
                "best":{
                    "G": sys.maxsize,
                    "D": sys.maxsize
                }
            },
            "evasion_rate":{
                "best":{
                    "byte_histogram_model_2017": 1.0,
                    "byte_histogram_model_2018": 1.0,
                    "ember_model_2017": 1.0,
                    "ember_model_2018": 1.0,
                    "sorel20m_lightgbm_model": 1.0,
                    "sorel20m_ffnn_model": 1.0,
                    "malconv_model": 1.0,
                    "nonnegative_malconv_model": 1.0,
                    "malconvgct_model": 1.0
                },
                "list":{
                    "byte_histogram_model_2017": [],
                    "byte_histogram_model_2018": [],
                    "ember_model_2017": [],
                    "ember_model_2018": [],
                    "sorel20m_lightgbm_model": [],
                    "sorel20m_ffnn_model": [],
                    "malconv_model": [],
                    "nonnegative_malconv_model": [],
                    "malconvgct_model": []
                }
            },
            "fid":{
                "best":sys.maxsize,
                "scores": []
            },
            "distance_metrics": {
                "train": {
                    "jensenshannon": {
                        "between_original_and_fake": [],
                        "between_original_and_benign": [],
                        "between_fake_and_benign":[],
                        "between_fake": []
                    }
                },
                "validation": {
                    "jensenshannon": {
                        "between_original_and_fake": [],
                        "between_original_and_benign": [],
                        "between_fake_and_benign": [],
                        "between_fake": []
                    },
                }
            },
            "bytez_size": {
                "original": [],
                "fake": []
            }
        }

    def initialize_evaluation_info_dictionary(self):
        return {
            "losses":{
                "G": 0,
                "D": 0
            },
            "detections": {
                "original": {
                    "byte_histogram_model_2017": 0,
                    "byte_histogram_model_2018": 0,
                    "ember_model_2017": 0,
                    "ember_model_2018": 0,
                    "sorel20m_lightgbm_model": 0,
                    "sorel20m_ffnn_model": 0,
                    "malconv_model": 0,
                    "nonneg_malconv_model": 0,
                    "malconvgct_model": 0
                },
                "fake": {
                    "byte_histogram_model_2017": 0,
                    "byte_histogram_model_2018": 0,
                    "ember_model_2017": 0,
                    "ember_model_2018": 0,
                    "sorel20m_lightgbm_model": 0,
                    "sorel20m_ffnn_model": 0,
                    "malconv_model": 0,
                    "nonneg_malconv_model": 0,
                    "malconvgct_model": 0
                }
            },
            "intermediate_features": {
                "benign": [],
                "fake": []
            },
            "fid":sys.maxsize,
            "distance_metrics": {
                "jensenshannon": {
                    "between_original_and_fake": [],
                    "between_original_and_benign": [],
                    "between_fake_and_benign": [],
                    "between_fake": []
                }
            },
            "bytez_size": {
                "original": [],
                "fake": []
            }
        }

    def print_info(self, info:dict, size:int):
        print("Byte histogram model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["byte_histogram_model_2017"], size,
            info["detections"]["fake"]["byte_histogram_model_2017"], size))
        print("Byte histogram model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["byte_histogram_model_2018"], size,
            info["detections"]["fake"]["byte_histogram_model_2018"], size))
        print("EMBER model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["ember_model_2017"], size,
            info["detections"]["fake"]["ember_model_2017"], size))
        print("EMBER model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["ember_model_2018"], size,
            info["detections"]["fake"]["ember_model_2018"], size))
        print("SOREL-20M LightGBM model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["sorel20m_lightgbm_model"], size,
            info["detections"]["fake"]["sorel20m_lightgbm_model"], size))
        print("SOREL-20M FFNN model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["sorel20m_ffnn_model"], size,
            info["detections"]["fake"]["sorel20m_ffnn_model"], size))
        print("MalConv model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["malconv_model"], size,
            info["detections"]["fake"]["malconv_model"], size))
        print("Non-negative MalConv model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["nonneg_malconv_model"], size,
            info["detections"]["fake"]["nonneg_malconv_model"], size))
        print("MalConvGCT model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["malconvgct_model"], size,
            info["detections"]["fake"]["malconvgct_model"], size))
        print("Frechlet Inception Distance: {}".format(info["fid"]))
        for metric in info["distance_metrics"]:
            print("{} distance between original and fake samples: {}".format(metric, sum(info["distance_metrics"][metric]["between_original_and_fake"])/len(info["distance_metrics"][metric]["between_original_and_fake"])))
            print("{} distance between fake samples: {}".format(metric, sum(info["distance_metrics"][metric]["between_original_and_fake"])/len(info["distance_metrics"][metric]["between_original_and_fake"])))

        try:
            print("Average size of original executables: {}".format(sum(info["bytez_size"]["original"])/len(info["bytez_size"]["original"])))
            print("Average size of fake executables: {}".format(sum(info["bytez_size"]["fake"])/len(info["bytez_size"]["fake"])))
        except ZeroDivisionError:
            pass

    def log_info(self, info: dict, size: int):
        self.logger.info("\n\nEvaluation results:")
        self.logger.info("Byte histogram model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["byte_histogram_model_2017"], size,
            info["detections"]["fake"]["byte_histogram_model_2017"], size))
        self.logger.info("Byte histogram model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["byte_histogram_model_2018"], size,
            info["detections"]["fake"]["byte_histogram_model_2018"], size))
        self.logger.info("EMBER model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["ember_model_2017"], size,
            info["detections"]["fake"]["ember_model_2017"], size))
        self.logger.info("EMBER model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["ember_model_2018"], size,
            info["detections"]["fake"]["ember_model_2018"], size))
        self.logger.info("SOREL-20M LightGBM model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["sorel20m_lightgbm_model"], size,
            info["detections"]["fake"]["sorel20m_lightgbm_model"], size))
        self.logger.info("SOREL-20M FFNN model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["sorel20m_ffnn_model"], size,
            info["detections"]["fake"]["sorel20m_ffnn_model"], size))
        self.logger.info("MalConv model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["malconv_model"], size,
            info["detections"]["fake"]["malconv_model"], size))
        self.logger.info("Non-negative MalConv model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["nonneg_malconv_model"], size,
            info["detections"]["fake"]["nonneg_malconv_model"], size))
        self.logger.info("MalConvGCT model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["malconvgct_model"], size,
            info["detections"]["fake"]["malconvgct_model"], size))
        self.logger.info("Frechlet Inception Distance: {}".format(info["fid"]))
        for metric in info["distance_metrics"]:
            self.logger.info("{} distance between original and fake samples: {}".format(metric, sum(info["distance_metrics"][metric]["between_original_and_fake"])/len(info["distance_metrics"][metric]["between_original_and_fake"])))
            self.logger.info("{} distance between fake samples: {}".format(metric, sum(info["distance_metrics"][metric]["between_original_and_fake"])/len(info["distance_metrics"][metric]["between_original_and_fake"])))

        try:
            self.logger.info("Average size of original executables: {}".format(
                sum(info["bytez_size"]["original"]) / len(info["bytez_size"]["original"])))
            self.logger.info("Average size of fake executables: {}".format(
                sum(info["bytez_size"]["fake"]) / len(info["bytez_size"]["fake"])))
        except ZeroDivisionError:
            pass


    def write_to_file(self, info:dict, output_filepath:str, size:int):
        with open(output_filepath, "w") as output_file:
            # Evasion rates
            output_file.write("Results\n\n")
            output_file.write(
                "Byte histogram model 2017; Originally detected: {}/{}; Fake detected: {}/{}\n".format(
                    info["detections"]["original"]["byte_histogram_model_2017"], size,
                    info["detections"]["fake"]["byte_histogram_model_2017"], size))
            output_file.write(
                "Byte histogram model 2018; Originally detected: {}/{}; Fake detected: {}/{}\n".format(
                    info["detections"]["original"]["byte_histogram_model_2018"], size,
                    info["detections"]["fake"]["byte_histogram_model_2018"], size))
            output_file.write("EMBER model 2017; Originally detected: {}/{}; Fake detected: {}/{}\n".format(
                info["detections"]["original"]["ember_model_2017"], size,
                info["detections"]["fake"]["ember_model_2017"], size))
            output_file.write("EMBER model 2018; Originally detected: {}/{}; Fake detected: {}/{}\n".format(
                info["detections"]["original"]["ember_model_2018"], size,
                info["detections"]["fake"]["ember_model_2018"], size))
            output_file.write(
                "SOREL-20M LightGBM model; Originally detected: {}/{}; Fake detected: {}/{}\n".format(
                    info["detections"]["original"]["sorel20m_lightgbm_model"],
                    size,
                    info["detections"]["fake"]["sorel20m_lightgbm_model"],
                    size))
            output_file.write(
                "SOREL-20M FFNN model; Originally detected: {}/{}; Fake detected: {}/{}\n".format(
                    info["detections"]["original"]["sorel20m_ffnn_model"],
                    size,
                    info["detections"]["fake"]["sorel20m_ffnn_model"], size))
            output_file.write("MalConv model; Originally detected: {}/{}; Fake detected: {}/{}\n\n".format(
                info["detections"]["original"]["malconv_model"], size,
                info["detections"]["fake"]["malconv_model"], size))
            output_file.write("Non-negative MalConv model; Originally detected: {}/{}; Fake detected: {}/{}\n\n".format(
                info["detections"]["original"]["nonneg_malconv_model"], size,
                info["detections"]["fake"]["nonneg_malconv_model"], size))
            output_file.write("MalConvGCT model; Originally detected: {}/{}; Fake detected: {}/{}\n\n".format(
                info["detections"]["original"]["malconvgct_model"], size,
                info["detections"]["fake"]["malconvgct_model"], size))
            output_file.write("FID score: {}\n".format(info["fid"]))
            for metric in info["distance_metrics"]:
                output_file.write("{} distance between original and fake samples: {}\n".format(metric, sum(
                    info["distance_metrics"][metric]["between_original_and_fake"]) / len(
                    info["distance_metrics"][metric]["between_original_and_fake"])))
                output_file.write("{} distance between fake samples: {}\n".format(metric, sum(
                    info["distance_metrics"][metric]["between_original_and_fake"]) / len(
                    info["distance_metrics"][metric]["between_original_and_fake"])))

            try:
                output_file.write("Average size of original executables: {}".format(
                    sum(info["bytez_size"]["original"]) / len(info["bytez_size"]["original"])))
                output_file.write("Average size of fake executables: {}".format(
                    sum(info["bytez_size"]["fake"]) / len(info["bytez_size"]["fake"])))
            except ZeroDivisionError:
                pass

    def train(self, training_parameters:dict):
        criterions = self.initialize_criterions()
        optimizers = self.initialize_optimizers(training_parameters)
        self.watch_models(log="gradients", log_freq=training_parameters["eval_interval"])
        info = self.initialize_training_info_dictionary()
        for epoch in range(training_parameters["num_epochs"]):
            optimizers, info = self.run_training_epoch(criterions, optimizers, info, training_parameters, epoch)
            if epoch % training_parameters["validation_interval"] == 0:
                # Validate on the whole validation data
                self.validate(self.validation_discriminator_dataset, self.validation_generator_dataset, epoch, training_parameters["validation_batch_size"])
            if epoch % training_parameters["save_interval"] == 0:
                torch.save(self.generator,
                           os.path.join(self.output_filepath, "generator/generator_checkoint_{}.pt".format(epoch)))
                torch.save(self.discriminator, os.path.join(self.output_filepath,
                                                            "discriminator/discriminator_checkpoint_{}.pt".format(
                                                                epoch)))
        # Save generator and discriminator models (last epoch)
        torch.save(self.generator, os.path.join(self.output_filepath, "generator/generator.pt"))
        torch.save(self.discriminator, os.path.join(self.output_filepath, "discriminator/discriminator.pt"))
        self.plot_results(info)

    def plot_results(self, info:dict):
        # Plot evasion rates during evaluations
        plot_evasion_rate(info["evasion_rate"]["list"]["byte_histogram_model_2017"], "Byte histogram model 2017",
                          os.path.join(self.output_filepath, "plots/evasion_rate_byte_histogram_model_2017.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["byte_histogram_model_2018"], "Byte histogram model 2018",
                          os.path.join(self.output_filepath, "plots/evasion_rate_byte_histogram_model_2018.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["ember_model_2017"], "EMBER model 2017",
                          os.path.join(self.output_filepath, "plots/evasion_rate_ember_model_2017.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["ember_model_2018"], "EMBER model 2018",
                          os.path.join(self.output_filepath, "plots/evasion_rate_ember_model_2018.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["sorel20m_lightgbm_model"], "SOREL-20M LightGBM",
                          os.path.join(self.output_filepath, "plots/evasion_rate_sorel20m_lightgbm_model.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["sorel20m_ffnn_model"], "SOREL-20M FFNN",
                          os.path.join(self.output_filepath, "plots/evasion_rate_sorel20m_ffnn_model.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["malconv_model"], "MalConv",
                          os.path.join(self.output_filepath, "plots/evasion_rate_malconv_model.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["nonnegative_malconv_model"], "Non-negative MalConv",
                          os.path.join(self.output_filepath, "plots/evasion_rate_nonnegative_malconv_model.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["malconvgct_model"], "MalConvGCT",
                          os.path.join(self.output_filepath, "plots/evasion_rate_malconvgct_model.png"))

        for metric in info["distance_metrics"]["train"].keys():
            plot_distance_metric(
                info["distance_metrics"]["train"][metric]["between_original_and_fake"],
                "Training {} distance between original and fake samples",
                os.path.join(self.output_filepath, "plots/training_{}_distance_between_original_and_fake_samples.png")
            )
            plot_distance_metric(
                info["distance_metrics"]["train"][metric]["between_fake"],
                "Training {} distance between fake samples",
                os.path.join(self.output_filepath, "plots/training_{}_distance_between_original_and_fake_samples.png")
            )

            plot_distance_metric(
                info["distance_metrics"]["validation"][metric]["between_original_and_fake"],
                "Validation {} distance between original and fake samples",
                os.path.join(self.output_filepath, "plots/validation_{}_distance_between_original_and_fake_samples.png")
            )
            plot_distance_metric(
                info["distance_metrics"]["validation"][metric]["between_fake"],
                "Validation {} distance between fake samples",
                os.path.join(self.output_filepath, "plots/validation_{}_distance_between_original_and_fake_samples.png")
            )

        with open(os.path.join(self.output_filepath, "generator/losses.csv"), "w") as output_file:
            for x in info["losses"]["train"]["G"]:
                output_file.write("{}\n".format(x))
        with open(os.path.join(self.output_filepath, "discriminator/losses.csv"), "w") as output_file:
            for x in info["losses"]["train"]["D"]:
                output_file.write("{}\n".format(x))
        plot_generator_and_discriminator_training_loss(info["losses"]["train"]["G"], info["losses"]["train"]["D"],
                                                       os.path.join(self.output_filepath,
                                                                    "plots/generator_and_discriminator_training_loss.png"))
        plot_generator_and_discriminator_validation_loss(info["losses"]["validation"]["G"],
                                                         info["losses"]["validation"]["D"],
                                                         os.path.join(self.output_filepath,
                                                                      "plots/generator_and_discriminator_validation_loss.png"))

        with open(os.path.join(self.output_filepath, "fid_scores.csv"), "w") as output_file:
            for x in info["fid"]["scores"]:
                output_file.write("{}\n".format(x))
                plot_frechlet_inception_distance(info["fid"]["scores"], os.path.join(self.output_filepath,
                                                                                     "plots/frechlet_inception_distances.png"))



    def log(self, d_features, fake_samples, g_features, g_ember_features_v1, g_ember_features_v2, g_raw_paths, g_raw_npz_paths, batch_size:int, info:dict):
        """
        Check fake samples against the malware detectors
        """
        benign_features = d_features.cpu().detach().numpy()
        fake_features = fake_samples.cpu().detach().numpy()

        # Convert Torch tensors to numpy arrays
        g_features = g_features.cpu().detach().numpy().astype(np.float32)
        g_ember_features_v1 = g_ember_features_v1.cpu().detach().numpy()
        g_ember_features_v2 = g_ember_features_v2.cpu().detach().numpy()

        info["distance_metrics"]["train"]["jensenshannon"]["between_original_and_fake"].append(jensen_shannon_distance_np(g_features, fake_features))
        info["distance_metrics"]["train"]["jensenshannon"]["between_original_and_benign"].append(jensen_shannon_distance_np(g_features, benign_features))
        info["distance_metrics"]["train"]["jensenshannon"]["between_fake_and_benign"].append(jensen_shannon_distance_np(fake_features, benign_features))
        info["distance_metrics"]["train"]["jensenshannon"]["between_fake"].append(pdist(fake_features, metric="jensenshannon"))

        #info["distance_metrics"]["train"] = self.calculate_distance_between_fake_samples(fake_features, info["distance_metrics"]["train"])
        #info["distance_metrics"]["train"] = self.calculate_distance_between_original_and_fake_samples(g_features, fake_features, info["distance_metrics"]["train"])

        ember_features_v1_batch = []
        ember_features_v2_batch = []
        original_bytez_batch = []
        fake_bytez_batch = []

        original_bytez_sizes = []
        fake_bytez_sizes = []
        for k in range(fake_features.shape[0]):
            if self.malconv_model or self.nonneg_malconv_model or self.malconvgct_model:
                original_bytez_arr = np.load(g_raw_npz_paths[k]+".npz", allow_pickle=True)["arr_0"].astype(np.uint16)
                original_bytez_list = original_bytez_arr.tolist()
                original_histogram = calculate_histogram(original_bytez_list)

                current_histogram, current_normalized_histogram = heuristic_approach(
                    original_bytez_list,
                    g_features[k],
                    fake_features[k]
                )
                fake_bytez_list = append_bytes_given_target_histogram(
                    original_bytez_list,
                    original_histogram,
                    current_histogram
                )
                fake_bytez_arr = np.array(fake_bytez_list)

                original_bytez_length = original_bytez_arr.shape[0]
                fake_bytez_length = fake_bytez_arr.shape[0]


                b = np.ones((self.malconv_max_length,), dtype=np.uint16) * self.PADDING_CHAR  # (1048576,)
                hex_values = original_bytez_arr[:self.malconv_max_length]  # (1048576,)
                b[:len(hex_values)] = hex_values  # (1048576,)
                b = np.expand_dims(b, axis=0)
                original_bytez_batch.append(b)
                original_bytez_sizes.append(original_bytez_length)

                c = np.ones((self.malconv_max_length,), dtype=np.uint16) * self.PADDING_CHAR  # (1048576,)
                hex_values = fake_bytez_arr[:self.malconv_max_length]  # (1048576,)
                c[:len(hex_values)] = hex_values  # (1048576,)
                c = np.expand_dims(c, axis=0)
                fake_bytez_batch.append(c)
                fake_bytez_sizes.append(fake_bytez_length)

            # Replace histogram features... Not the most correct way but the faster. Only for training/validation
            ember_features_v1_batch.append(
                self.ember_feature_extractor_v1.replace_byte_histogram_features(g_ember_features_v1[k],
                                                                                fake_features[k]))
            # Extract EMBER features (v2)
            ember_features_v2_batch.append(
                self.ember_feature_extractor_v2.replace_byte_histogram_features(g_ember_features_v2[k],
                                                                                fake_features[k]))

        fake_ember_features_v1 = np.array(ember_features_v1_batch).astype(np.float32)
        fake_ember_features_v2 = np.array(ember_features_v2_batch).astype(np.float32)

        if self.malconv_model or self.nonneg_malconv_model or self.malconvgct_model:
            original_bytez_batch = np.array(original_bytez_batch).astype(np.float32)
            original_bytez_batch = np.squeeze(original_bytez_batch, axis=1)
            fake_bytez_batch = np.array(fake_bytez_batch).astype(np.float32)
            fake_bytez_batch = np.squeeze(fake_bytez_batch, axis=1)

        if self.features_model_2017:
            originally_detected_byte_histogram_model_2017 = self.evaluate_on_histogram_model_2017(g_features)
            fake_detected_byte_histogram_model_2017 = self.evaluate_on_histogram_model_2017(fake_features)
        else:
            originally_detected_byte_histogram_model_2017 = 0
            fake_detected_byte_histogram_model_2017 = 0

        if self.features_model_2018:
            originally_detected_byte_histogram_model_2018 = self.evaluate_on_histogram_model_2018(g_features)
            fake_detected_byte_histogram_model_2018 = self.evaluate_on_histogram_model_2018(fake_features)
        else:
            originally_detected_byte_histogram_model_2018 = 0
            fake_detected_byte_histogram_model_2018 = 0

        if self.ember_model_2017:
            originally_detected_ember_model_2017 = self.evaluate_on_ember_model_2017(g_ember_features_v1)
            fake_detected_ember_model_2017 = self.evaluate_on_ember_model_2017(fake_ember_features_v1)
        else:
            originally_detected_ember_model_2017 = 0
            fake_detected_ember_model_2017 = 0

        if self.ember_model_2018:
            originally_detected_ember_model_2018 = self.evaluate_on_ember_model_2018(g_ember_features_v2)
            fake_detected_ember_model_2018 = self.evaluate_on_ember_model_2018(fake_ember_features_v2)
        else:
            originally_detected_ember_model_2018 = 0
            fake_detected_ember_model_2018 = 0

        if self.sorel20m_lightgbm_model:
            originally_detected_sorel20m_lightgbm_model = self.evaluate_on_sorel20m_lightgbm_model(g_ember_features_v2)
            fake_detected_sorel20m_lightgbm_model = self.evaluate_on_sorel20m_lightgbm_model(fake_ember_features_v2)
        else:
            originally_detected_sorel20m_lightgbm_model = 0
            fake_detected_sorel20m_lightgbm_model = 0

        if self.sorel20m_ffnn_model:
            originally_detected_sorel20m_ffnn_model = self.evaluate_on_sorel20m_ffnn_model(
                g_ember_features_v2)
            fake_detected_sorel20m_ffnn_model = self.evaluate_on_sorel20m_ffnn_model(fake_ember_features_v2)
        else:
            originally_detected_sorel20m_ffnn_model = 0
            fake_detected_sorel20m_ffnn_model = 0

        if self.malconv_model:
            originally_detected_malconv_model = self.evaluate_on_malconv_model(original_bytez_batch)
            fake_detected_malconv_model = self.evaluate_on_malconv_model(fake_bytez_batch)
        else:
            originally_detected_malconv_model = 0
            fake_detected_malconv_model = 0

        if self.nonneg_malconv_model:
            originally_detected_nonneg_malconv_model = self.evaluate_on_nonnegative_malconv_model(original_bytez_batch)
            fake_detected_nonneg_malconv_model = self.evaluate_on_nonnegative_malconv_model(fake_bytez_batch)
        else:
            originally_detected_nonneg_malconv_model = 0
            fake_detected_nonneg_malconv_model = 0

        if self.malconvgct_model:
            originally_detected_malconvgct_model = self.evaluate_on_malconvgct_model(original_bytez_batch)
            fake_detected_malconvgct_model = self.evaluate_on_malconvgct_model(fake_bytez_batch)
        else:
            originally_detected_malconvgct_model = 0
            fake_detected_malconvgct_model = 0

        self.logger.info("Byte histogram model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_byte_histogram_model_2017, batch_size,
            fake_detected_byte_histogram_model_2017, batch_size))
        self.logger.info("Byte histogram model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_byte_histogram_model_2018, batch_size,
            fake_detected_byte_histogram_model_2018, batch_size))
        self.logger.info("EMBER model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_ember_model_2017, batch_size,
            fake_detected_ember_model_2017, batch_size))
        self.logger.info("EMBER model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_ember_model_2018, batch_size,
            fake_detected_ember_model_2018, batch_size))
        self.logger.info("SOREL-20M LightGBM model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_sorel20m_lightgbm_model, batch_size,
            fake_detected_sorel20m_lightgbm_model, batch_size))
        self.logger.info("SOREL-20M FFNN model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_sorel20m_ffnn_model, batch_size,
            fake_detected_sorel20m_ffnn_model, batch_size))
        self.logger.info("MalConv model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_malconv_model, batch_size,
            fake_detected_malconv_model, batch_size))
        self.logger.info("Non-negative MalConv model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_nonneg_malconv_model, batch_size,
            fake_detected_nonneg_malconv_model, batch_size))
        self.logger.info("MalConvGCT model; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_malconvgct_model, batch_size,
            fake_detected_malconvgct_model, batch_size))

        for metric in info["distance_metrics"]["train"]:
            self.logger.info("{} distance between original and fake samples: {}".format(metric, info["distance_metrics"]["train"][metric]["between_original_and_fake"][-1]))
            self.logger.info("{} distance between original and benign samples: {}".format(metric, info["distance_metrics"]["train"][metric]["between_original_and_benign"][-1]))
            self.logger.info("{} distance between fake and benign samples: {}".format(metric, info["distance_metrics"]["train"][metric]["between_fake_and_benign"][-1]))
            self.logger.info("{} distance between fake samples: {}".format(metric, info["distance_metrics"]["train"][metric]["between_fake"][-1]))
            if self.is_wandb == True:
                wandb.log(
                    {
                        "Log: {} distance between original and fake samples".format(metric):
                            info["distance_metrics"]["train"][metric]["between_original_and_fake"][-1],
                        "Log: {} distance between original and benign samples".format(metric):
                            info["distance_metrics"]["train"][metric]["between_original_and_benign"][-1],
                        "Log: {} distance between fake and benign samples".format(metric):
                            info["distance_metrics"]["train"][metric]["between_fake_and_benign"][-1],
                        "Log: {} distance between fake samples".format(metric):
                            info["distance_metrics"]["train"][metric]["between_fake"][-1],
                    }
                )

        try:
            self.logger.info("Average size of original executables: {}".format(sum(original_bytez_sizes)/len(original_bytez_sizes)))
            self.logger.info("Average size of fake executables: {}".format(sum(fake_bytez_sizes)/len(fake_bytez_sizes)))
            wandb.log({
                "Original executables average size": sum(original_bytez_sizes)/len(original_bytez_sizes),
                "Fake executables average size": sum(fake_bytez_sizes)/len(fake_bytez_sizes)
            })
        except ZeroDivisionError:
            pass


        if self.is_wandb == True:
            detection_rates = {
                "byte_histogram_model_2017": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "byte_histogram_model_2018": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "ember_model_2017": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "ember_model_2018": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "sorel20m_lightgbm": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "sorel20m_ffnn": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "malconv": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "nonneg_malconv": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "malconvgct": {
                    "original": 0.0,
                    "fake": 0.0
                }
            }

            if originally_detected_byte_histogram_model_2017 is not None:
                detection_rates["byte_histogram_model_2017"]["original"] = originally_detected_byte_histogram_model_2017 / batch_size
                detection_rates["byte_histogram_model_2017"]["fake"] = fake_detected_byte_histogram_model_2017 / batch_size
            if originally_detected_byte_histogram_model_2018 is not None:
                detection_rates["byte_histogram_model_2018"]["original"] = originally_detected_byte_histogram_model_2018 / batch_size
                detection_rates["byte_histogram_model_2018"]["fake"] = fake_detected_byte_histogram_model_2018 / batch_size
            if originally_detected_ember_model_2017 is not None:
                detection_rates["ember_model_2017"]["original"] = originally_detected_ember_model_2017 / batch_size
                detection_rates["ember_model_2017"]["fake"] = fake_detected_ember_model_2017 / batch_size
            if originally_detected_ember_model_2018 is not None:
                detection_rates["ember_model_2018"]["original"] = originally_detected_ember_model_2018 / batch_size
                detection_rates["ember_model_2018"]["fake"] = fake_detected_ember_model_2018 / batch_size
            if originally_detected_sorel20m_lightgbm_model is not None:
                detection_rates["sorel20m_lightgbm"]["original"] = originally_detected_sorel20m_lightgbm_model / batch_size
                detection_rates["sorel20m_lightgbm"]["fake"] = fake_detected_sorel20m_lightgbm_model / batch_size
            if originally_detected_sorel20m_ffnn_model is not None:
                detection_rates["sorel20m_ffnn"]["original"] = originally_detected_sorel20m_ffnn_model / batch_size
                detection_rates["sorel20m_ffnn"]["fake"] = fake_detected_sorel20m_ffnn_model / batch_size
            if originally_detected_malconv_model is not None:
                detection_rates["malconv"]["original"] = originally_detected_malconv_model / batch_size
                detection_rates["malconv"]["fake"] = fake_detected_malconv_model / batch_size
            if originally_detected_nonneg_malconv_model is not None:
                detection_rates["nonneg_malconv"]["original"] = originally_detected_nonneg_malconv_model / batch_size
                detection_rates["nonneg_malconv"]["fake"] = fake_detected_nonneg_malconv_model / batch_size
            if originally_detected_malconvgct_model is not None:
                detection_rates["malconvgct"]["original"] = originally_detected_malconvgct_model / batch_size
                detection_rates["malconvgct"]["fake"] = fake_detected_malconvgct_model / batch_size

            wandb.log(
                {
                    "Log: Detection rate of byte histogram 2017 on the original samples":
                        detection_rates["byte_histogram_model_2017"]["original"],
                    "Log: Detection rate of byte histogram 2017 on the fake samples": detection_rates["byte_histogram_model_2017"][
                        "fake"],
                    "Log: Detection rate of byte histogram 2018 on the original samples":
                        detection_rates["byte_histogram_model_2018"]["original"],
                    "Log: Detection rate of byte histogram  2018 on the fake samples":
                        detection_rates["byte_histogram_model_2018"][
                            "fake"],
                    "Log: Detection rate of ember 2017 on the original samples":
                        detection_rates["ember_model_2017"]["original"],
                    "Log: Detection rate of ember 2017 on the fake samples":
                        detection_rates["ember_model_2017"][
                            "fake"],
                    "Log: Detection rate of ember 2018 on the original samples":
                        detection_rates["ember_model_2018"]["original"],
                    "Log: Detection rate of ember 2018 on the fake samples":
                        detection_rates["ember_model_2018"][
                            "fake"],
                    "Log: Detection rate of SOREL-20M LigthGBM on the original samples":
                        detection_rates["sorel20m_lightgbm"]["original"],
                    "Log: Detection rate of SOREL-20M LigthGBM on the fake samples":
                        detection_rates["sorel20m_lightgbm"][
                            "fake"],
                    "Log: Detection rate of SOREL-20M FFNN on the original samples":
                        detection_rates["sorel20m_ffnn"]["original"],
                    "Log: Detection rate of SOREL-20M FFNN on the fake samples":
                        detection_rates["sorel20m_ffnn"][
                            "fake"],
                    "Log: Detection rate of MalConv on the original samples":
                        detection_rates["malconv"]["original"],
                    "Log: Detection rate of MalConv on the fake samples":
                        detection_rates["malconv"][
                            "fake"],
                    "Log: Detection rate of non-negative MalConv on the original samples":
                        detection_rates["nonneg_malconv"]["original"],
                    "Log: Detection rate of non-negative MalConv on the fake samples":
                        detection_rates["nonneg_malconv"][
                            "fake"],
                    "Log: Detection rate of MalConvGCT on the original samples":
                        detection_rates["malconvgct"]["original"],
                    "Log: Detection rate of MalConvGCT on the fake samples":
                        detection_rates["malconvgct"][
                            "fake"],
                },
            )

        return info


    def validate(self, epoch: int, batch_size: int = 32):
        benign_dataloader = self.initialize_dataloader(self.validation_discriminator_dataset, batch_size, shuffle=False, drop_last=True)
        malicious_dataloader = self.initialize_dataloader(self.validation_generator_dataset, batch_size, shuffle=False, drop_last=True)

        output_filepath = os.path.join(self.output_filepath,
                                       "validation_results/validation_{}epoch.txt".format(epoch))
        evaluation_info = self.check_evasion_rate(
            self.validation_discriminator_dataset,
            benign_dataloader,
            self.validation_generator_dataset,
            malicious_dataloader,
            output_filepath)
        self.log_to_wandb(evaluation_info, len(self.validation_generator_dataset), "Validation")

    def test(self, benign_dataset: torch.utils.data.Dataset, malicious_dataset: torch.utils.data.Dataset, batch_size: int = 32, replace: bool = True, approach: str = "exact", output_filepath: str = None, shuffle: bool = False):
        benign_dataloader = self.initialize_dataloader(benign_dataset, batch_size, shuffle=shuffle, drop_last=True)
        malicious_dataloader = self.initialize_dataloader(malicious_dataset, batch_size, shuffle=shuffle, drop_last=True)

        if output_filepath is None:
            output_filepath = os.path.join(self.output_filepath, "testing_results/testing_results.txt")
        evaluation_info = self.check_evasion_rate(
            benign_dataset,
            benign_dataloader,
            malicious_dataset,
            malicious_dataloader,
            output_filepath,
            replace=replace,
            approach=approach
        )
        self.log_to_wandb(evaluation_info, len(malicious_dataset), "Test")

    def evaluate_against_ml_models(self, evaluation_info:dict, fake_samples, g_features, g_ember_features_v1, g_ember_features_v2, g_raw_paths, g_raw_npz_paths, replace=True, approach="exact"):
        """
        Evaluate current batch against the ML detectors
        :return:
        """
        # Check samples against malware detectors
        fake_features = fake_samples.cpu().detach().numpy()

        # Convert Torch tensors to numpy arrays
        g_features = g_features.cpu().detach().numpy().astype(np.float32)
        g_ember_features_v1 = g_ember_features_v1.cpu().detach().numpy()
        g_ember_features_v2 = g_ember_features_v2.cpu().detach().numpy()

        histogram_features_batch = []
        ember_features_v1_batch = []
        ember_features_v2_batch = []
        original_bytez_batch = []
        fake_bytez_batch = []

        original_bytez_sizes = []
        fake_bytez_sizes = []

        if self.malconv_model or self.nonneg_malconv_model:
            original_bytez_arr = np.load(g_raw_npz_paths[0] + ".npz", allow_pickle=True)["arr_0"].astype(np.uint16)
            original_bytez_list = original_bytez_arr.tolist()
            original_histogram = calculate_histogram(original_bytez_list)

            if approach == "exact":
                current_histogram, current_normalized_histogram = heuristic_approach(
                    original_bytez_list,
                    g_features[0],
                    fake_features[0]
                )
            elif approach == "approximated":
                raise Exception("ToDo: approximated approach to calculate bytes")

            fake_bytez_list = append_bytes_given_target_histogram(
                original_bytez_list,
                original_histogram,
                current_histogram
            )
            print("Original #bytes: {}; Fake #bytes: {}".format(len(original_bytez_list), len(fake_bytez_list)))
            fake_bytez_arr = np.array(fake_bytez_list)

            original_bytez_length = original_bytez_arr.shape[0]
            fake_bytez_length = fake_bytez_arr.shape[0]

            b = np.ones((self.malconv_max_length,), dtype=np.uint16) * self.PADDING_CHAR  # (1048576,)
            hex_values = original_bytez_arr[:self.malconv_max_length]  # (1048576,)
            b[:len(hex_values)] = hex_values  # (1048576,)
            b = np.expand_dims(b, axis=0)
            original_bytez_batch.append(b)
            original_bytez_sizes.append(original_bytez_length)

            c = np.ones((self.malconv_max_length,), dtype=np.uint16) * self.PADDING_CHAR  # (1048576,)
            hex_values = fake_bytez_arr[:self.malconv_max_length]  # (1048576,)
            c[:len(hex_values)] = hex_values  # (1048576,)
            c = np.expand_dims(c, axis=0)
            fake_bytez_batch.append(c)
            fake_bytez_sizes.append(fake_bytez_length)

            evaluation_info["bytez_size"]["original"].append(original_bytez_length)
            evaluation_info["bytez_size"]["fake"].append(fake_bytez_length)

        #if len(bytez_int_list) < 100663296:
        # Histogram features
        try:
            histogram_features_batch.append(current_normalized_histogram)
        except UnboundLocalError:
            histogram_features_batch.append(np.squeeze(fake_features))

        if replace == True:
            try:
                ember_features_v1_batch.append(
                    self.ember_feature_extractor_v1.replace_byte_histogram_features(np.squeeze(g_ember_features_v1),
                                                                                    current_normalized_histogram))
                ember_features_v2_batch.append(
                    self.ember_feature_extractor_v2.replace_byte_histogram_features(np.squeeze(g_ember_features_v2),
                                                                                    current_normalized_histogram))
            except UnboundLocalError:
                ember_features_v1_batch.append(
                    self.ember_feature_extractor_v1.replace_byte_histogram_features(np.squeeze(g_ember_features_v1),
                                                                                    np.squeeze(fake_features)))
                ember_features_v2_batch.append(
                    self.ember_feature_extractor_v2.replace_byte_histogram_features(np.squeeze(g_ember_features_v2),
                                                                                    np.squeeze(fake_features)))
        else:
            pe_modifier = PEModifier(g_raw_paths[0])
            binary = pe_modifier._get_binary(fake_bytez_list)
            bytez, bytez_int_list, binary = pe_modifier._binary_to_bytez(binary)
            # Extract EMBER features (v1)
            ember_features_v1_batch.append(self.ember_feature_extractor_v1.feature_vector(bytez))
            # Extract EMBER features (v2)
            ember_features_v2_batch.append(self.ember_feature_extractor_v2.feature_vector(bytez))


        histogram_features_batch = np.array(histogram_features_batch).astype(np.float32)
        if self.ember_model_2017:
            ember_features_v1_batch = np.array(ember_features_v1_batch).astype(np.float32)
        if self.ember_model_2018 or self.sorel20m_lightgbm_model or self.sorel20m_ffnn_model:
            ember_features_v2_batch = np.array(ember_features_v2_batch).astype(np.float32)
        if self.malconv_model or self.nonneg_malconv_model or self.malconvgct_model:
            original_bytez_batch = np.array(original_bytez_batch)#.astype(np.float32)
            fake_bytez_batch = np.array(fake_bytez_batch)#.astype(np.float32)
            original_bytez_batch = np.squeeze(original_bytez_batch, axis=1)
            fake_bytez_batch = np.squeeze(fake_bytez_batch, axis=1)

        fake_histogram_features = torch.from_numpy(histogram_features_batch)
        act1 = self.inception_model.retrieve_features(fake_histogram_features)
        act1 = act1.cpu().detach().numpy()
        evaluation_info["intermediate_features"]["fake"].append(act1)

        if self.features_model_2017 is not None:
            original_detections_histogram_model_2017 = self.evaluate_on_histogram_model_2017(g_features)
            fake_detections_histogram_model_2017 = self.evaluate_on_histogram_model_2017(
            histogram_features_batch)
            evaluation_info["detections"]["original"]["byte_histogram_model_2017"] += original_detections_histogram_model_2017
            evaluation_info["detections"]["fake"]["byte_histogram_model_2017"] += fake_detections_histogram_model_2017
        else:
            evaluation_info["detections"]["original"][
                "byte_histogram_model_2017"] += 0
            evaluation_info["detections"]["fake"]["byte_histogram_model_2017"] += 0

        if self.features_model_2018:
            original_detections_histogram_model_2018 = self.evaluate_on_histogram_model_2018(g_features)
            fake_detections_histogram_model_2018 = self.evaluate_on_histogram_model_2018(
                histogram_features_batch)
            evaluation_info["detections"]["original"]["byte_histogram_model_2018"] += original_detections_histogram_model_2018
            evaluation_info["detections"]["fake"]["byte_histogram_model_2018"] += fake_detections_histogram_model_2018
        else:
            evaluation_info["detections"]["original"][
                "byte_histogram_model_2018"] += 0
            evaluation_info["detections"]["fake"]["byte_histogram_model_2018"] += 0

        if self.ember_model_2017:
            original_detections_ember_model_2017 = self.evaluate_on_ember_model_2017(g_ember_features_v1)
            fake_detections_ember_model_2017 = self.evaluate_on_ember_model_2017(ember_features_v1_batch)
            evaluation_info["detections"]["original"]["ember_model_2017"] += original_detections_ember_model_2017
            evaluation_info["detections"]["fake"]["ember_model_2017"] += fake_detections_ember_model_2017
        else:
            evaluation_info["detections"]["original"]["ember_model_2017"] += 0
            evaluation_info["detections"]["fake"]["ember_model_2017"] += 0

        if self.ember_model_2018:
            original_detections_ember_model_2018 = self.evaluate_on_ember_model_2018(g_ember_features_v2)
            fake_detections_ember_model_2018 = self.evaluate_on_ember_model_2018(ember_features_v2_batch)
            evaluation_info["detections"]["original"]["ember_model_2018"] += original_detections_ember_model_2018
            evaluation_info["detections"]["fake"]["ember_model_2018"] += fake_detections_ember_model_2018
        else:
            evaluation_info["detections"]["original"]["ember_model_2018"] += 0
            evaluation_info["detections"]["fake"]["ember_model_2018"] += 0

        if self.sorel20m_lightgbm_model:
            original_detections_sorel20m_lightgbm_model = self.evaluate_on_sorel20m_lightgbm_model(
                g_ember_features_v2)
            fake_detections_sorel20m_lightgbm_model = self.evaluate_on_sorel20m_lightgbm_model(
                ember_features_v2_batch)
            evaluation_info["detections"]["original"][
                "sorel20m_lightgbm_model"] += original_detections_sorel20m_lightgbm_model
            evaluation_info["detections"]["fake"][
                "sorel20m_lightgbm_model"] += fake_detections_sorel20m_lightgbm_model
        else:
            evaluation_info["detections"]["original"][
                "sorel20m_lightgbm_model"] += 0
            evaluation_info["detections"]["fake"][
                "sorel20m_lightgbm_model"] += 0

        if self.sorel20m_ffnn_model:
            original_detections_sorel20m_ffnn_model = self.evaluate_on_sorel20m_ffnn_model(g_ember_features_v2)
            fake_detections_sorel20m_ffnn_model = self.evaluate_on_sorel20m_ffnn_model(ember_features_v2_batch)
            evaluation_info["detections"]["original"][
                "sorel20m_ffnn_model"] += original_detections_sorel20m_ffnn_model
            evaluation_info["detections"]["fake"]["sorel20m_ffnn_model"] += fake_detections_sorel20m_ffnn_model
        else:
            evaluation_info["detections"]["original"][
                "sorel20m_ffnn_model"] += 0
            evaluation_info["detections"]["fake"]["sorel20m_ffnn_model"] += 0

        if self.malconv_model:
            original_detections_malconv_model = self.evaluate_on_malconv_model(original_bytez_batch)
            fake_detections_malconv_model = self.evaluate_on_malconv_model(fake_bytez_batch)
            evaluation_info["detections"]["original"]["malconv_model"] += original_detections_malconv_model
            evaluation_info["detections"]["fake"]["malconv_model"] += fake_detections_malconv_model
        else:
            evaluation_info["detections"]["original"]["malconv_model"] += 0
            evaluation_info["detections"]["fake"]["malconv_model"] += 0

        if self.nonneg_malconv_model:
            original_detections_nonneg_malconv_model = self.evaluate_on_nonnegative_malconv_model(original_bytez_batch)
            fake_detections_nonneg_malconv_model = self.evaluate_on_nonnegative_malconv_model(fake_bytez_batch)
            evaluation_info["detections"]["original"]["nonneg_malconv_model"] += original_detections_nonneg_malconv_model
            evaluation_info["detections"]["fake"]["nonneg_malconv_model"] += fake_detections_nonneg_malconv_model
        else:
            evaluation_info["detections"]["original"][
                "nonneg_malconv_model"] += 0
            evaluation_info["detections"]["fake"]["nonneg_malconv_model"] += 0

        if self.malconvgct_model:
            original_detections_malconvgct_model = self.evaluate_on_malconvgct_model(original_bytez_batch)
            fake_detections_malconvgct_model = self.evaluate_on_malconvgct_model(fake_bytez_batch)
            evaluation_info["detections"]["original"]["malconvgct_model"] += original_detections_malconvgct_model
            evaluation_info["detections"]["fake"]["malconvgct_model"] += fake_detections_malconvgct_model
        else:
            evaluation_info["detections"]["original"]["malconvgct_model"] += 0
            evaluation_info["detections"]["fake"]["malconvgct_model"] += 0
        return evaluation_info

    def check_evasion_rate(self, benign_dataset: torch.utils.data.Dataset, benign_dataloader: DataLoader, malicious_dataset: torch.utils.data.Dataset, malicious_dataloader: DataLoader, output_filepath:str, replace:bool = True, approach:str = "exact"):
        self.generator.eval()
        self.discriminator.eval()

        evaluation_info = self.initialize_evaluation_info_dictionary()

        fake_features_list = []
        original_features_list = []

        j = 0
        for (g_features, g_ember_features_v1, g_ember_features_v2, g_raw_paths, g_raw_npz_paths, g_y) in malicious_dataloader:
            print("{}; Current samples: {}".format(j, g_raw_paths))
            self.logger.info("Current samples: {}".format(g_raw_paths))
            g_features = g_features.to(self.device)
            # Create fake feature vector
            noise = torch.randn(g_features.shape[0], self.generator_parameters["z_size"]).to(
                self.device)
            fake_samples = self.generator([g_features, noise])

            act1 = self.inception_model.retrieve_features(fake_samples.cpu().detach())
            act1 = act1.cpu().detach().numpy()
            evaluation_info["intermediate_features"]["fake"].append(act1)

            fake_features = fake_samples.cpu().detach().numpy()

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

        original_features_arr = np.array(original_features_list)
        original_features_arr = np.squeeze(original_features_arr, axis=1)
        fake_features_arr = np.array(fake_features_list)
        fake_features_arr = np.squeeze(fake_features_arr, axis=1)

        evaluation_info["distance_metrics"]["jensenshannon"]["between_original_and_fake"].append(
            jensen_shannon_distance_np(original_features_arr, fake_features_arr))
        evaluation_info["distance_metrics"]["jensenshannon"]["between_fake"].append(
            pdist(fake_features_arr, metric="jensenshannon"))

        #evaluation_info["distance_metrics"] = self.calculate_distance_between_fake_samples(fake_features_arr,
        #                                                                                   evaluation_info[
        #                                                                                       "distance_metrics"])
        #evaluation_info["distance_metrics"] = self.calculate_distance_between_original_and_fake_samples(original_features_arr,
        #                                                                                                fake_features_arr,
        #                                                                                                evaluation_info[
        #                                                                                                    "distance_metrics"])

        fid_score = calculate_fid(np.concatenate(evaluation_info["intermediate_features"]["fake"]), self.benign_intermediate_features)
        evaluation_info["fid"] = fid_score
        self.print_info(evaluation_info, len(malicious_dataset))
        self.log_info(evaluation_info, len(malicious_dataset))
        self.write_to_file(evaluation_info, output_filepath, len(malicious_dataset))

        self.generator.train()
        self.discriminator.train()
        return evaluation_info

    def save_models(self, info:dict, evasion_rates:dict):
        if self.features_model_2017 is not None:
            info["evasion_rate"]["list"]["byte_histogram_model_2017"].append(evasion_rates["byte_histogram_model_2017"])
            if evasion_rates["byte_histogram_model_2017"] < info["evasion_rate"]["best"]["byte_histogram_model_2017"]:
                info["evasion_rate"]["best"]["byte_histogram_model_2017"] = evasion_rates["byte_histogram_model_2017"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_features_model_2017.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_byte_histogram_model_2017"] = evasion_rates["byte_histogram_model_2017"]

        if self.features_model_2018 is not None:
            info["evasion_rate"]["list"]["byte_histogram_model_2018"].append(evasion_rates["byte_histogram_model_2018"])
            if evasion_rates["byte_histogram_model_2018"] < info["evasion_rate"]["best"]["byte_histogram_model_2018"]:
                info["evasion_rate"]["best"]["byte_histogram_model_2018"] = evasion_rates["byte_histogram_model_2018"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_features_model_2018.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_byte_histogram_model_2018"] = evasion_rates["byte_histogram_model_2018"]

        if self.ember_model_2017 is not None:
            info["evasion_rate"]["list"]["ember_model_2017"].append(evasion_rates["ember_model_2017"])
            if evasion_rates["ember_model_2017"] < info["evasion_rate"]["best"]["ember_model_2017"]:
                info["evasion_rate"]["best"]["ember_model_2017"] = evasion_rates["ember_model_2017"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_ember_model_2017.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_ember_model_2017"] = evasion_rates["ember_model_2017"]

        if self.ember_model_2018 is not None:
            info["evasion_rate"]["list"]["ember_model_2018"].append(evasion_rates["ember_model_2018"])
            if evasion_rates["ember_model_2018"] < info["evasion_rate"]["best"]["ember_model_2018"]:
                info["evasion_rate"]["best"]["ember_model_2018"] = evasion_rates["ember_model_2018"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_ember_model_2018.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_ember_model_2018"] = evasion_rates["ember_model_2018"]

        if self.sorel20m_lightgbm_model is not None:
            info["evasion_rate"]["list"]["sorel20m_lightgbm_model"].append(evasion_rates["sorel20m_lightgbm_model"])
            if evasion_rates["sorel20m_lightgbm_model"] < info["evasion_rate"]["best"]["sorel20m_lightgbm_model"]:
                info["evasion_rate"]["best"]["sorel20m_lightgbm_model"] = evasion_rates["sorel20m_lightgbm_model"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_sorel20m_lightgbm_model.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_sorel20m_lightgbm_model"] = evasion_rates["sorel20m_lightgbm_model"]

        if self.sorel20m_ffnn_model is not None:
            info["evasion_rate"]["list"]["sorel20m_ffnn_model"].append(evasion_rates["sorel20m_ffnn_model"])
            if evasion_rates["sorel20m_ffnn_model"] < info["evasion_rate"]["best"]["sorel20m_ffnn_model"]:
                info["evasion_rate"]["best"]["sorel20m_ffnn_model"] = evasion_rates["sorel20m_ffnn_model"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_sorel20m_ffnn_model.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_sorel20m_ffnn_model"] = evasion_rates["sorel20m_ffnn_model"]

        if self.malconv_model is not None:
            info["evasion_rate"]["list"]["malconv_model"].append(evasion_rates["malconv_model"])
            if evasion_rates["malconv_model"] < info["evasion_rate"]["best"]["malconv_model"]:
                info["evasion_rate"]["best"]["malconv_model"] = evasion_rates["malconv_model"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_malconv_model.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_malconv_model"] = evasion_rates["malconv_model"]

        if self.nonneg_malconv_model is not None:
            info["evasion_rate"]["list"]["nonnegative_malconv_model"].append(evasion_rates["nonnegative_malconv_model"])
            if evasion_rates["nonnegative_malconv_model"] < info["evasion_rate"]["best"]["nonnegative_malconv_model"]:
                info["evasion_rate"]["best"]["nonnegative_malconv_model"] = evasion_rates["nonnegative_malconv_model"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_nonnegative_malconv_model.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_nonnegative_malconv_model"] = evasion_rates["nonnegative_malconv_model"]

        if self.malconvgct_model is not None:
            info["evasion_rate"]["list"]["malconvgct_model"].append(evasion_rates["malconvgct_model"])
            if evasion_rates["malconvgct_model"] < info["evasion_rate"]["best"]["malconvgct_model"]:
                info["evasion_rate"]["best"]["malconvgct_model"] = evasion_rates["malconvgct_model"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_malconvgct_model.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_malconvgct_model"] = evasion_rates["malconvgct_model"]

        return info


    def log_to_wandb(self, evaluation_info, dataset_size, mode):
        """
        Log the evaluation information to Weights&Biases

        :param evaluation_info:
        :param dataset_size:
        :param mode:
        :return:
        """
        if self.is_wandb == True:
            wandb.log(
                {
                    "{}: Detection rate of byte histogram 2017 on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["byte_histogram_model_2017"]/dataset_size,
                    "{}: Detection rate of byte histogram 2017 on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["byte_histogram_model_2017"] / dataset_size,
                    "{}: Detection rate of byte histogram 2018 on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["byte_histogram_model_2018"] / dataset_size,
                    "{}: Detection rate of byte histogram 2018 on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["byte_histogram_model_2018"] / dataset_size,

                    "{}: Detection rate of EMBER 2017 on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["ember_model_2017"] / dataset_size,
                    "{}: Detection rate of EMBER 2017 on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["ember_model_2017"] / dataset_size,
                    "{}: Detection rate of EMBER 2018 on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["ember_model_2018"] / dataset_size,
                    "{}: Detection rate of EMBER 2018 on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["ember_model_2018"] / dataset_size,

                    "{}: Detection rate of SOREL-20M LightGBM on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["sorel20m_lightgbm_model"] / dataset_size,
                    "{}: Detection rate of SOREL-20M LightGBM on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["sorel20m_lightgbm_model"] / dataset_size,
                    "{}: Detection rate of SOREL-20M FFNN on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["sorel20m_ffnn_model"] / dataset_size,
                    "{}: Detection rate of SOREL-20M FFNN on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["sorel20m_ffnn_model"] / dataset_size,

                    "{}: Detection rate of MalConv on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["malconv_model"] / dataset_size,
                    "{}: Detection rate of MalConv on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["malconv_model"] / dataset_size,
                    "{}: Detection rate of Non-negative MalConv on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["nonneg_malconv_model"] / dataset_size,
                    "{}: Detection rate of Non-negative MalConv on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["nonneg_malconv_model"] / dataset_size,
                    "{}: Detection rate of MalConvGCT on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["malconvgct_model"] / dataset_size,
                    "{}: Detection rate of MalConvGCT on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["malconvgct_model"] / dataset_size,
                }
            )

            wandb.log(
                {
                    "{}: FID score".format(mode): evaluation_info["fid"]
                }
            )

            for metric in evaluation_info["distance_metrics"]:
                wandb.log(
                    {
                        "{}: {} distance between original and fake samples".format(mode, metric): sum(
                            evaluation_info["distance_metrics"][metric]["between_original_and_fake"]) / len(
                            evaluation_info["distance_metrics"][metric]["between_original_and_fake"]),
                        "{}: {} distance between fake samples".format(mode, metric): sum(
                            evaluation_info["distance_metrics"][metric]["between_fake"]) / len(
                            evaluation_info["distance_metrics"][metric]["between_fake"])
                    }
                )
                try:
                    wandb.log(
                        {
                            "{}: {} distance between fake and benign samples".format(mode, metric): sum(
                            evaluation_info["distance_metrics"][metric]["between_fake_and_benign"]) / len(
                            evaluation_info["distance_metrics"][metric]["between_fake_and_benign"])
                        }
                    )
                except Exception:
                    pass


            try:
                wandb.log(
                    {
                        "{}: Average size of original executables".format(mode): sum(evaluation_info["bytez_size"]["original"])/len(evaluation_info["bytez_size"]["original"]),
                        "{}: Average size of fake executables".format(mode): sum(evaluation_info["bytez_size"]["fake"])/len(evaluation_info["bytez_size"]["fake"])
                    }
                )
            except ZeroDivisionError:
                pass


