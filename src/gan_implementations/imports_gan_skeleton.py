import copy

import torch
from src.gan_implementations.gan_skeleton import SkeletonGAN
from src.feature_extractors.imports_info_extractor import ImportsInfoExtractor
from src.gan_implementations.imports_dataset import ImportsDataset
from src.ml_models.torch_models.inception_detector.iat_net import IATNetwork
from src.pe_modifier import PEModifier
from src.gan_implementations.utils import plot_distance_metric, plot_average_collisions, plot_average_imported_functions, plot_evasion_rate, load_json, check_collisions_among_generated_samples, plot_generator_and_discriminator_training_loss, plot_generator_and_discriminator_validation_loss, plot_frechlet_inception_distance
from src.gan_implementations.metrics.frechlet_inception_distance import calculate_fid
from scipy.spatial.distance import pdist
from src.gan_implementations.metrics.euclidean_distance import euclidean_distance_np
from src.gan_implementations.metrics.minkowski_distance import minkowski_distance_np
from src.gan_implementations.metrics.hamming_distance import hamming_distance_np
import json
import os
import sys
from abc import abstractmethod
import numpy as np
from torch.utils.data import DataLoader
import wandb

class ImportsGAN(SkeletonGAN):

    def build_blackbox_detectors(
            self,
            features_model_2017_filepath: str,
            is_features_model_2017: bool,
            features_model_2018_filepath: str,
            is_features_model_2018: bool,
            hashed_features_model_2017_filepath: str,
            is_hashed_features_model_2017: bool,
            hashed_features_model_2018_filepath: str,
            is_hashed_features_model_2018: bool,
            ember_model_2017_filepath: str,
            is_ember_model_2017: bool,
            ember_model_2018_filepath: str,
            is_ember_model_2018: bool,
            sorel20m_lightgbm_model_filepath: str,
            is_sorel20m_lightgbm_model: bool,
            sorel20m_ffnn_model_filepath: str,
            is_sorel20m_ffnn_model: bool,
            malconv_model_filepath: str,
            is_malconv_model: bool,
            nonneg_malconv_model_filepath: str,
            is_nonneg_malconv_model: bool,
            malconvgct_model_filepath: str,
            is_malconvgct_model: bool
    ):
        self.features_model_2017 = None
        self.features_model_2018 = None
        self.hashed_features_model_2017 = None
        self.hashed_features_model_2018 = None
        self.ember_model_2017 = None
        self.ember_model_2018 = None
        self.sorel20m_lightgbm_model = None
        self.sorel20m_ffnn_model = None
        self.malconv_model = None
        self.nonneg_malconv_model = None
        self.malconvgct_model = None

        if is_features_model_2017:
            self.features_model_2017 = self.build_lightgbm_blackbox_detector(
                features_model_2017_filepath)
        if is_features_model_2018:
            self.features_model_2018 = self.build_lightgbm_blackbox_detector(
                features_model_2018_filepath)
        if is_hashed_features_model_2017:
            self.hashed_features_model_2017 = self.build_lightgbm_blackbox_detector(
                hashed_features_model_2017_filepath)
        if is_hashed_features_model_2018:
            self.hashed_features_model_2018 = self.build_lightgbm_blackbox_detector(
                hashed_features_model_2018_filepath)
        if is_ember_model_2017:
            self.ember_model_2017 = self.build_lightgbm_blackbox_detector(ember_model_2017_filepath)
        if is_ember_model_2018:
            self.ember_model_2018 = self.build_lightgbm_blackbox_detector(ember_model_2018_filepath)
        if is_sorel20m_lightgbm_model:
            self.sorel20m_lightgbm_model = self.build_lightgbm_blackbox_detector(sorel20m_lightgbm_model_filepath)
        if is_sorel20m_ffnn_model:
            self.sorel20m_ffnn_model = self.build_sorel20m_ffnn_blackbox_detector(sorel20m_ffnn_model_filepath)
        if is_malconv_model:
            self.malconv_model = self.build_malconv_blackbox_detector(malconv_model_filepath)
            _, self.malconv_max_length, self.malconv_embedding_size = self.malconv_model.layers[1].output_shape
            self.PADDING_CHAR = 256
            self.NUM_DIFFERENT_CHARS=257
        if is_nonneg_malconv_model:
            self.nonneg_malconv_model = self.build_nonnegative_malconv_detector(nonneg_malconv_model_filepath)
        if is_malconvgct_model:
            self.malconvgct_model = self.build_malconvgct_detector(malconvgct_model_filepath)

    def initialize_vocabulary_mapping(self, vocabulary_mapping_filepath):
        with open(vocabulary_mapping_filepath, "r") as input_file:
            self.vocabulary_mapping = json.load(input_file)

    def initialize_inverse_vocabulary_mapping(self, inverse_vocabulary_mapping_filepath):
        with open(inverse_vocabulary_mapping_filepath, "r") as input_file:
            self.inverse_vocabulary_mapping = json.load(input_file)

    def initialize_feature_extractor(self):
        self.feature_extractor = ImportsInfoExtractor(self.vocabulary_mapping, self.inverse_vocabulary_mapping)

    def initialize_training_and_validation_discriminator_datasets(
            self,
            goodware_imports_features_filepath,
            goodware_hashed_imports_features_filepath,
            goodware_imports_filepath,
            goodware_ember_features_filepath_version1,
            goodware_ember_features_filepath_version2,
            goodware_raw_executables_filepath,
            training_goodware_annotations_filepath,
            validation_goodware_annotations_filepath
    ):
        """
        Intializes  the discriminator's training and validation datasets

        :param goodware_imports_features_filepath:
        :param goodware_hashed_imports_features_filepath:
        :param goodware_imports_filepath:
        :param goodware_ember_features_filepath_version1:
        :param goodware_ember_features_filepath_version2:
        :param goodware_raw_executables_filepath:
        :param training_goodware_annotations_filepath:
        :param validation_goodware_annotations_filepath:
        :return:
        """
        # Create discriminator's training and validation dataset
        self.training_discriminator_dataset = ImportsDataset(
            goodware_imports_features_filepath,
            goodware_hashed_imports_features_filepath,
            goodware_imports_filepath,
            goodware_ember_features_filepath_version1,
            goodware_ember_features_filepath_version2,
            goodware_raw_executables_filepath,
            training_goodware_annotations_filepath
        )
        self.validation_discriminator_dataset = ImportsDataset(
            goodware_imports_features_filepath,
            goodware_hashed_imports_features_filepath,
            goodware_imports_filepath,
            goodware_ember_features_filepath_version1,
            goodware_ember_features_filepath_version2,
            goodware_raw_executables_filepath,
            validation_goodware_annotations_filepath
        )

    def initialize_training_and_validation_generator_datasets(
            self,
            malware_imports_features_filepath,
            malware_hashed_imports_features_filepath,
            malware_imports_filepath,
            malware_ember_features_filepath_version1,
            malware_ember_features_filepath_version2,
            malware_raw_executables_filepath,
            training_malware_annotations_filepath,
            validation_malware_annotations_filepath
    ):
        """
        Intializes the generator's training and validation datasets

        :param malware_imports_features_filepath:
        :param malware_hashed_imports_features_filepath:
        :param malware_imports_filepath:
        :param malware_ember_features_filepath_version1:
        :param malware_ember_features_filepath_version2:
        :param malware_raw_executables_filepath:
        :param training_malware_annotations_filepath:
        :param validation_malware_annotations_filepath:
        :return:
        """
        self.training_generator_dataset = ImportsDataset(
            malware_imports_features_filepath,
            malware_hashed_imports_features_filepath,
            malware_imports_filepath,
            malware_ember_features_filepath_version1,
            malware_ember_features_filepath_version2,
            malware_raw_executables_filepath,
            training_malware_annotations_filepath
        )
        self.validation_generator_dataset = ImportsDataset(
            malware_imports_features_filepath,
            malware_hashed_imports_features_filepath,
            malware_imports_filepath,
            malware_ember_features_filepath_version1,
            malware_ember_features_filepath_version2,
            malware_raw_executables_filepath,
            validation_malware_annotations_filepath
        )

    def load_inception_model(self, inception_parameters, inception_checkpoint):
        self.inception_model = IATNetwork(inception_parameters)
        self.inception_model.load_state_dict(torch.load(inception_checkpoint, map_location=torch.device("cpu")))
        self.inception_model.eval()
        #self.inception_model.to(torch.device("cpu"))
        return self.inception_model

    def load_inception_benign_intermediate_features(self, inception_features_filepath:str):
        self.benign_intermediate_features = np.load(inception_features_filepath, allow_pickle=True)["arr_0"]
        return self.benign_intermediate_features

    def evaluate_on_model_2017(self, features):
        if self.features_model_2017 is not None:
            model_output = self.features_model_2017.predict(features)
            detections = model_output.round().astype(int)
            detected = np.sum(detections)
            return detected
        return None

    def evaluate_on_model_2018(self, features):
        if self.features_model_2018 is not None:
            model_output = self.features_model_2018.predict(features)
            detections = model_output.round().astype(int)
            detected = np.sum(detections)
            return detected
        return None

    def evaluate_on_hashed_model_2017(self, hashed_features):
        if self.hashed_features_model_2017 is not None:
            model_output = self.hashed_features_model_2017.predict(hashed_features)
            detections = model_output.round().astype(int)
            detected = np.sum(detections)
            return detected
        return None

    def evaluate_on_hashed_model_2018(self, hashed_features):
        if self.hashed_features_model_2018 is not None:
            model_output = self.hashed_features_model_2018.predict(hashed_features)
            detections = model_output.round().astype(int)
            detected = np.sum(detections)
            return detected
        return None

    @abstractmethod
    def run_training_epoch(self, criterions:dict, optimizers:dict, info:dict, training_parameters:dict, epoch:int):
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
                    "model_2017": 1.0,
                    "model_2018": 1.0,
                    "hashed_model_2017": 1.0,
                    "hashed_model_2018": 1.0,
                    "ember_model_2017": 1.0,
                    "ember_model_2018": 1.0,
                    "sorel20m_lightgbm_model": 1.0,
                    "sorel20m_ffnn_model": 1.0,
                    "malconv_model": 1.0,
                    "nonnegative_malconv_model": 1.0
                },
                "list":{
                    "model_2017": [],
                    "model_2018": [],
                    "hashed_model_2017": [],
                    "hashed_model_2018": [],
                    "ember_model_2017": [],
                    "ember_model_2018": [],
                    "sorel20m_lightgbm_model": [],
                    "sorel20m_ffnn_model": [],
                    "malconv_model": [],
                    "nonnegative_malconv_model": []
                }
            },
            "fid":{
                "best":sys.maxsize,
                "scores": []
            },
            "imported_functions": {
                "original": [],
                "fake": [],
            },
            "collisions": [],
            "distance_metrics":{
                "train":{
                    "euclidean": {
                        "between_original_and_fake":[],
                        "between_fake":[]
                    },
                    "minkowski": {
                        "between_original_and_fake":[],
                        "between_fake":[]
                    },
                    "hamming": {
                        "between_original_and_fake":[],
                        "between_fake":[]
                    },
                },
                "validation":{
                    "euclidean": {
                        "between_original_and_fake": [],
                        "between_fake": []
                    },
                    "minkowski": {
                        "between_original_and_fake": [],
                        "between_fake": []
                    },
                    "hamming": {
                        "between_original_and_fake": [],
                        "between_fake": []
                    },
                }
            }
        }

    def initialize_evaluation_info_dictionary(self):
        return {
            "detections": {
                "original": {
                    "model_2017": 0,
                    "model_2018": 0,
                    "hashed_model_2017": 0,
                    "hashed_model_2018": 0,
                    "ember_model_2017": 0,
                    "ember_model_2018": 0,
                    "sorel20m_lightgbm_model": 0,
                    "sorel20m_ffnn_model": 0,
                    "malconv_model": 0,
                    "nonneg_malconv_model": 0
                },
                "fake": {
                    "model_2017": 0,
                    "model_2018": 0,
                    "hashed_model_2017": 0,
                    "hashed_model_2018": 0,
                    "ember_model_2017": 0,
                    "ember_model_2018": 0,
                    "sorel20m_lightgbm_model": 0,
                    "sorel20m_ffnn_model": 0,
                    "malconv_model": 0,
                    "nonneg_malconv_model": 0
                }
            },
            "imported_functions": {
                "original": [],
                "fake": [],
            },
            "collisions": [],
            "intermediate_features": {
                "benign": [],
                "fake": []
            },
            "fid":sys.maxsize,
            "distance_metrics": {
                "euclidean": {
                    "between_original_and_fake":[],
                    "between_fake":[]
                },
                "minkowski": {
                    "between_original_and_fake":[],
                    "between_fake":[]
                },
                "hamming": {
                    "between_original_and_fake":[],
                    "between_fake":[]
                },
            },
        }

    def print_info(self, info:dict, size:int):
        print("Model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["model_2017"], size,
            info["detections"]["fake"]["model_2017"], size))
        print("Model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["model_2018"], size,
            info["detections"]["fake"]["model_2018"], size))
        print("Hashed model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["hashed_model_2017"], size,
            info["detections"]["fake"]["hashed_model_2017"], size))
        print("Hashed model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["hashed_model_2018"], size,
            info["detections"]["fake"]["hashed_model_2018"], size))
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
        print("Original average imported functions: {}\n".format(
            np.sum(info["imported_functions"]["original"]) / len(info["imported_functions"]["original"])))
        print("Fake average imported functions: {}\n".format(
            np.sum(info["imported_functions"]["fake"]) / len(info["imported_functions"]["fake"])))
        # Number of collisions
        print("Average number of collisions {}\n".format(np.sum(info["collisions"]) / len(info["collisions"])))
        try:
            print("Frechlet Inception Distance: {}".format(info["fid"]))
            for metric in info["distance_metrics"]:
                print("{} distance between original and fake samples: {}".format(metric, sum(info["distance_metrics"][metric]["between_original_and_fake"])/len(info["distance_metrics"][metric]["between_original_and_fake"])))
                #print("{} distance between fake samples: {}".format(metric, sum(info["distance_metrics"][metric]["between_fake"])/len(info["distance_metrics"][metric]["between_fake"])))
        except ZeroDivisionError:
            pass
        except ValueError:
            pass

    def log_info(self, info:dict, size:int):
        self.logger.info("\n\nEvaluation results:")
        self.logger.info("Model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["model_2017"], size,
            info["detections"]["fake"]["model_2017"], size))
        self.logger.info("Model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["model_2018"], size,
            info["detections"]["fake"]["model_2018"], size))
        self.logger.info("Hashed model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["hashed_model_2017"], size,
            info["detections"]["fake"]["hashed_model_2017"], size))
        self.logger.info("Hashed model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            info["detections"]["original"]["hashed_model_2018"], size,
            info["detections"]["fake"]["hashed_model_2018"], size))
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
        self.logger.info("Original average imported functions: {}\n".format(
            np.sum(info["imported_functions"]["original"]) / len(info["imported_functions"]["original"])))
        self.logger.info("Fake average imported functions: {}\n".format(
            np.sum(info["imported_functions"]["fake"]) / len(info["imported_functions"]["fake"])))
        # Number of collisions
        self.logger.info("Average number of collisions {}\n".format(np.sum(info["collisions"]) / len(info["collisions"])))
        self.logger.info("Frechlet Inception Distance: {}".format(info["fid"]))
        try:
            for metric in info["distance_metrics"]:
                self.logger.info("{} distance between original and fake samples: {}".format(metric, sum(info["distance_metrics"][metric]["between_original_and_fake"])/len(info["distance_metrics"][metric]["between_original_and_fake"])))
                #self.logger.info("{} distance between fake samples: {}".format(metric, sum(info["distance_metrics"][metric]["between_fake"])/len(info["distance_metrics"][metric]["between_fake"])))
        except ZeroDivisionError:
            pass

    def write_to_file(self, info:dict, output_filepath:str, size:int):
        with open(output_filepath, "w") as output_file:
            # Evasion rates
            output_file.write("Results\n\n")
            output_file.write(
                "Model 2017; Originally detected: {}/{}; Fake detected: {}/{}\n".format(
                    info["detections"]["original"]["model_2017"], size,
                    info["detections"]["fake"]["model_2017"], size))
            output_file.write(
                "Model 2018; Originally detected: {}/{}; Fake detected: {}/{}\n".format(
                    info["detections"]["original"]["model_2018"], size,
                    info["detections"]["fake"]["model_2018"], size))
            output_file.write(
                "Hashed model 2017; Originally detected: {}/{}; Fake detected: {}/{}\n".format(
                    info["detections"]["original"]["hashed_model_2017"], size,
                    info["detections"]["fake"]["hashed_model_2017"], size))
            output_file.write(
                "Hashed model 2018; Originally detected: {}/{}; Fake detected: {}/{}\n".format(
                    info["detections"]["original"]["hashed_model_2018"], size,
                    info["detections"]["fake"]["hashed_model_2018"], size))
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

            # Number of imported functions
            output_file.write("Original average imported functions: {}\n".format(
                np.sum(info["imported_functions"]["original"]) / len(info["imported_functions"]["original"])))
            output_file.write("Fake average imported functions: {}\n".format(
                np.sum(info["imported_functions"]["fake"]) / len(info["imported_functions"]["fake"])))
            # Number of collisions
            output_file.write(
                "Average number of collisions {}\n".format(
                    np.sum(info["collisions"]) / len(info["collisions"])))
            output_file.write("FID score: {}\n".format(info["fid"]))
            try:
                for metric in info["distance_metrics"]:
                    output_file.write("{} distance between original and fake samples: {}\n".format(metric, sum(
                        info["distance_metrics"][metric]["between_original_and_fake"]) / len(
                        info["distance_metrics"][metric]["between_original_and_fake"])))
                    #output_file.write("{} distance between fake samples: {}\n".format(metric, sum(
                    #    info["distance_metrics"][metric]["between_fake"]) / len(
                    #    info["distance_metrics"][metric]["between_fake"])))
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
                self.validate(self.validation_generator_dataset, epoch, training_parameters["batch_size"])

            if epoch % training_parameters["save_interval"] == 0:
                torch.save(self.generator,
                           os.path.join(self.output_filepath, "generator/generator_checkpoint_{}.pt".format(epoch)))
                torch.save(self.discriminator, os.path.join(self.output_filepath,
                                                            "discriminator/discriminator_checkpoint_{}.pt".format(
                                                                epoch)))
        # Save generator and discriminator models (last epoch)
        torch.save(self.generator, os.path.join(self.output_filepath, "generator/generator.pt"))
        torch.save(self.discriminator, os.path.join(self.output_filepath, "discriminator/discriminator.pt"))
        self.plot_results(info)



    def plot_results(self, info:dict):
        # Plot evasion rates during evaluations
        plot_evasion_rate(info["evasion_rate"]["list"]["model_2017"], "Model 2017",
                          os.path.join(self.output_filepath, "plots/evasion_rate_model_2017.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["model_2018"], "Model 2018",
                          os.path.join(self.output_filepath, "plots/evasion_rate_model_2018.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["hashed_model_2017"], "Hashed model 2017",
                          os.path.join(self.output_filepath, "plots/evasion_rate_hashed_model_2017.png"))
        plot_evasion_rate(info["evasion_rate"]["list"]["hashed_model_2018"], "Hashed model 2018",
                          os.path.join(self.output_filepath, "plots/evasion_rate_hashed_model_2018.png"))
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

        with open(os.path.join(self.output_filepath, "original_average_imported_functions.csv"), "w") as output_file:
            for x in info["imported_functions"]["original"]:
                output_file.write("{}\n".format(x))
        with open(os.path.join(self.output_filepath, "fake_average_imported_functions.csv"), "w") as output_file:
            for x in info["imported_functions"]["fake"]:
                output_file.write("{}\n".format(x))
        plot_average_imported_functions(info["imported_functions"]["original"], info["imported_functions"]["fake"],
                                        os.path.join(self.output_filepath,
                                                     "plots/average_imported_functions.png"))
        with open(os.path.join(self.output_filepath, "average_collisions.csv"), "w") as output_file:
            for x in info["collisions"]:
                output_file.write("{}\n".format(x))
        plot_average_collisions(info["collisions"], os.path.join(self.output_filepath, "plots/average_collisions.png"))

    def log(self, d_features, fake_samples, g_features, g_hashed_features, g_paths, g_ember_features_v1, g_ember_features_v2, g_raw_paths, batch_size:int, info:dict):
        """
        Check fake samples against the malware detectors
        """
        benign_features = d_features.cpu().detach().numpy()
        fake_features = fake_samples.cpu().detach().numpy()
        fake_features = fake_features.round().astype(int)

        # Convert Torch tensors to numpy arrays
        d_features = d_features.cpu().detach().numpy().astype(int)
        g_features = g_features.cpu().detach().numpy().astype(int)
        g_hashed_features = g_hashed_features.cpu().detach().numpy()
        g_ember_features_v1 = g_ember_features_v1.cpu().detach().numpy()
        g_ember_features_v2 = g_ember_features_v2.cpu().detach().numpy()

        info["distance_metrics"]["train"]["euclidean"]["between_original_and_fake"].append(
            euclidean_distance_np(g_features, fake_features))
        info["distance_metrics"]["train"]["euclidean"]["between_fake"].append(
            pdist(fake_features, metric="euclidean"))

        info["distance_metrics"]["train"]["minkowski"]["between_original_and_fake"].append(
            minkowski_distance_np(g_features, fake_features))
        info["distance_metrics"]["train"]["minkowski"]["between_fake"].append(
            pdist(fake_features, metric="minkowski"))

        info["distance_metrics"]["train"]["hamming"]["between_original_and_fake"].append(
            hamming_distance_np(g_features, fake_features))
        info["distance_metrics"]["train"]["hamming"]["between_fake"].append(
            pdist(fake_features, metric="hamming"))

        benign_imported_functions = np.sum(d_features, axis=1)
        original_imported_functions = np.sum(g_features, axis=1)
        fake_imported_functions = np.sum(fake_features, axis=1)
        maximum_imported_functions = np.sum(np.maximum(g_features, fake_features), axis=1)
        self.logger.info(
            "Number of imported functions in benign executables: {}; Avg: {};\n"
            "Number of imported functions in original executables: {}; Avg: {};\n"
            "Number of imported functions in fake executables: {}; Avg: {};\n"
            "Number of imported functions in maximum(original, fake): {}; Avg: {}".format(
                benign_imported_functions,
                np.sum(benign_imported_functions) / float(batch_size),
                original_imported_functions,
                np.sum(original_imported_functions) / float(batch_size),
                fake_imported_functions,
                np.sum(fake_imported_functions) / float(batch_size),
                maximum_imported_functions,
                np.sum(maximum_imported_functions) / float(batch_size)
            )
        )

        if self.is_wandb == True:
            wandb.log(
                {
                    "Log: Average imported functions in benign executables": np.sum(benign_imported_functions) / float(
                        batch_size),
                    "Log: Average imported functions in original malicious executables": np.sum(
                        original_imported_functions) / float(batch_size),
                    "Log: Average imported functions in fake malicious executables": np.sum(
                        fake_imported_functions) / float(batch_size),
                }
            )

        g_paths = list(g_paths)
        g_dicts = [load_json(g_paths[i]) for i in range(len(g_paths))]
        fake_hashed_features = np.array([self.feature_extractor.apply_hashing_trick(
            self.feature_extractor.update_imports_dictionary(g_dicts[i], g_features[i], fake_features[i]))
            for i in
            range(fake_features.shape[0])])

        fake_ember_features_v1 = np.array([self.ember_feature_extractor_v1.replace_hashed_import_features(
            g_ember_features_v1[i], self.feature_extractor.apply_hashing_trick(
                self.feature_extractor.update_imports_dictionary(g_dicts[i], g_features[i], fake_features[i])))
            for i in
            range(fake_features.shape[0])])

        fake_ember_features_v2 = np.array([self.ember_feature_extractor_v2.replace_hashed_import_features(
            g_ember_features_v2[i], self.feature_extractor.apply_hashing_trick(
                self.feature_extractor.update_imports_dictionary(g_dicts[i], g_features[i], fake_features[i])))
            for i in
            range(fake_features.shape[0])])

        originally_detected_model_2017 = self.evaluate_on_model_2017(g_features)
        fake_detected_model_2017 = self.evaluate_on_model_2017(fake_features)

        originally_detected_model_2018 = self.evaluate_on_model_2018(g_features)
        fake_detected_model_2018 = self.evaluate_on_model_2018(fake_features)

        originally_detected_hashed_model_2017 = self.evaluate_on_hashed_model_2017(g_hashed_features)
        fake_detected_hashed_model_2017 = self.evaluate_on_hashed_model_2017(fake_hashed_features)

        originally_detected_hashed_model_2018 = self.evaluate_on_hashed_model_2018(g_hashed_features)
        fake_detected_hashed_model_2018 = self.evaluate_on_hashed_model_2018(fake_hashed_features)


        originally_detected_ember_model_2017 = self.evaluate_on_ember_model_2017(g_ember_features_v1)
        fake_detected_ember_model_2017 = self.evaluate_on_ember_model_2017(fake_ember_features_v1)


        originally_detected_ember_model_2018 = self.evaluate_on_ember_model_2018(g_ember_features_v2)
        fake_detected_ember_model_2018 = self.evaluate_on_ember_model_2018(fake_ember_features_v2)


        originally_detected_sorel20m_lightgbm_model = self.evaluate_on_sorel20m_lightgbm_model(g_ember_features_v2)
        fake_detected_sorel20m_lightgbm_model = self.evaluate_on_sorel20m_lightgbm_model(fake_ember_features_v2)


        originally_detected_sorel20m_ffnn_model = self.evaluate_on_sorel20m_ffnn_model(
            g_ember_features_v2)
        fake_detected_sorel20m_ffnn_model = self.evaluate_on_sorel20m_ffnn_model(fake_ember_features_v2)

        self.logger.info("Model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_model_2017, batch_size,
            fake_detected_model_2017, batch_size))
        self.logger.info("Model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_model_2018, batch_size,
            fake_detected_model_2018, batch_size))
        self.logger.info("Hashed model 2017; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_hashed_model_2017, batch_size,
            fake_detected_hashed_model_2017, batch_size))
        self.logger.info("Hashed model 2018; Originally detected: {}/{}; Fake detected: {}/{}".format(
            originally_detected_hashed_model_2018, batch_size,
            fake_detected_hashed_model_2018, batch_size))
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

        avg_collisions = check_collisions_among_generated_samples(fake_features)
        self.logger.info("Average collisions: {}".format(np.sum(avg_collisions) / avg_collisions.shape[0]))
        if self.is_wandb == True:
            wandb.log(
                {
                    "Log: average collisions between fake samples": np.sum(avg_collisions) / avg_collisions.shape[0]
                }
            )

        for metric in info["distance_metrics"]["train"]:
            self.logger.info("{} distance between original and fake samples: {}".format(metric, info["distance_metrics"]["train"][metric]["between_original_and_fake"][-1]))
            #self.logger.info("{} distance between fake samples: {}".format(metric, info["distance_metrics"]["train"][metric]["between_fake"][-1]))
            if self.is_wandb == True:
                wandb.log(
                    {
                        "Log: {} distance between original and fake samples".format(metric):
                            info["distance_metrics"]["train"][metric]["between_original_and_fake"][-1],
                    }
                )
                try:
                    wandb.log(
                        {
                            "Log: {} distance between fake samples".format(metric):
                                info["distance_metrics"]["train"][metric]["between_fake"][-1]
                        }
                    )
                except ValueError as e:
                    pass
        if self.is_wandb == True:

            detection_rates = {
                "model_2017": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "model_2018": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "hashed_model_2017": {
                    "original": 0.0,
                    "fake": 0.0
                },
                "hashed_model_2018": {
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
            }
            detection_rates["model_2017"]["original"] = originally_detected_model_2017 / batch_size
            detection_rates["model_2017"]["fake"] = originally_detected_model_2017 / batch_size
            detection_rates["model_2018"]["original"] = originally_detected_model_2018 / batch_size
            detection_rates["model_2018"]["fake"] = originally_detected_model_2018 / batch_size

            detection_rates["hashed_model_2017"]["original"] = originally_detected_hashed_model_2017 / batch_size
            detection_rates["hashed_model_2017"]["fake"] = fake_detected_hashed_model_2017 / batch_size
            detection_rates["hashed_model_2018"]["original"] = originally_detected_hashed_model_2018 / batch_size
            detection_rates["hashed_model_2018"]["fake"] = fake_detected_hashed_model_2018 / batch_size

            detection_rates["ember_model_2017"]["original"] = originally_detected_ember_model_2017 / batch_size
            detection_rates["ember_model_2017"]["fake"] = fake_detected_ember_model_2017 / batch_size
            detection_rates["ember_model_2018"]["original"] = originally_detected_ember_model_2018 / batch_size
            detection_rates["ember_model_2018"]["fake"] = fake_detected_ember_model_2018 / batch_size

            detection_rates["sorel20m_lightgbm"]["original"] = originally_detected_sorel20m_lightgbm_model/ batch_size
            detection_rates["sorel20m_lightgbm"]["fake"] = fake_detected_sorel20m_lightgbm_model / batch_size
            detection_rates["sorel20m_ffnn"]["original"] = originally_detected_sorel20m_ffnn_model / batch_size
            detection_rates["sorel20m_ffnn"]["fake"] = fake_detected_sorel20m_ffnn_model / batch_size

            wandb.log(
                {"Log: Detection rate of model 2017 on the original samples":
                        detection_rates["model_2017"]["original"],
                    "Log: Detection rate of model 2017 on the fake samples": detection_rates["model_2017"][
                        "fake"],
                    "Log: Detection rate of model 2018 on the original samples":
                        detection_rates["model_2018"]["original"],
                    "Log: Detection rate of model 2018 on the fake samples":
                        detection_rates["model_2018"][
                            "fake"],
                    "Log: Detection rate of hashed 2017 on the original samples":
                        detection_rates["hashed_model_2017"]["original"],
                    "Log: Detection rate of hashed 2017 on the fake samples": detection_rates["hashed_model_2017"][
                        "fake"],
                    "Log: Detection rate of hashed 2018 on the original samples":
                        detection_rates["hashed_model_2018"]["original"],
                    "Log: Detection rate of hashed 2018 on the fake samples":
                        detection_rates["hashed_model_2018"][
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
                },
            )

        return info

    def validate(self, dataset:torch.utils.data.Dataset, epoch:int, batch_size:int=32):
        dataloader = self.initialize_dataloader(dataset, batch_size, shuffle=False, drop_last=False)
        output_filepath = os.path.join(self.output_filepath,
                               "validation_results/validation_{}epoch.txt".format(epoch))
        evaluation_info = self.check_evasion_rate(dataset, dataloader, output_filepath)
        self.log_to_wandb(evaluation_info, len(dataset), "Validation")

    def test(self, dataset:torch.utils.data.Dataset, batch_size: int = 32, replace: bool = True, output_filepath: str = None, shuffle: bool = False):
        dataloader = self.initialize_dataloader(dataset, batch_size, shuffle=shuffle, drop_last=False)
        if output_filepath is None:
            output_filepath= os.path.join(self.output_filepath, "testing_results/testing_results.txt")
        evaluation_info = self.check_evasion_rate(dataset, dataloader, output_filepath, replace=replace)
        self.log_to_wandb(evaluation_info, len(dataset), "Test")


    def evaluate_against_ml_models(self, evaluation_info:dict, fake_samples, g_features, g_hashed_features, g_ember_features_v1, g_ember_features_v2, g_paths, g_raw_paths, replace=True):
        """
        Evaluate current batch against the ML detectors
        :return:
        """
        # Check samples against malware detectors
        fake_features = fake_samples.cpu().detach().numpy()
        fake_features = fake_features.round().astype(int)

        # Convert Torch tensors to numpy arrays
        g_features = g_features.cpu().detach().numpy().astype(int)
        g_hashed_features = g_hashed_features.cpu().detach().numpy()
        g_ember_features_v1 = g_ember_features_v1.cpu().detach().numpy()
        g_ember_features_v2 = g_ember_features_v2.cpu().detach().numpy()

        if fake_features.shape[0] > 1:
            evaluation_info["distance_metrics"]["euclidean"]["between_original_and_fake"].append(
                euclidean_distance_np(g_features, fake_features))
            evaluation_info["distance_metrics"]["euclidean"]["between_fake"].append(
                pdist(fake_features, metric="euclidean"))

            evaluation_info["distance_metrics"]["minkowski"]["between_original_and_fake"].append(
                minkowski_distance_np(g_features, fake_features))
            evaluation_info["distance_metrics"]["minkowski"]["between_fake"].append(
                pdist(fake_features, metric="minkowski"))

            evaluation_info["distance_metrics"]["hamming"]["between_original_and_fake"].append(
                hamming_distance_np(g_features, fake_features))
            evaluation_info["distance_metrics"]["hamming"]["between_fake"].append(
                pdist(fake_features, metric="hamming"))

        original_imported_functions = np.sum(g_features, axis=1)
        fake_imported_functions = np.sum(fake_features, axis=1)
        collisions = check_collisions_among_generated_samples(fake_features)

        # print("Maximum arr: {};{}".format(maximum_arr.shape, maximum_arr))
        # print("Maximum imported funtions: {};{}".format(maximum_imported_functions.shape, maximum_imported_functions))
        # print("Collisions: {};{}".format(collisions.shape, collisions))
        evaluation_info["imported_functions"]["original"].extend(original_imported_functions)
        evaluation_info["imported_functions"]["fake"].extend(fake_imported_functions)
        evaluation_info["collisions"].extend(collisions)

        g_paths = list(g_paths)
        g_dicts = [load_json(g_paths[i]) for i in range(len(g_paths))]  # Need to use the imports dictionary
        fake_hashed_features = np.array([self.feature_extractor.apply_hashing_trick(
            self.feature_extractor.update_imports_dictionary(g_dicts[i], g_features[i], fake_features[i]))
            for i in
            range(fake_features.shape[0])])

        fake_hashed_features = torch.from_numpy(fake_hashed_features)
        act1 = self.inception_model.retrieve_features(fake_hashed_features)
        act1 = act1.cpu().detach().numpy()
        evaluation_info["intermediate_features"]["fake"].append(act1)

        x_original = []
        x_adversarial = []

        if replace == True:
            fake_ember_features_v1 = np.array([self.ember_feature_extractor_v1.replace_hashed_import_features(
                g_ember_features_v1[i], self.feature_extractor.apply_hashing_trick(
                    self.feature_extractor.update_imports_dictionary(g_dicts[i], g_features[i], fake_features[i])))
                for i in
                range(fake_features.shape[0])])

            fake_ember_features_v2 = np.array([self.ember_feature_extractor_v2.replace_hashed_import_features(
                g_ember_features_v2[i], self.feature_extractor.apply_hashing_trick(
                    self.feature_extractor.update_imports_dictionary(g_dicts[i], g_features[i], fake_features[i])))
                for i in
                range(fake_features.shape[0])])
            x_original = np.squeeze(np.array(x_original))
            x_adversarial = np.squeeze(np.array(x_adversarial))
        else:
            # Need to get original and resulting byte sequence!

            fake_ember_features_v1 = []
            fake_ember_features_v2 = []
            exceptions_produced = 0
            #if self.malconv_model or self.nonneg_malconv_model:
            for i in range(fake_features.shape[0]):
                try:
                    with open(g_raw_paths[0], "rb") as binary_file:
                        bytez_content = binary_file.read()
                        original_bytez_arr = np.frombuffer(bytez_content, dtype=np.uint8).astype(np.int16)
                        original_bytez_arr_malconvgct = np.frombuffer(bytez_content, dtype=np.uint8).astype(
                            np.int16) + 1  # index 0 will be special padding index
                    
                    pe_modifier = PEModifier(g_raw_paths[0])
                    binary = pe_modifier._get_binary(original_bytez_arr.tolist())
                    adversarial_bytez, adversarial_bytez_int_list, adversarial_lief_binary = pe_modifier.add_imports(
                        fake_features[i], self.inverse_vocabulary_mapping)




                    adversarial_bytez, adversarial_bytez_int_list, adversarial_lief_binary = pe_modifier.add_imports(
                        fake_features[i], self.inverse_vocabulary_mapping)
                    hex_values = adversarial_bytez_int_list[:self.malconv_max_length]  # (1048576,)
                    c[:len(hex_values)] = hex_values  # (1048576,)
                    c = np.expand_dims(c, axis=0)

                    x_original.append(copy.deepcopy(b))
                    x_adversarial.append(copy.deepcopy(c))

                    fake_ember_features_v1.append(self.ember_feature_extractor_v1.feature_vector(adversarial_bytez))
                    fake_ember_features_v2.append(self.ember_feature_extractor_v2.feature_vector(adversarial_bytez))
                except Exception as e:
                    exceptions_produced += 1
                    print(e)
            fake_ember_features_v1 = np.array(fake_ember_features_v1)
            fake_ember_features_v2 = np.array(fake_ember_features_v2)
            x_original = np.squeeze(np.array(x_original), axis=1)
            x_adversarial = np.squeeze(np.array(x_adversarial), axis=1)

        # Features
        if self.features_model_2017 is not None:
            originally_detected_model_2017 = self.evaluate_on_model_2017(g_features)
            fake_detected_model_2017 = self.evaluate_on_model_2017(fake_features)
            evaluation_info["detections"]["original"]["model_2017"] += originally_detected_model_2017
            evaluation_info["detections"]["fake"]["model_2017"] += fake_detected_model_2017
        else:
            evaluation_info["detections"]["original"]["model_2017"] += 0
            evaluation_info["detections"]["fake"]["model_2017"] += 0

        if self.features_model_2018 is not None:
            originally_detected_model_2018 = self.evaluate_on_model_2018(g_features)
            fake_detected_model_2018 = self.evaluate_on_model_2018(fake_features)
            evaluation_info["detections"]["original"]["model_2018"] += originally_detected_model_2018
            evaluation_info["detections"]["fake"]["model_2018"] += fake_detected_model_2018
        else:
            evaluation_info["detections"]["original"]["model_2018"] += 0
            evaluation_info["detections"]["fake"]["model_2018"] += 0

        # Hashed features
        if self.hashed_features_model_2017 is not None:
            originally_detected_hashed_model_2017 = self.evaluate_on_hashed_model_2017(g_hashed_features)
            fake_detected_hashed_model_2017 = self.evaluate_on_hashed_model_2017(fake_hashed_features)
            evaluation_info["detections"]["original"]["hashed_model_2017"] += originally_detected_hashed_model_2017
            evaluation_info["detections"]["fake"]["hashed_model_2017"] += fake_detected_hashed_model_2017
        else:
            evaluation_info["detections"]["original"]["hashed_model_2017"] += 0
            evaluation_info["detections"]["fake"]["hashed_model_2017"] += 0

        if self.hashed_features_model_2018 is not None:
            originally_detected_hashed_model_2018 = self.evaluate_on_hashed_model_2018(g_hashed_features)
            fake_detected_hashed_model_2018 = self.evaluate_on_hashed_model_2018(fake_hashed_features)
            evaluation_info["detections"]["original"]["hashed_model_2018"] += originally_detected_hashed_model_2018
            evaluation_info["detections"]["fake"]["hashed_model_2018"] += fake_detected_hashed_model_2018
        else:
            evaluation_info["detections"]["original"]["hashed_model_2018"] += 0
            evaluation_info["detections"]["fake"]["hashed_model_2018"] += 0

        # EMBER features
        if self.ember_model_2017 is not None:
            originally_detected_ember_model_2017 = self.evaluate_on_ember_model_2017(g_ember_features_v1)
            fake_detected_ember_model_2017 = self.evaluate_on_ember_model_2017(fake_ember_features_v1)
            evaluation_info["detections"]["original"]["ember_model_2017"] += originally_detected_ember_model_2017
            evaluation_info["detections"]["fake"]["ember_model_2017"] += fake_detected_ember_model_2017
        else:
            evaluation_info["detections"]["original"]["ember_model_2017"] += 0
            evaluation_info["detections"]["fake"]["ember_model_2017"] += 0

        if self.ember_model_2018 is not None:
            originally_detected_ember_model_2018 = self.evaluate_on_ember_model_2018(g_ember_features_v2)
            fake_detected_ember_model_2018 = self.evaluate_on_ember_model_2018(fake_ember_features_v2)
            evaluation_info["detections"]["original"]["ember_model_2018"] += originally_detected_ember_model_2018
            evaluation_info["detections"]["fake"]["ember_model_2018"] += fake_detected_ember_model_2018
        else:
            evaluation_info["detections"]["original"]["ember_model_2018"] += 0
            evaluation_info["detections"]["fake"]["ember_model_2018"] += 0

        if self.sorel20m_lightgbm_model is not None:
            originally_detected_sorel20m_lightgbm_model = self.evaluate_on_sorel20m_lightgbm_model(g_ember_features_v2)
            fake_detected_sorel20m_lightgbm_model = self.evaluate_on_sorel20m_lightgbm_model(fake_ember_features_v2)
            evaluation_info["detections"]["original"][
                "sorel20m_lightgbm_model"] += originally_detected_sorel20m_lightgbm_model
            evaluation_info["detections"]["fake"]["sorel20m_lightgbm_model"] += fake_detected_sorel20m_lightgbm_model
        else:
            evaluation_info["detections"]["original"][
                "sorel20m_lightgbm_model"] += 0
            evaluation_info["detections"]["fake"]["sorel20m_lightgbm_model"] += 0

        if self.sorel20m_ffnn_model is not None:
            originally_detected_sorel20m_ffnn_model = self.evaluate_on_sorel20m_ffnn_model(
                g_ember_features_v2)
            fake_detected_sorel20m_ffnn_model = self.evaluate_on_sorel20m_ffnn_model(fake_ember_features_v2)
            evaluation_info["detections"]["original"]["sorel20m_ffnn_model"] += originally_detected_sorel20m_ffnn_model
            evaluation_info["detections"]["fake"]["sorel20m_ffnn_model"] += fake_detected_sorel20m_ffnn_model
        else:
            evaluation_info["detections"]["original"]["sorel20m_ffnn_model"] += 0
            evaluation_info["detections"]["fake"]["sorel20m_ffnn_model"] += 0

        if self.malconv_model is not None:
            originally_detected_malconv_model = self.evaluate_on_malconv_model(x_original)
            fake_detected_malconv_model = self.evaluate_on_malconv_model(x_adversarial)
            evaluation_info["detections"]["original"]["malconv_model"] += originally_detected_malconv_model
            evaluation_info["detections"]["fake"]["malconv_model"] += fake_detected_malconv_model
            if exceptions_produced != 0:
                evaluation_info["detections"]["original"]["malconv_model"] += exceptions_produced
                evaluation_info["detections"]["fake"]["malconv_model"] += exceptions_produced
        else:
            evaluation_info["detections"]["original"]["malconv_model"] += 0
            evaluation_info["detections"]["fake"]["malconv_model"] += 0

        if self.nonneg_malconv_model is not None:
            originally_detected_nonnegative_malconv_model = self.evaluate_on_nonnegative_malconv_model(x_original)
            fake_detected_nonnegative_malconv_model = self.evaluate_on_nonnegative_malconv_model(x_adversarial)
            evaluation_info["detections"]["original"][
                "nonneg_malconv_model"] += originally_detected_nonnegative_malconv_model
            evaluation_info["detections"]["fake"]["nonneg_malconv_model"] += fake_detected_nonnegative_malconv_model
            if exceptions_produced != 0:
                evaluation_info["detections"]["original"][
                    "nonneg_malconv_model"] += exceptions_produced
                evaluation_info["detections"]["fake"]["nonneg_malconv_model"] += exceptions_produced
        else:
            evaluation_info["detections"]["original"][
                "nonneg_malconv_model"] += 0
            evaluation_info["detections"]["fake"]["nonneg_malconv_model"] += 0
        return evaluation_info

    def check_evasion_rate(self, dataset: torch.utils.data.Dataset, dataloader: DataLoader, output_filepath: str, replace: bool = True):
        self.generator.eval()
        self.discriminator.eval()

        evaluation_info = self.initialize_evaluation_info_dictionary()

        j = 0
        for (g_features, g_hashed_features, g_paths, g_ember_features_v1, g_ember_features_v2, g_raw_paths,
             g_y) in dataloader:
            print("{};{}".format(j,g_raw_paths))
            g_features = g_features.to(self.device)
            # Create fake feature vector
            noise = torch.randn(g_features.shape[0], self.generator_parameters["z_size"]).to(
                self.device)
            fake_samples = self.generator([g_features, noise])
            evaluation_info = self.evaluate_against_ml_models(
                evaluation_info,
                fake_samples,
                g_features,
                g_hashed_features,
                g_ember_features_v1,
                g_ember_features_v2,
                g_paths,
                g_raw_paths,
                replace=replace
            )
            j += 1

            if j == 100:
                break

        fid_score = calculate_fid(np.concatenate(evaluation_info["intermediate_features"]["fake"]), self.benign_intermediate_features)
        evaluation_info["fid"] = fid_score
        self.print_info(evaluation_info, len(dataset))
        self.log_info(evaluation_info, len(dataset))
        self.write_to_file(evaluation_info, output_filepath, len(dataset))

        self.generator.train()
        self.discriminator.train()
        return evaluation_info

    def save_models(self, info:dict, evasion_rates:dict):
        # Features
        if self.features_model_2017 is not None:
            info["evasion_rate"]["list"]["model_2017"].append(evasion_rates["model_2017"])
            if evasion_rates["model_2017"] < info["evasion_rate"]["best"]["model_2017"]:
                info["evasion_rate"]["best"]["model_2017"] = evasion_rates["model_2017"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_features_model_2017.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_model_2017"] = evasion_rates["model_2017"]

        if self.features_model_2018 is not None:
            info["evasion_rate"]["list"]["model_2018"].append(evasion_rates["model_2018"])
            if evasion_rates["model_2018"] < info["evasion_rate"]["best"]["model_2018"]:
                info["evasion_rate"]["best"]["model_2018"] = evasion_rates["model_2018"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_features_model_2018.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_model_2018"] = evasion_rates["model_2018"]

        # Hashed features
        if self.hashed_features_model_2017 is not None:
            info["evasion_rate"]["list"]["hashed_model_2017"].append(evasion_rates["hashed_model_2017"])
            if evasion_rates["hashed_model_2017"] < info["evasion_rate"]["best"]["hashed_model_2017"]:
                info["evasion_rate"]["best"]["hashed_model_2017"] = evasion_rates["hashed_model_2017"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_hashed_features_model_2017.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_hashed_model_2017"] = evasion_rates["hashed_model_2017"]

        if self.hashed_features_model_2018 is not None:
            info["evasion_rate"]["list"]["hashed_model_2018"].append(evasion_rates["hashed_model_2018"])
            if evasion_rates["hashed_model_2018"] < info["evasion_rate"]["best"]["hashed_model_2018"]:
                info["evasion_rate"]["best"]["hashed_model_2018"] = evasion_rates["hashed_model_2018"]
                torch.save(self.generator, os.path.join(self.output_filepath,
                                                        "generator/generator_best_hashed_features_model_2018.pt"))
                if self.is_wandb == True:
                    wandb.run.summary["best_hashed_model_2018"] = evasion_rates["hashed_model_2018"]

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
                    "{}: Detection rate of model 2017 on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["model_2017"] / dataset_size,
                    "{}: Detection rate of model 2017 on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["model_2017"] / dataset_size,
                    "{}: Detection rate of model 2018 on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["model_2018"] / dataset_size,
                    "{}: Detection rate of model 2018 on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["model_2018"] / dataset_size,

                    "{}: Detection rate of hashed 2017 on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["hashed_model_2017"]/dataset_size,
                    "{}: Detection rate of hashed 2017 on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["hashed_model_2017"] / dataset_size,
                    "{}: Detection rate of hashed 2018 on the original malicious samples".format(mode):
                        evaluation_info["detections"]["original"]["hashed_model_2018"] / dataset_size,
                    "{}: Detection rate of hashed 2018 on the fake malicious samples".format(mode):
                        evaluation_info["detections"]["fake"]["hashed_model_2018"] / dataset_size,

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
                }
            )

            wandb.log(
                {
                    "{}: FID score".format(mode): evaluation_info["fid"]
                }
            )

            wandb.log(
                {
                    "{}: Average imported functions in original malicious executables".format(mode): sum(
                        evaluation_info["imported_functions"]["original"]) / len(
                        evaluation_info["imported_functions"]["original"]),
                    "{}: Average imported functions in fake malicious executables".format(mode): sum(
                        evaluation_info["imported_functions"]["fake"]) / len(evaluation_info["imported_functions"]["fake"]),
                    "{}: Average collisions between fake malicious executables".format(mode): sum(
                        evaluation_info["collisions"]) / len(evaluation_info["collisions"])
                }
            )

            for metric in evaluation_info["distance_metrics"]:
                print("Metric: ", metric)
                if metric == "euclidean":
                    wandb.log(
                        {
                            "{}: {} distance between original and fake samples".format(mode, metric): sum(
                                evaluation_info["distance_metrics"][metric]["between_original_and_fake"]) / len(
                                evaluation_info["distance_metrics"][metric]["between_original_and_fake"])
                        }
                    )
                    try:
                        wandb.log(
                            {
                                "{}: {} distance between fake samples".format(mode, metric): sum(
                                    evaluation_info["distance_metrics"][metric]["between_fake"]) / len(
                                    evaluation_info["distance_metrics"][metric]["between_fake"])
                            }
                        )
                    except ValueError as e:
                        pass



