from abc import ABC, abstractmethod
from src.feature_extractors.ember_feature_extractor import EmberFeatureExtractor
from src.ml_models.torch_models.malconv_detector.malconv import restore_model
from src.ml_models.torch_models.sorel20m_detector.nets import PENetwork
from src.ml_models.torch_models.nonnegative_malconv_detector.malconv_model import MalConvModel
from src.ml_models.torch_models.malconvgct_detector.MalConvGCT_nocat import MalConvGCT
from torch.utils.data import DataLoader
from scipy.spatial.distance import pdist
import torch
import os
import json
import lightgbm as lgb
import numpy as np
import sys
import logging
import wandb


class SkeletonGAN(ABC):
    def init_directories(self, output_filepath):
        self.output_filepath = output_filepath
        try:
            self.logger.info("Creating directory structure to save the checkpoints and plots")
            os.makedirs(os.path.join(output_filepath, "generator"))
            os.makedirs(os.path.join(output_filepath, "discriminator"))
            os.makedirs(os.path.join(output_filepath, "plots"))
            os.makedirs(os.path.join(output_filepath, "validation_results"))
            os.makedirs(os.path.join(output_filepath, "testing_results"))
            os.makedirs(os.path.join(output_filepath, "logging"))
            self.logger.info("Directory structure created.\n {}\n{}\n{}\n{}\n{}\n".format(
                os.path.join(output_filepath, "generator"),
                os.path.join(output_filepath, "discriminator"),
                os.path.join(output_filepath, "plots"),
                os.path.join(output_filepath, "validation_results"),
                os.path.join(output_filepath, "testing_results"),
                os.path.join(output_filepath, "logging")))

        except FileExistsError as fee:
            self.logger.info(fee)

    def create_logger(self, output_filepath, train=True):
        self.logger = logging.getLogger('GAN Logger')
        self.logger.setLevel(logging.DEBUG)

        # create file handler and set level to debug
        try:
            os.makedirs(output_filepath)
        except FileExistsError as fee:
            print(fee)

        if train is True:
            ch = logging.FileHandler(filename=os.path.join(output_filepath, "execution.log"), mode="w")
        else:
            ch = logging.FileHandler(filename=os.path.join(output_filepath, "test.log"), mode="w")

        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

        # 'application' code
        self.logger.info('GAN Logger created.')

    def init_cuda(self, cuda_device=None):
        if cuda_device is None:
            self.cuda = False
            self.dev = "cpu"
        else:
            self.cuda = True
            self.dev = "cuda:{}".format(cuda_device)
        self.logger.info("Device: {}".format(self.dev))
        self.device = torch.device(self.dev)
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.logger.info("Tensor type: {}".format(type(self.Tensor)))

    def init_wandb(self, training_parameters:dict, is_wandb:bool=True):
        self.is_wandb = is_wandb
        if self.is_wandb == True:
            wandb.init(
                project=training_parameters["project"],
                entity=training_parameters["entity"],
                config={
                    "learning_rate": training_parameters["lr"],
                    "beta1": training_parameters["beta1"],
                    "beta2": training_parameters["beta2"],
                    "epochs": training_parameters["num_epochs"],
                    "batch_size": training_parameters["batch_size"],
                    "label_smoothing": training_parameters["label_smoothing"]
            })


    def initialize_ember_feature_extractors(self, feature_version:int=1):
        self.feature_version = feature_version
        self.ember_feature_extractor_v1 = EmberFeatureExtractor(feature_version=1)
        self.ember_feature_extractor_v2 = EmberFeatureExtractor(feature_version=2)

    def build_blackbox_detectors(
            self,
            features_model_2017_filepath: str,
            is_features_model_2017: bool,
            features_model_2018_filepath: str,
            is_features_model_2018: bool,
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
            nonneg_malconv_model_filepath:str,
            is_nonneg_malconv_model: bool,
            malconvgct_model_filepath: str,
            is_malconvgct_model: bool
    ):
        self.features_model_2017 = None
        self.features_model_2018 = None
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

    def build_lightgbm_blackbox_detector(self, lightgbm_model_filepath:str):
        lightgbm_model = lgb.Booster(model_file=lightgbm_model_filepath)
        return lightgbm_model

    def build_malconv_blackbox_detector(self, malconv_model_filepath:str):
        malconv_model = restore_model(malconv_model_filepath)
        return malconv_model

    def build_sorel20m_ffnn_blackbox_detector(self, sorel20m_ffnn_checkpoint_file: str):
        model = PENetwork(use_malware=True, use_counts=False, use_tags=False, n_tags=None,
                          feature_dimension=2381)
        model.load_state_dict(torch.load(sorel20m_ffnn_checkpoint_file, map_location=torch.device("cpu")))
        #model.to(torch.device("cpu"))
        return model

    def build_nonnegative_malconv_detector(self, nonnegative_malconv_filepath:str):
        model = MalConvModel(nonnegative_malconv_filepath, thresh=0.35, name='nonneg_malconv' )
        return model

    def build_malconvgct_detector(self, malconvgct_filepath: str):
        model = MalConvGCT(channels=256, window_size=256,
                           stride=64, )
        model_checkpoint = torch.load(malconvgct_filepath, map_location=torch.device("cpu"))
        model.load_state_dict(model_checkpoint['model_state_dict'], strict=False)
        return model

    @abstractmethod
    def load_inception_model(self, inception_parameters:dict, inception_checkpoint:str):
        pass

    @abstractmethod
    def build_generator_network(self, generator_parameters:dict):
        pass

    @abstractmethod
    def build_discriminator_network(self, discriminator_parameters: dict):
        pass

    @abstractmethod
    def train(self, training_parameters:dict):
        pass

    @abstractmethod
    def log(self, *args, **kwargs):
        pass

    @abstractmethod
    def initialize_criterions(self):
        pass

    @abstractmethod
    def initialize_optimizers(self, training_parameters:dict):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass

    @abstractmethod
    def initialize_training_and_validation_generator_datasets(self, *args, **kwargs):
        pass

    @abstractmethod
    def initialize_training_and_validation_discriminator_datasets(self, *args, **kwargs):
        pass

    @abstractmethod
    def initialize_feature_extractor(self):
        pass

    def evaluate_on_ember_model_2017(self, ember_features_v1):
        if self.ember_model_2017 is not None:
            model_output = self.ember_model_2017.predict(ember_features_v1)
            detections = model_output.round().astype(int)
            detected = np.sum(detections)
            return detected
        return None

    def evaluate_on_ember_model_2018(self, ember_features_v2):
        if self.ember_model_2018 is not None:
            model_output = self.ember_model_2018.predict(ember_features_v2)
            detections = model_output.round().astype(int)
            detected = np.sum(detections)
            return detected
        return None

    def evaluate_on_sorel20m_lightgbm_model(self, ember_features_v2):
        if self.sorel20m_lightgbm_model is not None:
            model_output = self.sorel20m_lightgbm_model.predict(ember_features_v2)
            detections = model_output.round().astype(int)
            detected = np.sum(detections)
            return detected
        return None

    def evaluate_on_sorel20m_ffnn_model(self, ember_features_v2):
        if self.sorel20m_ffnn_model is not None:
            # Get black-box model predictions
            ember_features_v2 = torch.from_numpy(ember_features_v2)

            sorel20m_ffnn_output = self.sorel20m_ffnn_model(ember_features_v2)["malware"]
            sorel20m_ffnn_output = torch.squeeze(sorel20m_ffnn_output, dim=1)
            sorel20m_ffnn_output = sorel20m_ffnn_output.cpu().detach().numpy()

            detections = sorel20m_ffnn_output.round().astype(int)
            detected = np.sum(detections)
            return detected

    def evaluate_on_malconv_model(self, x:np.array):
        if self.malconv_model is not None:
            output = self.malconv_model.predict(x)
            detections = output.round().astype(int)
            detected = np.sum(detections)
            return detected
        return None

    def evaluate_on_malconvgct_model(self, x: np.array):
        if self.malconvgct_model is not None:
            if x.ndim == 1:
                #print("Shape: ", x.shape)
                y_pred, y_pred_bool = self.malconvgct_model.predict(x)
                #print("Output y_pred: {}; y_pred_bool: {}".format(y_pred, y_pred_bool))
                #print("Output boolean: {}".format(y_pred_bool))
                if type(y_pred) is float:
                    detections = 1 if y_pred_bool == True else 0
                else:
                    detections = 1 if y_pred_bool == True else 0 # y_pred[:, 0].round().astype(int)
                detected = np.sum(detections)
                return detected
            else:
                detected = 0
                for k in range(x.shape[0]):
                    y_pred, y_pred_bool = self.malconvgct_model.predict(x[k])
                    # print(y_pred, y_pred_bool)
                    detected += 1 if y_pred_bool == True else 0
                return detected
        return None

    def evaluate_on_nonnegative_malconv_model(self, x: np.array):
        if self.nonneg_malconv_model is not None:
            print("X:", x)
            print("Shape: ",x.shape)

            if x.ndim == 1:
                y_pred, y_pred_bool = self.nonneg_malconv_model.predict(x)
                print("Output y_pred: {}; y_pred_bool: {}".format(y_pred, y_pred_bool))
                #print("Output boolean: {}".format(y_pred_bool))

                if type(y_pred) is float:
                    detections = 1 if y_pred_bool == True else 0
                else:
                    detections = 1 if y_pred_bool == True else 0
                detected = np.sum(detections)
                return detected
            else:
                detected = 0
                for k in range(x.shape[0]):
                    y_pred, y_pred_bool = self.nonneg_malconv_model.predict(x[k])
                    print("Output y_pred: {}; y_pred_bool: {}".format(y_pred, y_pred_bool))

                    #print(y_pred, y_pred_bool)
                    detected += 1 if y_pred_bool == True else 0
                return detected
        return None

    def initialize_dataloader(self, dataset:torch.utils.data.Dataset, batch_size:int, shuffle:bool=True, drop_last:bool=True):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        return dataloader

    def calculate_distance_between_fake_samples(self, features, metrics_info:dict):
        if isinstance(features, torch.cuda.FloatTensor) or isinstance(features, torch.FloatTensor):
            features = features.cpu().detach().numpy()
        for metric in metrics_info.keys():
            distance = pdist(features, metric=metric)
            metrics_info[metric]["between_fake"].append(float(np.sum(distance))/float(distance.shape[0]))
        return metrics_info

    def calculate_distance_between_original_and_fake_samples(self, original_features, end_features, metrics_info:dict):
        if isinstance(original_features, torch.cuda.FloatTensor) or isinstance(original_features, torch.FloatTensor):
            original_features = original_features.cpu().detach().numpy()
        if isinstance(end_features, torch.cuda.FloatTensor) or isinstance(end_features, torch.FloatTensor):
            end_features = end_features.cpu().detach().numpy()
        for metric in metrics_info.keys():
            distances = []
            for i in range(original_features.shape[0]):
                arr = np.array([original_features[i], end_features[i]])
                distances.append(pdist(arr, metric=metric)[0])
            metrics_info[metric]["between_original_and_fake"].append(float(sum(distances))/float(len(distances)))
        return metrics_info

    def watch_models(self, log="all", log_freq=200, log_graph=False):
        if self.is_wandb == True:
            wandb.watch(self.discriminator, log=log, log_freq=log_freq, log_graph=log_graph)
            wandb.watch(self.generator, log=log, log_freq=log_freq, log_graph=log_graph)
