import sys
import argparse
sys.path.append("../../../../")
from src.gan_implementations.malgan.byte.malgan_v2 import SkeletonMalGAN
from src.gan_implementations.utils import load_json
from src.gan_implementations.byte_histogram_dataset import ByteHistogramDataset


class MalGAN(SkeletonMalGAN):

    def predict_labels_with_blackbox_model(self, features:dict):
        if self.ember_model_2017 is not None:
            model_output = self.ember_model_2017.predict(features["ember_v1"])
            y_pred = model_output.round().astype(int)
            return y_pred
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MalGAN training')
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
                        default="models/MalGAN_ember_v1/")
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

    gan = MalGAN()
    gan.create_logger(args.output_filepath)
    gan.init_directories(args.output_filepath)
    gan.init_cuda(args.cuda_device)
    gan.initialize_feature_extractor()
    gan.initialize_ember_feature_extractors()

    gan.initialize_training_and_validation_generator_datasets(
        args.malware_histogram_features_filepath,
        args.malware_ember_features_filepath_version1,
        args.malware_ember_features_filepath_version2,
        args.malware_raw_executables_filepath,
        args.malware_raw_npz_executables_filepath,
        args.training_malware_annotations_filepath,
        args.validation_malware_annotations_filepath
    )
    gan.initialize_training_and_validation_discriminator_datasets(
        args.goodware_histogram_features_filepath,
        args.goodware_ember_features_filepath_version1,
        args.goodware_ember_features_filepath_version2,
        args.goodware_raw_executables_filepath,
        args.goodware_raw_npz_executables_filepath,
        args.training_goodware_annotations_filepath,
        args.validation_goodware_annotations_filepath
    )

    gan.build_generator_network(args.generator_parameters_filepath)
    gan.build_discriminator_network(args.discriminator_parameters_filepath)

    gan.build_blackbox_detectors(
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
    gan.load_inception_model(inception_parameters, args.inception_checkpoint)
    gan.load_inception_benign_intermediate_features(args.inception_features_filepath)

    training_parameters = load_json(args.training_parameters_filepath)
    gan.init_wandb(training_parameters, args.wandb)
    gan.train(training_parameters)

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
        gan.test(test_generator_dataset, training_parameters["validation_batch_size"])
