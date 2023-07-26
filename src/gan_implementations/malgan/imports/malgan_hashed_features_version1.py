import sys
import argparse
sys.path.append("../../../../")
from src.gan_implementations.malgan.imports.malgan import SkeletonMalGAN
from src.gan_implementations.utils import load_json
from src.gan_implementations.imports_dataset import ImportsDataset


class MalGAN(SkeletonMalGAN):
    def predict_labels_with_blackbox_model(self, features:dict):
        if self.hashed_features_model_2017 is not None:
            model_output = self.hashed_features_model_2017.predict(features["hashed"])
            y_pred = model_output.round().astype(int)
            return y_pred
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MalGAN training with hashed black-box (version 1)')
    parser.add_argument("--malware_imports_features_filepath",
                        type=str,
                        help="Filepath to maliciouss import features",
                        default="../../../npz/BODMAS/imports_features/BODMAS/")
    parser.add_argument("--malware_hashed_imports_features_filepath",
                        type=str,
                        help="Filepath to malicious hashed import features",
                        default="../../../npz/BODMAS/hashed_imports_features/BODMAS")
    parser.add_argument("--malware_imports_filepath",
                        type=str,
                        help="Filepath to maliciouss import dictionaries",
                        default="../../../npz/BODMAS/imports/BODMAS/")
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
    parser.add_argument("--training_malware_annotations_filepath",
                        type=str,
                        help="Filepath to malicious annotations (training)",
                        default="../../../annotations/imports/data/bodmas_train.csv")
    parser.add_argument("--validation_malware_annotations_filepath",
                        type=str,
                        help="Filepath to malicious annotations (validation)",
                        default="../../../annotations/imports/data/bodmas_validation.csv")
    parser.add_argument("--goodware_imports_features_filepath",
                        type=str,
                        help="Filepath to benign import features",
                        default="../../../npz/BODMAS/imports_features/benign/")
    parser.add_argument("--goodware_hashed_imports_features_filepath",
                        type=str,
                        help="Filepath to benign hashed import features",
                        default="../../../npz/BODMAS/hashed_imports_features/benign/")
    parser.add_argument("--goodware_imports_filepath",
                        type=str,
                        help="Filepath to benign import dictionaries",
                        default="../../../npz/BODMAS/imports/benign")
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
    parser.add_argument("--training_goodware_annotations_filepath",
                        type=str,
                        help="Filepath to benign annotations (training)",
                        default="../../../annotations/imports/data/benign_train.csv")
    parser.add_argument("--validation_goodware_annotations_filepath",
                        type=str,
                        help="Filepath to benign annotations (validation)",
                        default="../../../annotations/imports/data/benign_validation.csv")
    parser.add_argument("--vocabulary_mapping_filepath",
                        type=str,
                        help="Vocabulary mapping filepath",
                        default="../../../feature_extractors/imports_vocabulary/ember/vocabulary_mapping_top150.json")
    parser.add_argument("--inverse_vocabulary_mapping_filepath",
                        type=str,
                        help="Inverse vocabulary mapping filepath",
                        default="../../../feature_extractors/imports_vocabulary/ember/inverse_vocabulary_mapping_top150.json")
    parser.add_argument("--output_filepath",
                        type=str,
                        help="Generator and discriminator torch models filepaths",
                        default="models/MalGAN_top150_import_features/")
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
                        default="../../../annotations/imports/data/bodmas_test.csv")
    parser.add_argument("--test_goodware_annotations_filepath",
                        type=str,
                        help="Filepath to benign annotations (test)",
                        default="../../../annotations/imports/data/benign_test.csv")
    parser.add_argument("--feature_version",
                        type=int,
                        help="EMBER feature version",
                        default=2)

    parser.add_argument("--imports_lightgbm_model_filepath_version1",
                        type=str,
                        help="Imports black-box detector (Version 1)",
                        default="../../../ml_models/lightgbm_models/imports_detector/top150_imports_model_2017.txt")
    parser.add_argument("--imports_lightgbm_model_filepath_version2",
                        type=str,
                        help="Imports black-box detector (Version 2)",
                        default="../../../ml_models/lightgbm_models/imports_detector/top150_imports_model_2018.txt")
    parser.add_argument('--imports-version1', dest='imports_version1', action='store_true')
    parser.add_argument('--no-imports-version1', dest='imports_version1', action='store_false')
    parser.set_defaults(imports_version1=False)
    parser.add_argument('--imports-version2', dest='imports_version2', action='store_true')
    parser.add_argument('--no-imports-version2', dest='imports_version2', action='store_false')
    parser.set_defaults(imports_version2=False)

    parser.add_argument("--hashed_imports_lightgbm_model_filepath_version1",
                        type=str,
                        help="Hashed imports black-box detector (Version 1)",
                        default="../../../ml_models/lightgbm_models/hashed_imports_detector/hashed_features_model_2017.txt")
    parser.add_argument("--hashed_imports_lightgbm_model_filepath_version2",
                        type=str,
                        help="Hashed imports black-box detector (Version 2)",
                        default="../../../ml_models/lightgbm_models/hashed_imports_detector/hashed_features_model_2018.txt")
    parser.add_argument('--hashed-version1', dest='hashed_version1', action='store_true')
    parser.add_argument('--no-hashed-version1', dest='hashed_version1', action='store_false')
    parser.set_defaults(hashed_version1=False)
    parser.add_argument('--hashed-version2', dest='hashed_version2', action='store_true')
    parser.add_argument('--no-hashed-version2', dest='hashed_version2', action='store_false')
    parser.set_defaults(hashed_version2=False)

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

    parser.add_argument("--malconvgct_model_filepath",
                        type=str,
                        help="MalConvGCT black-box detector (Version 2)",
                        default="../../../ml_models/torch_models/malconvgct_detector/models/malconvGCT_nocat.checkpoint")
    parser.add_argument('--malconvgct', dest='malconvgct', action='store_true')
    parser.add_argument('--no-malconvgct', dest='malconvgct', action='store_false')
    parser.set_defaults(malconvgct=False)

    parser.add_argument("--cuda_device",
                        type=int,
                        help="Cuda device",
                        default=None)

    parser.add_argument("--inception_parameters_filepath",
                        type=str,
                        help="Inception parameters",
                        default="../../../ml_models/torch_models/inception_detector/network_parameters/iat_net_params.json")
    parser.add_argument("--inception_checkpoint",
                        type=str,
                        help="Inception checkpoint",
                        default="../../../ml_models/torch_models/inception_detector/models/imports_model_2017/best/best_model.pt")
    parser.add_argument("--inception_features_filepath",
                        type=str,
                        help="Inception benign intermediate features",
                        default="../../../ml_models/torch_models/inception_detector/intermediate_features/imports_2017.npz")

    parser.add_argument('--wandb', dest='wandb', action='store_true')
    parser.add_argument('--no-wandb', dest='wandb', action='store_false')
    parser.set_defaults(wandb=True)
    args = parser.parse_args()

    malgan = MalGAN()
    malgan.create_logger(args.output_filepath)
    malgan.init_directories(args.output_filepath)
    malgan.init_cuda(args.cuda_device)
    malgan.initialize_vocabulary_mapping(args.vocabulary_mapping_filepath)
    malgan.initialize_inverse_vocabulary_mapping(args.inverse_vocabulary_mapping_filepath)
    malgan.initialize_feature_extractor()
    malgan.initialize_ember_feature_extractors()

    malgan.initialize_training_and_validation_generator_datasets(
        args.malware_imports_features_filepath,
        args.malware_hashed_imports_features_filepath,
        args.malware_imports_filepath,
        args.malware_ember_features_filepath_version1,
        args.malware_ember_features_filepath_version2,
        args.malware_raw_executables_filepath,
        args.training_malware_annotations_filepath,
        args.validation_malware_annotations_filepath
    )
    malgan.initialize_training_and_validation_discriminator_datasets(
        args.goodware_imports_features_filepath,
        args.goodware_hashed_imports_features_filepath,
        args.goodware_imports_filepath,
        args.goodware_ember_features_filepath_version1,
        args.goodware_ember_features_filepath_version2,
        args.goodware_raw_executables_filepath,
        args.training_goodware_annotations_filepath,
        args.validation_goodware_annotations_filepath
    )

    malgan.build_generator_network(args.generator_parameters_filepath)
    malgan.build_discriminator_network(args.discriminator_parameters_filepath)
    malgan.build_blackbox_detectors(
        args.imports_lightgbm_model_filepath_version1,
        args.imports_version1,
        args.imports_lightgbm_model_filepath_version2,
        args.imports_version2,
        args.hashed_imports_lightgbm_model_filepath_version1,
        args.hashed_version1,
        args.hashed_imports_lightgbm_model_filepath_version2,
        args.hashed_version2,
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
        args.nonneg_malconv,
        args.malconvgct_model_filepath,
        args.malconvgct
    )
    inception_parameters = load_json(args.inception_parameters_filepath)
    malgan.load_inception_model(inception_parameters, args.inception_checkpoint)
    malgan.load_inception_benign_intermediate_features(args.inception_features_filepath)

    training_parameters = load_json(args.training_parameters_filepath)
    malgan.init_wandb(training_parameters, args.wandb)
    malgan.train(training_parameters)

    if args.test_malware_annotations_filepath is not None and args.test_goodware_annotations_filepath is not None:
        test_discriminator_dataset = ImportsDataset(
            args.goodware_imports_features_filepath,
            args.goodware_hashed_imports_features_filepath,
            args.goodware_imports_filepath,
            args.goodware_ember_features_filepath_version1,
            args.goodware_ember_features_filepath_version2,
            args.goodware_raw_executables_filepath,
            args.test_goodware_annotations_filepath
        )
        test_generator_dataset = ImportsDataset(
            args.malware_imports_features_filepath,
            args.malware_hashed_imports_features_filepath,
            args.malware_imports_filepath,
            args.malware_ember_features_filepath_version1,
            args.malware_ember_features_filepath_version2,
            args.malware_raw_executables_filepath,
            args.test_malware_annotations_filepath
        )
        malgan.test(test_generator_dataset, training_parameters["batch_size"])