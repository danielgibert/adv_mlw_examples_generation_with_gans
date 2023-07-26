import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
import numpy as np


def load_json(filepath):
    with open(filepath, "r") as input_file:
        data = json.load(input_file)
    return data

def check_collisions_among_generated_samples(generated_samples):
    colls = []
    for i in range(generated_samples.shape[0]):
        sample_collisions = np.bitwise_xor(generated_samples[i], generated_samples)
        collisions = np.sum(sample_collisions, axis=1)
        #print("Sample: {}; Collisions: {}".format(i, collisions))
        colls.append(np.sum(collisions) / collisions.shape[0])
        #print("Average collisions: {}".format(colls[-1]))
    return np.array(colls)

def plot_generator_and_discriminator_training_loss(G_losses, D_losses, output_filepath):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_filepath)
    #plt.show()

def plot_generator_and_discriminator_validation_loss(G_losses, D_losses, output_filepath):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_filepath)
    #plt.show()

def plot_average_imported_functions(original_average_functions, fake_average_functions, output_filepath):
    plt.figure(figsize=(10, 5))
    plt.title("Average Imported Functions")
    plt.plot(original_average_functions, label="Original imported functions")
    plt.plot(fake_average_functions, label="Fake imported functions")
    plt.xlabel("Evaluation steps")
    plt.ylabel("Avg. imported functions")
    plt.legend()
    plt.savefig(output_filepath)
    #plt.show()

def plot_average_collisions(average_collisions, output_filepath):
    plt.figure(figsize=(10, 5))
    plt.title("Average Collisions")
    plt.plot(average_collisions, label="Avg")
    plt.xlabel("Evaluation steps")
    plt.ylabel("Avg. collisions")
    plt.legend()
    plt.savefig(output_filepath)
    #plt.show()

def plot_frechlet_inception_distance(frechlet_distances, output_filepath):
    plt.figure(figsize=(10, 5))
    plt.title("Frechlet Inception Distances")
    plt.plot(frechlet_distances, label="FID")
    plt.xlabel("Evaluation steps")
    plt.ylabel("FID")
    plt.legend()
    plt.savefig(output_filepath)
    #plt.show()

def plot_evasion_rate(evasion_rates, title, output_filepath):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(evasion_rates, label="Evasion rate")
    plt.xlabel("Evaluation epochs")
    plt.ylabel("Evasion rate")
    plt.legend()
    plt.savefig(output_filepath)
    #plt.show()

def plot_distance_metric(distances, title, output_filepath):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(distances, label="Distance")
    plt.xlabel("Evaluation steps")
    plt.ylabel("Distance")
    plt.legend()
    plt.savefig(output_filepath)
    #plt.show()