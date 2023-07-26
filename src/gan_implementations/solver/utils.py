import numpy as np
import copy
import subprocess
import csv


def calculate_histogram(sequence:list):
    histogram = np.array([0 for i in range(256)])
    for x in sequence:
        histogram[x] += 1
    return histogram

def calculate_normalized_histogram(sequence:list):
    histogram = np.array([0 for i in range(256)])
    for x in sequence:
        histogram[x] += 1
    return histogram / len(sequence)

def append_bytes_given_target_histogram(byte_sequence, original_histogram, target_histogram):
    target_byte_sequence = copy.deepcopy(byte_sequence)
    for i in range(original_histogram.shape[0]):
        diff = target_histogram[i] - original_histogram[i]
        for j in range(diff):
            target_byte_sequence.append(i)
    return target_byte_sequence

def append_bytes_given_target_bytes(original_byte_sequence: list, bytes_to_append: tuple):
    target_byte_sequence = copy.deepcopy(original_byte_sequence)
    for tup in bytes_to_append:
        for j in range(tup[1]):
            target_byte_sequence.append(tup[0])
    return target_byte_sequence

def heuristic_approach(original_byte_sequence, original_normalized_histogram, target_normalized_histogram):
    original_histogram = calculate_histogram(original_byte_sequence)
    target_length = int(max([original_histogram[i]/target_normalized_histogram[i] for i in range(original_histogram.shape[0])]))
    resulting_histogram = np.array([int(target_length*target_normalized_histogram[i])  for i in range(original_histogram.shape[0])])
    return resulting_histogram, resulting_histogram / target_length

def create_byte_sizes_file(byte_histogram: np.array, output_filepath: str):
    with open(output_filepath, "w") as output_file:
        for i, value in enumerate(byte_histogram.tolist()):
            output_file.write("{} {}\n".format(i, value))

def create_ratios_file(normalized_byte_histogram: np.array, output_filepath: str):
    with open(output_filepath, "w") as output_file:
        for i, value in enumerate(normalized_byte_histogram.tolist()):
            output_file.write("{} {}\n".format(i, value))

def run_solver(gap=0.001):
    print("Gap: ", gap)
    if gap == 0.01:
        result = subprocess.run(
            [
                "sh",
                "run-solver-gap0.01.sh"
            ]
        )
    if gap == 0.008:
        result = subprocess.run(
            [
                "sh",
                "run-solver-gap0.008.sh"
            ]
        )
    if gap == 0.005:
        result = subprocess.run(
            [
                "sh",
                "run-solver-gap0.005.sh"
            ]
        )
    if gap == 0.003:
        result = subprocess.run(
            [
                "sh",
                "run-solver-gap0.003.sh"
            ]
        )
    if gap == 0.001:
        print("Entering here!")
        result = subprocess.run(
            [
                "sh",
                "run-solver-gap0.001.sh"
            ]
        )
    if gap == 0.0008:
        result = subprocess.run(
            [
                "sh",
                "run-solver-gap0.0008.sh"
            ]
        )
    if gap == 0.0005:
        result = subprocess.run(
            [
                "sh",
                "run-solver-gap0.0005.sh"
            ]
        )
    if gap == 0.0003:
        result = subprocess.run(
            [
                "sh",
                "run-solver-gap0.0003.sh"
            ]
        )
    if gap == 0.0001:
        result = subprocess.run(
            [
                "sh",
                "run-solver-gap0.0001.sh"
            ]
        )
    print(result.returncode)

def read_solution(solution_filepath: str):
    solution = []
    with open(solution_filepath, "r") as solution_file:
        reader = csv.reader(solution_file, delimiter='\t')
        for row in reader:
            #print((int(row[0].split("#")[-1]), int(float(row[1]))))
            solution.append((int(row[0].split("#")[-1]), int(float(row[1]))))
    return solution




