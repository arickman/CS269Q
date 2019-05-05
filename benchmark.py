from pyquil import Program, get_qc
from pyquil.api import QVMConnection
from pyquil.gates import *
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

NANO_SECOND_DENOMINATOR = 1000000000.0
NUM_STEPS = 100
NUM_TRIALS = 1000
OMEGA_D = 1e6

qvm = QVMConnection()

T1_target_benchmark = {0: 10e-6, 1: 10e-6, 2:14e-6, 3: 1e-6, 4: 5e-6}
T2_target_benchmark = {0: 2e-6, 1: 10e-6, 2:14e-6, 3: 0.5e-6, 4: 5e-6}

qubits = [0, 1, 2, 3, 4]

p0_0 = (0.5, 1/2e-6, np.pi / (2 * OMEGA_D), 0.5)
p0_1 = (0.5, 1/8e-6, np.pi / (2 * OMEGA_D), 0.5)
p0_2 = (0.45, 1/8e-6, np.pi / (2 * OMEGA_D), 0.5)
p0_3 = (0.5, 1/0.5e-6, np.pi / (2 * OMEGA_D), 0.5)
p0_4 = (0.45, 1/1e-6, np.pi / (2 * OMEGA_D), 0.5)
guesses_T2 = [p0_0, p0_1, p0_2, p0_3, p0_4]

# Be sure to rename this file to benchmark.py or your solution will not be autograded properly

# THESE NEXT TWO LINES ARE NEEDED WHEN YOU SUBMIT YOUR PROJECT
# they are what activate the QVM in the background that your code runs against.
# Feel free to comment them out for local testing, but be sure to enable them when submitting
import subprocess
subprocess.Popen("/src/qvm/qvm -S > qvm.log 2>&1", shell=True)

# define type of function to search
def func_t1(x, a, k, b):
    return a * np.exp(-k*x) + b

def func_t2(t, a, k, o, c):
    return (a * np.exp(-2*k*t)) * np.sin(OMEGA_D * (t - o)) + c

def get_model(filename):
    with open(filename, 'r') as file:
        model_info = file.read()
        return model_info

def run_t1(qubit, filename):
    qvm = QVMConnection()
    # qc = get_qc('9q-square-qvm')
    model_info = get_model(filename)
    noisy_p = Program("NOISY-RX-PLUS-180 " + str(qubit))
    noisy_p = Program(model_info) + noisy_p
    probabilities = []
    x = []
    delay_duration = 0.0 / NANO_SECOND_DENOMINATOR
    for t in range(NUM_STEPS): # each loop is 50 ns
        noisy_p_measured = noisy_p.copy().measure_all()
        result = np.sum(qvm.run(noisy_p_measured, trials=NUM_TRIALS))
        # if (t % 10 == 0):
        #     print("T: {} Noisy Result: {}".format(t, result))
        probability = float(result) / NUM_TRIALS
        probabilities.append(probability)
        x.append(delay_duration)
        noisy_p = noisy_p + Program(("NOISY-I " + str(qubit))) # apply noisy I every single loop
        delay_duration += 50.0 / NANO_SECOND_DENOMINATOR
    
    y_np = np.array(probabilities)
    x_np = np.array(x)
    plt.plot(x_np, y_np)
    opt, pcov = curve_fit(func_t1, x_np, y_np, maxfev=1000000,
        p0=(0.5, 1.0/1e-6, 0.5))
    plt.plot(x_np, func_t1(x_np, *opt), 'r--')
    plt.show()
    a, k, b = opt
    return 1.0 / k

def get_score(target, benchmarks):
    total_score = 0.0
    for qubit in benchmarks:
        curr_score = max((-5.0/(target[qubit] ** 2)) * (abs(\
            benchmarks[qubit] - target[qubit]) ** 2) + 5, 0)
        print("Qubit: {}, Score: {}".format(qubit, curr_score))
        total_score += curr_score
    return total_score

# you are to complete this function
def benchmark_T1(filename):
    # ...
    # the return type is a dictionary whose keys are qubit ids and whose value is the T1 in seconds
    t1_dict = {}
    for qubit in qubits:
        cur_t1 = run_t1(qubit, filename)
        t1_dict[qubit] = cur_t1
    print(t1_dict)
    print(get_score(T1_target_benchmark, t1_dict))
    return t1_dict

def run_t2(qubit, filename):
    qvm = QVMConnection()
    model_info = get_model(filename)
    # noisy_p = Program(RX(math.pi / 2, qubit))
    noisy_p =  Program("NOISY-RX-PLUS-90 " + str(qubit))
    noisy_p = Program(model_info) + noisy_p
    probabilities = []
    x = []
    delay_duration = 0.0 / NANO_SECOND_DENOMINATOR
    for t in range(NUM_STEPS):
        noisy_p_measured = noisy_p.copy()
        noisy_p_measured += Program(RZ(delay_duration * OMEGA_D, qubit))
        noisy_p_measured += Program("NOISY-RX-PLUS-90 " + str(qubit))
        noisy_p_measured = noisy_p_measured.measure_all()
        result = np.sum(qvm.run(noisy_p_measured, trials=NUM_TRIALS))
        # if (t % 10 == 0):
        #     print("T: {} Noisy Result: {}".format(t, result))
        probability = float(result) / NUM_TRIALS
        probabilities.append(probability)
        x.append(delay_duration)
        noisy_p += Program("NOISY-I " + str(qubit))
        delay_duration += 50.0 / NANO_SECOND_DENOMINATOR

    y_np = np.array(probabilities)
    x_np = np.array(x)
    plt.plot(x_np, y_np)
    opt, pcov = curve_fit(func_t2, x_np, y_np, maxfev=1000000,
        p0=guesses_T2[qubit])
    plt.plot(x_np, func_t2(x_np, *opt), 'r--')
    plt.show()
    a, k, o, c = opt
    print(opt)
    return 1.0 / k

# you are to complete this function
def benchmark_T2(filename):
    # ...
    # the return type is a dictionary whose keys are qubit ids and whose value is the T2 in seconds
    t2_dict = {}
    for qubit in qubits:
        cur_t2 = run_t2(qubit, filename) 
        t2_dict[qubit] = cur_t2
    print(t2_dict)
    print(get_score(T2_target_benchmark, t2_dict))
    return t2_dict

if __name__ == "__main__":
    #benchmark_T1("noise_model.quil")
    benchmark_T2("noise_model.quil")