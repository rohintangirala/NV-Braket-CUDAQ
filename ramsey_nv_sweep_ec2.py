import cudaq, time, os
import numpy as np
import cupy as cp
from cudaq import spin, ScalarOperator
from cudaq import evolve, Schedule, operators
import matplotlib.pyplot as plt

cudaq.mpi.initialize()
# Set CUDA Q backend to dynamics
cudaq.set_target("dynamics")

# Constants
omega_L = 2*np.pi*1.07e6 # 13C Larmor
A_parallel = 204.9e3 # electron-13C hyperfine coupling
A_perp = 123.3e3 # electron-13C hyperfine coupling
h_bar = 1  # eV/Hz
rabi = 2*np.pi*10e6 # electron Rabi drive
delta = 2*np.pi*1e6 # electron drive detuning

omega_L_N = 2 * np.pi * 0.3e6      # 14N Larmor
A_parallel_N = 2.14e6      # electron-14N hyperfine coupling
A_perp_N = 2.7e6   # electron-14N hyperfine coupling

# Spin-1 operators
Sx1 = (1/np.sqrt(2)) * np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=np.complex128)

Sy1 = (1/np.sqrt(2)) * np.array([
    [0, -1j, 0],
    [1j, 0, -1j],
    [0, 1j, 0]
], dtype=np.complex128)

Sz1 = np.array([
    [1, 0, 0],
    [0, 0, 0],
    [0, 0, -1]
], dtype=np.complex128)

Id1 = np.eye(3, dtype=np.complex128)

def callback_tensor_sx(t):
    return np.array(Sx1, dtype=np.complex128)

def callback_tensor_sy(t):
    return np.array(Sy1, dtype=np.complex128)

def callback_tensor_sz(t):
    return np.array(Sz1, dtype=np.complex128)

operators.define("sx1", [3], callback_tensor_sx)
operators.define("sy1", [3], callback_tensor_sy)
operators.define("sz1", [3], callback_tensor_sz)

# Hilbert space dimensions of subsystems
dimensions = {0: 2, 1: 3} # electron (0), nitrogen (1)
num_carbons = 0
for i in range(num_carbons):
    dimensions[2+i] = 2


electron_state = cp.array([1, 0], dtype=cp.complex128)
nuclear_state = cp.array([1, 0], dtype=cp.complex128)
nitrogen_state = cp.array([1, 0, 0], dtype=cp.complex128)
psi = cp.kron(electron_state, nitrogen_state)

for i in range(num_carbons):
    psi = cp.kron(nuclear_state, psi)
rho = cp.outer(psi, psi.conj())
rho0 = cudaq.State.from_data(rho)

def Iz(qubit):
    return (h_bar / 2) * spin.z(qubit)

def Ix(qubit):
    return (h_bar / 2) * spin.x(qubit)

def Iy(qubit):
    return (h_bar / 2) * spin.y(qubit)

gamma = 1e5 # pure dephasing decay rate (incoherent bath)

# ************************** IMPORTANT PARAMETERS TO CHANGE *************************
max_nuclei = int(os.environ['MAX_NUCLEI'])  # MAX OF CARBON NUCLEI TO SIMULATE IN SWEEP
num_detunings = 3 # NUMBER OF RAMSEY DETUNINGS TO SIMULATE IN EACH EXPERIMENT


print("Number of coherent carbon nuclei:", max_nuclei)
# Hilbert space dimensions of subsystems
dimensions = {0: 2, 1: 3}
num_carbons = max_nuclei
for i in range(num_carbons):
    dimensions[2+i] = 2

psi = cp.kron(electron_state, nitrogen_state)

for i in range(num_carbons):
    psi = cp.kron(nuclear_state, psi)
rho = cp.outer(psi, psi.conj())

rho0 = cudaq.State.from_data(rho)

detunings = np.linspace(-2e6, 2e6, num_detunings)
population_e_dephasing_by_delta = []

init_time = time.time()

for delta in detunings:
    print("delta:", delta)
    hamiltonian = (
        delta/2 * spin.z(0) # detuning term
        + rabi/2 * spin.x(0) # drive term
        +omega_L_N * operators.instantiate("sz1", [1])
        + A_parallel_N*spin.z(0)*operators.instantiate("sz1", [1])
        + A_perp_N*spin.z(0)*operators.instantiate("sx1", [1])
    )

    for i in range(num_carbons):
        hamiltonian = hamiltonian + (
            omega_L*Iz(2+i) # larmor
            + A_parallel*spin.z(0)*Iz(2+i) # hyperfine
            + A_perp*spin.z(0)*Ix(2+i) # hyperfine
        )

    hamiltonian_tau = (
        delta/2 * spin.z(0) # detuning term
        +omega_L_N * operators.instantiate("sz1", [1])
        + A_parallel_N*spin.z(0)*operators.instantiate("sz1", [1])
        + A_perp_N*spin.z(0)*operators.instantiate("sx1", [1])
    )

    for i in range(num_carbons):
        hamiltonian_tau = hamiltonian_tau + (
            omega_L*Iz(2+i) # larmor
            + A_parallel*spin.z(0)*Iz(2+i) # hyperfine
            + A_perp*spin.z(0)*Ix(2+i) # hyperfine
        )

    # Time evolution schedule for initial pi/2 pulse
    t_final = 25e-9
    n_steps = 25
    steps = np.linspace(0, t_final, n_steps)
    schedule = Schedule(steps, ["t"])

    # Time evolution for initial pi/2 pulse
    evolution_result = evolve(
        hamiltonian,
        dimensions,
        schedule,
        rho0,
        observables=[
        ],
        collapse_operators=[np.sqrt(gamma)*spin.z(0)],
        store_intermediate_results=False
    )

    rho1 = evolution_result.final_state()

    # Time evolution schedule
    taus = np.linspace(2e-6/100, 2e-6, 100)
    final_tau_states = []
    #n_steps = 10 * tau

    for i in range(len(taus)):
        t_final = taus[i]  
        n_steps = 10 * t_final / 4e-6/50
        steps = np.linspace(0, t_final, n_steps)
        schedule = Schedule(steps, ["t"])

        evolution_result_tau = evolve(
            hamiltonian_tau,
            dimensions,
            schedule,
            rho1,
            observables=[
            ],
            collapse_operators=[np.sqrt(gamma)*spin.z(0)],
            store_intermediate_results=False
        )

        rho2 = evolution_result_tau.final_state()

        final_tau_states.append(rho2)

    final_population_all_e_dephasing = np.zeros(len(taus))

    get_result = lambda idx, res: [
        exp_vals[idx].expectation() for exp_vals in res.expectation_values()
    ]   

    for i in range(len(taus)):
        # Time evolution schedule
        t_final = 25e-9
        n_steps = 25
        steps = np.linspace(0, t_final, n_steps)
        schedule = Schedule(steps, ["t"])

        evolution_result_final = evolve(
            hamiltonian,
            dimensions,
            schedule,
            final_tau_states[i],
            observables=[
                operators.number(0)
            ],
            collapse_operators=[np.sqrt(gamma)*spin.z(0)],
            store_intermediate_results=True
        )
        final_population = get_result(0, evolution_result_final)[-1]
        final_population_all_e_dephasing[i] = final_population
    population_e_dephasing_by_delta.append(final_population_all_e_dephasing)

cudaq.mpi.finalize()  
final_time = time.time()
time_elapse = final_time - init_time # unit in seconds

import pickle
with open("shared/gpu2_C"+str(max_nuclei)+".pkl", "wb") as file: # "wb" for write binary
    pickle.dump((time_elapse, population_e_dephasing_by_delta), file)


import boto3
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
s3_client = boto3.client("s3")
s3_client.upload_file("shared/gpu2_C"+str(max_nuclei)+".pkl", 'pcluster-custom-actions-891612555742-us-west-2', "gpu2_C"+str(max_nuclei)+".pkl")


print('Calculations finished!')
