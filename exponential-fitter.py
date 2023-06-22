import os
from pathlib import Path

import optax
import jax.config
jax.config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from jax_md import energy, quantity, space

import exp_energy

NFRAMES=50
AUTOEVPA=jnp.array(51.422067476, dtype='float64')
water_atom_masses = jnp.array([15.999,1.00784,1.00784], dtype='float64')

base_path = "inputdata/"
np.random.seed(123)

jax.config.update("jax_enable_x64", True)

#Reads in xyz files 
def read_input(inputlocation,natoms):
    file_path = inputlocation
    expanded_file_path = Path(file_path).expanduser()
    with expanded_file_path.open('r') as waterstrucfile:
        atomname = np.empty((NFRAMES, natoms), dtype='U')
        data = np.zeros([NFRAMES, natoms, 3], dtype="float64") 
        for i in range(0,NFRAMES):
            waterstrucfile.readline()
            waterstrucfile.readline()
            for j in range(0,natoms):
                line = waterstrucfile.readline().split()
                atomname[i][j]=line[0]
                data[i][j]=list(map(float, line[1:4]))
    return atomname,data


def calculate_molecular_properties(R, forces,species):
    N = forces.shape[1]
    n_wat= np.count_nonzero(species == 0)  # Count the number of zeros in the species array
    remaining_atoms = N-n_wat*3 
    # Reshape forces and position arrays to group atoms belonging to the same molecule
    forces_reshaped = forces[:, :n_wat * 3].reshape(-1, n_wat, 3, 3) 
    R_reshaped = R[:, :n_wat * 3].reshape(-1, n_wat, 3, 3)  

    # Calculate center of mass of each molecule
    center_of_mass = jnp.sum(R_reshaped * water_atom_masses[None, None, :, None], axis=2) / np.sum(water_atom_masses)

    # Sum forces acting on atoms of each molecule
    net_force = jnp.sum(forces_reshaped, axis=2)  # net force on each molecule

    # Calculate net torques for each molecule
    net_torques = jnp.cross(R_reshaped  - center_of_mass[:,:, None, :], forces_reshaped).sum(axis=2)  

    # If there are remaining atoms, append their forces and zeros for torques to the end of net_force and net_torques
    if remaining_atoms > 0:
        extra_forces = forces[:, n_wat * 3:]
        net_force = jnp.concatenate([net_force, extra_forces], axis=1)

    return net_force, net_torques

#computes energy given positions and compares to y_true and computes rmse
def root_mean_squared_error(params,y_true, R,species):
    displacement_fn, shift=space.free()
    energy_fn1 = exp_energy.exponential_potential_pair(displacement_fn,species=species,**params['params1'],r_onset=4, r_cutoff=6.0)
    force_fn1 = quantity.force(energy_fn1)
    fpred1=vmap(force_fn1)(R)
    fpred=fpred1
    true_net_forces, true_torques = calculate_molecular_properties(R, y_true,species)
    pred_net_forces, pred_torques = calculate_molecular_properties(R, fpred,species)
    # Compute and return RMSE for both forces and torques
    force_error = jnp.sqrt(jnp.mean(jnp.square(true_net_forces - pred_net_forces)))
    torque_error = jnp.sqrt(jnp.mean(jnp.square(true_torques - pred_torques)))    
    # Return the root of the sum of squares of the force and torque error
    #print(force_error,torque_error)
    return (force_error + torque_error) / 2

mse_grad = grad(root_mean_squared_error)

#Each set of quantum chemistry calculations is an MolecularSystem object
class MolecularSystem:
    def __init__(self, coords_file, fcc_file, frs_file, species, natoms):
        self.atomnames, self.coords = read_input(os.path.join(base_path, coords_file), natoms)
        _, self.fcc = read_input(os.path.join(base_path, fcc_file), natoms)
        _, self.frs = read_input(os.path.join(base_path, frs_file), natoms)
        self.species = jnp.array(species)
        self.natoms = natoms
        self.prepare_data()
    

    def prepare_data(self, train_ratio=0.8):
        # Shuffle the data randomly
        idx = np.random.permutation(self.coords.shape[0])
        self.coords=self.coords[idx]
        self.fcc=self.fcc[idx]
        self.frs=self.frs[idx]

        #Make arrays of training and validation data. Negative sign for fcc as orca gives grads not forces      
        num_train =int(train_ratio*NFRAMES)
        self.coordstrain = jnp.array(self.coords[0:num_train])
        self.ftargtrain = jnp.array((-self.fcc - self.frs)[0:num_train] * AUTOEVPA)
        self.coordsval = jnp.array(self.coords[num_train:])
        self.ftargval = jnp.array((-self.fcc - self.frs)[num_train:] * AUTOEVPA)

    def calculate_mse(self, params, train=True):
        if train:
            return root_mean_squared_error(params, self.ftargtrain, self.coordstrain, self.species)
        else:
            return root_mean_squared_error(params, self.ftargval, self.coordsval, self.species)
    
    def calculate_grad(self, params):
            return mse_grad(params, self.ftargtrain, self.coordstrain, self.species)
      
def main():
    #Inputing the 3 different sets of cluster calculations. 
    systems = {}
    systems["water"] = MolecularSystem("8-LB-revPBED3-pos-1.xyz",
                                   "H2O8-DZ-MP2-revPBED3struc-frcs.xyz",
                                   "H2O-H2O8-R2SCAN-NN4-LB-frc-1.xyz",    
                                   [0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1],
                                   27)
    systems["Na_water"] = MolecularSystem("8-K-LB-revPBED3-pos-1.xyz",
                                        "K-H2O8-DZ-MP2-revPBED3struc-frcs.xyz",
                                        "K-H2O8-R2SCAN-NN4-LB-frc-1.xyz",
                                        [0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,2],
                                        25)
    systems["Cl_water"] = MolecularSystem("8-Cl-LB-revPBED3-pos-1.xyz",
                                        "Cl-H2O8-DZ-MP2-revPBED3struc-frcs.xyz",                                         
                                        "Cl-H2O8-R2SCAN-NN4-LB-frc-1.xyz",
                                        [0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,3],
                                        25)
    #Can do ion pair clusters as well
    systems["NaCl_water"] = MolecularSystem("8-KCl-LB-revPBED3-pos-1.xyz",
                                        "KCl-H2O8-DZ-MP2-revPBED3struc-frcs.xyz",
                                        "KCl-H2O8-R2SCAN-NN4-LB-frc-1.xyz",  
                                        [0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,2,3],26)
    param_dict = {
         'params1': {'A': jnp.zeros((4, 4), dtype=jnp.float64) , 'B':jnp.ones((4, 4), dtype=jnp.float64)}}
    mse_lossval_best=1.0
#    reg_lambda = 0.00
    num_epochs = 1000
    lr=0.01
    adam = optax.adam(learning_rate=lr)
    state = adam.init(param_dict)
    for epoch in range(num_epochs):
        results = {}
        mse_tot=0.0    
        #computing total error given set of parameters
        for name, system in systems.items():
            mse_losstrain = system.calculate_mse(param_dict, train=True)
            mse_lossval = system.calculate_mse(param_dict, train=False)
            grads = system.calculate_grad(param_dict)
            results[name] = {"mse_losstrain": mse_losstrain, "mse_lossval": mse_lossval, "grads": grads}
            mse_tot += mse_lossval
        #Getting the losses and grads for each system in an array and printing out     
        mse_losstrain_list = [results[name]["mse_losstrain"] for name in systems]
        mse_lossval_list = [results[name]["mse_lossval"] for name in systems]
        grads= [results[name]["grads"] for name in systems]
        # Create a zero array of your desired shape (e.g., 3x3)      
    # Change the [2,3] element to 1
    # Note that Python uses 0-based indexing, so [2,3] in a 1-based system becomes [1,2] in Python         
        total_grads = jax.tree_map(lambda *xs: sum(xs), *grads)
        #You mask some potentials to only selectively optimise some pairs. 
#        mask = jnp.zeros((4,4))
#        mask = mask.at[0, 1].set(1)
#        mask = mask.at[1, 3].set(1)
#        total_grads_masked = {dict_key: {array_key: array_val * mask for array_key, array_val in dict_val.items()} 
#                                for dict_key, dict_val in total_grads.items()}
#        total_grads=total_grads_masked
        print("Epoch: ", epoch)
        print("Params: ")
        print(param_dict)
        print("MSE loss train: ")
        print(mse_losstrain_list)
        print("MSE loss val: ")
        print(mse_lossval_list)
        num_systems = len(systems)
        #Keeping track of lowest error epoch. 
        if mse_tot/num_systems < mse_lossval_best:
            mse_lossval_best = mse_tot/num_systems
            print("Best so far")
            print(mse_lossval_best)
            param_dict_best=param_dict
            epoch_best=epoch
            #updating weights with adam 
        updates, state = adam.update(total_grads, state, param_dict)
        param_dict = optax.apply_updates(param_dict, updates)
        #vannila gradient descent with l2 regulariation alternative: 
        #param_dict = jax.tree_map(lambda p, *grads: p - lr * sum(grads)- 2*lr*reg_lambda*p , param_dict, *grads)
    print("The best epoch was: ", epoch_best,"with an error of:",mse_lossval_best," with the following parameters: ")
    print(param_dict_best)
main()

