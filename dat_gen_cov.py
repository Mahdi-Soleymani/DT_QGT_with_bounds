import numpy as np
import h5py
import random
from gurobipy import Model, GRB
from tqdm import tqdm
import argparse
import os
import multiprocessing as mp

def pad_sequence(seq, max_len, pad_value=0):
    seq = np.array(seq, dtype=np.int8)
    pad_size = max_len - seq.shape[0]
    if pad_size > 0:
        seq = np.pad(seq, (0, pad_size), mode='constant', constant_values=np.int8(pad_value))
    return seq

def pad_sequence2d(seq, max_len, pad_value=0):
    if len(seq) == 0:
        return np.full((max_len, 0), pad_value, dtype=np.int8)
    seq = [np.array(q, dtype=np.int8) for q in seq]
    num_sequences = len(seq)
    seq_length = len(seq[0])
    if num_sequences < max_len:
        pad_matrix = np.full((max_len - num_sequences, seq_length), int(pad_value), dtype=np.int8)
        seq = np.vstack((seq, pad_matrix))
    return seq

def generate_covariance_maximizing_sample(k, max_len, pad_scalar_val, pad_vec_val):
    x = np.zeros(k, dtype=int)
    x_half = np.zeros(k, dtype=int)
    for i in range(k):
        idx = np.random.choice(k, 1)
        x[idx] += 1
        if random.random() < 0.5:
            x_half[idx] += 1

    x = x.reshape(-1, 1)
    x_half = x_half.reshape(-1, 1)

    model = Model("CovMax_ILP")
    model.setParam(GRB.Param.OutputFlag, 0)
    model.setParam(GRB.Param.Threads, 1)

    variables = [model.addVar(vtype=GRB.INTEGER, lb=0, ub=int(x[i].item()), name=f"x{i}") for i in range(k)]
    model.setParam(GRB.Param.PoolSearchMode, 2)
    model.setObjective(1, GRB.MAXIMIZE)
    model.optimize()

    q, r, rwrd = [], [], [-1]
    num_of_constraints = 0
    is_solved = False

    while not is_solved:
        num_solutions = model.SolCount
        if num_solutions < 2:
            break
        solution_matrix = np.zeros((num_solutions, k))
        for sol_index in range(num_solutions):
            model.setParam(GRB.Param.SolutionNumber, sol_index)
            solution_matrix[sol_index] = [var.Xn for var in variables]
        cov_matrix = np.cov(solution_matrix, rowvar=False)

        model_cov = Model("Maximize_Variance")
        model_cov.setParam(GRB.Param.OutputFlag, 0)
        I = model_cov.addVars(k, vtype=GRB.BINARY, name="I")
        quad_expr = sum(I[i] * cov_matrix[i, j] * I[j] for i in range(k) for j in range(k))
        model_cov.setObjective(quad_expr, GRB.MAXIMIZE)
        model_cov.optimize()

        selected_indices = [i for i in range(k) if I[i].X > 0.5]
        selected_mask = np.zeros(k, dtype=int)
        for i in selected_indices:
            selected_mask[i] = 1
        q.append(selected_mask)
        new_result = np.matmul(selected_mask, x_half)[0]
        r.append(int(new_result))

        model.addConstr(sum(variables[i] for i in selected_indices) == new_result, name=f"c{num_of_constraints}")
        num_of_constraints += 1
        model.optimize()

        if model.status == GRB.OPTIMAL:
            if model.SolCount <= 1:
                is_solved = True
                rwrd.append(0)
            else:
                rwrd.append(-1)
        else:
            rwrd.append(0)
            is_solved = True

    rtg, s = [], 0
    for reward in reversed(rwrd):
        s += reward
        rtg.append(s)
    rtg = list(reversed(rtg))
    mask_length = min(len(rtg), max_len)
    q_padded = pad_sequence2d(q[:max_len], max_len, pad_vec_val)
    r_padded = pad_sequence(r[:max_len], max_len, pad_scalar_val)
    rtg_padded = pad_sequence(rtg[:max_len], max_len, pad_scalar_val)
    return q_padded, r_padded, rtg_padded, np.int8(mask_length), np.squeeze(x)

def save_covmax_dataset(file_name, num_samples, k, max_len, pad_scalar_val, pad_vec_val):
    with h5py.File(file_name, 'w') as f:
        d_queries = f.create_dataset("queries", (num_samples, max_len, k), dtype='i1')
        d_results = f.create_dataset("results", (num_samples, max_len), dtype='i1')
        d_rtgs = f.create_dataset("rtgs", (num_samples, max_len), dtype='i1')
        d_mask_lengths = f.create_dataset("mask_lengths", (num_samples,), dtype='i1')
        d_bounds = f.create_dataset("upper_bounds", (num_samples, k), dtype='i1')

        for i in tqdm(range(num_samples), desc="Generating dataset"):
            q, r, rtg, mask_length, d_bound = generate_covariance_maximizing_sample(k, max_len, pad_scalar_val, pad_vec_val)
            d_queries[i] = q
            d_results[i] = r
            d_rtgs[i] = rtg
            d_mask_lengths[i] = mask_length
            d_bounds[i] = d_bound

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", type=int, default=1, help="Number of CPU cores to use")
    parser.add_argument('--num_samples', type=int, default=100, help='Total number of samples to generate')
    parser.add_argument('--file_name', type=str, default="dataset", help='Name of the output file')
    parser.add_argument("--k", type=int, default=10, help="Length of the query vector")

    args = parser.parse_args()
    k = args.k
    max_len = k
    pad_scalar_val = -10
    pad_vec_val = -30
    f_name = f"{args.file_name}_k{k}.h5"

    save_covmax_dataset(f_name, args.num_samples, k, max_len, pad_scalar_val, pad_vec_val)

    with h5py.File(f_name, 'r') as f:
        num_samples = f['queries'].shape[0]
        print(f"Number of samples in {f_name}: {num_samples}")

    file_size_mb = os.path.getsize(f_name) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
