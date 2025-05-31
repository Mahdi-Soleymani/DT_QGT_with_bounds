import torch
import sys
import time
sys.path.append("mingpt/")  # Adjust if needed
import mingpt.trainer_QGT as t
# Import your model class
#from model_QGT import model
import numpy as np
import random
from gurobipy import GRB, LinExpr
from torch.nn import functional as F
import gurobipy as gp

import matplotlib.pyplot as plt
import os



    
from mingpt.model_QGT import DecisionTransformer as QGT_model  # Alias your custom model
from gurobipy import Model as GurobiModel  # Alias Gurobi's Model


def random_integer_vector(k):
    x = np.zeros(k, dtype=int) # Initialize a zero vector
    x_half = np.zeros(k, dtype=int) # Initialize a zero vector
    for i in range(k):
        id=np.random.choice(k,1)
        x[id]+=1
        if random.random() < 0.5:
            x_half[id]+=1
    return x.reshape(-1,1),x_half.reshape(-1,1)

def pad_sequence(seq, max_len, pad_value=0):
    """Pads a sequence to max_len with pad_value"""

    seq = torch.tensor(seq, dtype=torch.float32)  # Convert to tensor



    pad_size = max_len - seq.shape[0]

    if pad_size > 0:
        zero_vector = pad_value*torch.ones(pad_size)
        seq = torch.cat((seq, zero_vector))

    return seq


def pad_sequence2d(seq, max_len, pad_value=0):
    """Pads a batch of sequences to max_len with pad_value"""

    # Convert the list of lists into a tensor
    #seq = [torch.tensor(q, dtype=torch.float32) for q in seq]  # Convert each query to a tensor
    seq = [q.clone().detach().to(dtype=torch.float32) for q in seq]
    # Stack into a 2D tensor (batch_size, seq_len)
    seq = torch.stack(seq)  # Shape: (batch_size, query_length)
    
    pad_size = max_len - seq.shape[0]
    
    if pad_size > 0:
        seq = F.pad(seq, (0, 0, 0, pad_size), value=pad_value)  # Pad along sequence dimension
    
    return seq



def test_sample(desired_num_of_queries, k, checkpoint_path, mode):
    # Initialize the model and config
    #mode="random"
    #mode="DT"
    #sampling="soft"
    #sampling="c"
    c=0
    sampling="hard"
    config = t.TrainerConfig(
        k=10,
        query_dim=k,
        lr=3e-4,
        max_epochs = 2,
        batch_size = 1,
        learning_rate = 3e-4,
        betas = (0.9, 0.95),
        grad_norm_clip = 1.0,
        weight_decay = 0.1,
        lr_decay = False,
        warmup_tokens = 375e6,
        final_tokens = 260e9,
        ckpt_path=checkpoint_path,  # Set a valid path if you want to save checkpoints
        num_workers=0,
        rtg_dim=1,
        n_embd=512,
        query_result_dim=1,
        block_size=10,### number of max timesteps in sequence (seq len=3 times this)
        embd_pdrop = 0.1,
        n_layer=6,
        n_head=8,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        pad_scalar_val=-100,
        pad_vec_val=-30,
        desired_num_of_queries=8,
        upper_bound_dim=10
    )
    config.k=k
    config.query_dim=config.k
    config.upper_bound_dim=config.k
    config.desired_num_of_queries=desired_num_of_queries

    # Initialize your model architecture (it should be the same as during training)
    DT_model = QGT_model(config)  # Use the same configuration used during training
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    # Load the saved model checkpoint  
    
    #checkpoint = torch.load("comic-mountain-67.pth",  map_location=torch.device("cpu"))
    #checkpoint = torch.load("zany-hill-68.pth",  map_location=torch.device("cpu"))
    #checkpoint = torch.load("misunderstood-serenity-69.pth",  map_location=torch.device("cpu"))
    #checkpoint = torch.load("morning-vortex-71.pth",  map_location='cpu', weights_only=True)
    #checkpoint = torch.load("dulcet-field-88.pth",  map_location='cpu', weights_only=True)
    #checkpoint = torch.load("bright-surf-92.pth",  map_location='cpu', weights_only=True)
    #checkpoint = torch.load("celestial-terrain-94.pth",  map_location='cpu', weights_only=True)
    #checkpoint = torch.load("eternal-feather-87.pth",  map_location='cpu', weights_only=True)
    #checkpoint = torch.load("peach-paper-102.pth",  map_location='cpu', weights_only=True) #k=10
    #checkpoint = torch.load("eternal-voice-4.pth",  map_location='cpu', weights_only=True) #k=5
    #checkpoint = torch.load("giddy-bee-1.pth",  map_location='cpu', weights_only=True) #k=4
    #checkpoint = torch.load("colorful-eon-1.pth",  map_location='cpu', weights_only=True) #k=3
    #checkpoint = torch.load("deep-darkness-1.pth",  map_location='cpu', weights_only=True)  #k=2
    #checkpoint = torch.load("desert-vortex-1.pth",  map_location='cpu', weights_only=True) #k=6
    #checkpoint = torch.load("grateful-fire-1.pth",  map_location='cpu', weights_only=True) #k=7
    #checkpoint = torch.load("volcanic-dawn-1.pth",  map_location='cpu', weights_only=True) #k=8
    #checkpoint = torch.load("revived-feather-12.pth",  map_location=device, weights_only=True) #k=8
    #checkpoint = torch.load("northern-smoke-8.pth",  map_location='cpu', weights_only=True) #k=8
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    
    
    # Load the model weights directly from the checkpoint
    DT_model.load_state_dict(checkpoint)

    # Set the model to evaluation mode
    DT_model.eval()


    max_len = config.k  # Set max length
    pad_scalar_val=config.pad_scalar_val
    pad_vec_val=config.pad_vec_val



    q, r, rtg, mask_length= [ torch.full((config.k,), pad_vec_val, dtype=torch.int)],[config.k],[-config.desired_num_of_queries], 1   # Generate a sequence
    queries=(pad_sequence2d(q, max_len,pad_vec_val))  # Pad queries
    results=(pad_sequence(r, max_len,pad_scalar_val))
    rtgs=(pad_sequence(rtg, max_len,pad_scalar_val))


    mask_length = torch.tensor(mask_length,device=device)
    results = results.to(device)
    rtgs    = rtgs.to(device)
    queries = queries.to(device)






    x,x_half=random_integer_vector(config.k)
    x_half_tensor=torch.tensor(x_half,dtype=torch.float32,device=device)
    G_model = GurobiModel("Incremental_ILP")



    #### to write nothing in the log 
    G_model.setParam(GRB.Param.OutputFlag, 0)
    # Create a list to store the variables for ILP
    variables = []

    # Add variables dynamically
    for i in range(0, config.k):
        variables.append(G_model.addVar(vtype=GRB.INTEGER, lb=0, ub=int(x[i].item()), name=f"x{i}"))


    # ###initial constraint
    # G_model.addConstr(gp.quicksum(variables) == config.k, name="sum_constraint")

    ### Enables solutions pool
    G_model.setParam(GRB.Param.PoolSearchMode, 2)

    # Set the objective (e.g., maximize x + y)
    G_model.setObjective(1 , GRB.MAXIMIZE)


    num_of_constraints=0
    is_solved=False

    rtgs = rtgs.unsqueeze(0)  # Adds batch dimension, result shape: [1, 10]
    results = results.unsqueeze(0)  # Adds batch dimension, result shape: [1, 10]
    queries = queries.unsqueeze(0)
    mask_length = mask_length.unsqueeze(0)  # Adds batch dimension, result shape: [1, 10, 10]

    while not is_solved and num_of_constraints<5 :

        with torch.no_grad():  # No need to track gradients during inference
            if mode=="DT":
                
                ### from model
                upper_bounds = torch.tensor(x, dtype=torch.float32).to(device)
                upper_bounds = upper_bounds.unsqueeze(0)  # Shape: (1, k)

                probs,_=DT_model( mask_length, rtgs,  results, upper_bounds, queries)
                if num_of_constraints<config.k:
                    probs=probs[:,num_of_constraints,:]
                else:
                    probs=probs[:,config.k-1,:]

            elif mode=="random":
            ######## Random queries
                probs=.5*torch.ones(config.batch_size,config.k).float()
                #probs = torch.randint(0, 2, (config.batch_size, config.block_size, config.k)).float()

        

        ###Sampling (soft)
        if sampling=="soft":
            next_query = torch.bernoulli(probs).to(device)
       
        elif sampling=="c":
        #thresholded Bernoulli sampler with a "certainty margin" c.
        
            samples = torch.bernoulli(probs)
            next_query = torch.where(
                probs> (1 - c), torch.ones_like(probs),           # confident 1
                torch.where(
                    probs < c, torch.zeros_like(probs),            # confident 0
                    samples                                        # otherwise: sample
                ))

        elif sampling=="hard":
        ### hard thresholding
            next_query = (probs > 0.5).float()

        next_query=next_query[0,:]
        num_of_constraints+=1

        if num_of_constraints<config.k:
            queries[:,num_of_constraints,:]=next_query
        else:
            next_query = next_query.to(queries.device)
            #queries = torch.cat([queries[:, 1:, :], next_query.unsqueeze(1)], dim=1)
            queries = torch.cat([queries[:, 1:, :], next_query.view(1, 1, -1).expand(queries.size(0), 1, -1)], dim=1)



        selected_variables=[]
        for i in range(config.k):
            if next_query[i]==1:
                selected_variables.append(variables[i])
        next_query = next_query.to(x_half_tensor.device)
        new_result=torch.matmul(next_query,x_half_tensor)
        constraint = sum(selected_variables) == new_result.item()
        new_result = new_result.to(results.device)

        # Add the new constraint

    
        G_model.addConstr(constraint, name=f"{num_of_constraints}")
        
 
    
        
        # Optimize the initial model
        G_model.optimize()

        # Check the initial solution
        if G_model.status == GRB.OPTIMAL:
            # Get the total number of sn        olutions
            num_of_solutions=G_model.SolCount
            if num_of_solutions<=1:
                is_solved=True
            else:
                if num_of_constraints<config.k:
                    rtgs[:,    num_of_constraints]=min(-1,-config.desired_num_of_queries+num_of_constraints)
                    results[:,num_of_constraints]=new_result
                    mask_length[:,]=num_of_constraints+1
                else:
                    rtgs = torch.cat([rtgs[:, 1:], torch.full((rtgs.size(0), 1), -1, device=device)], dim=1)
                    results = torch.cat([results[:, 1:], new_result.unsqueeze(1)], dim=1)
                    mask_length[:,]=config.k-1

                
        else:
            print(f"No solution found!")
        
    


    return num_of_constraints, is_solved



def run_test_sample(des_len, k, checkpoint, mode,_):
    return test_sample(des_len, k, checkpoint, mode)



def main():
    import concurrent.futures
    from tqdm import tqdm
    import argparse
    from functools import partial


    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations")
    parser.add_argument("--num_cores", type=int, default=6, help="Number of CPU cores to use")
    parser.add_argument("--des_len", type=int, default=6, help="Number of CPU cores to use")
    parser.add_argument("--k", type=int, default=10, help="k")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, choices=["random", "DT"], default="DT", help="Mode of querying")

    args = parser.parse_args()

    worker_fn = partial(run_test_sample, args.des_len, args.k, args.checkpoint, args.mode)

    inputs = [args.des_len, args.k] * args.num_iter  # 👈 make it iterable!


    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_cores) as executor:
        results = list(tqdm(executor.map(worker_fn,range(args.num_iter)), total=args.num_iter))

    # results=[]
    # for l in range(1000):
    #     results.append(test_sample())
    #     print(l)

    numbers, flags = zip(*results)
    result=np.array(numbers)
    print(f"des_len:{args.des_len}")
    print(result.mean())
    print(result.std())
    print(sum(flags))
    return result.mean()


if __name__ == "__main__":

    main()



    



        
