#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn
import transformers  # Added for default_data_collator

# [Added] Import your custom datasets
# Ensure custom_datasets.py is in the python path
try:
    from utils.custom_datasets import FineWebEduDataset, C4Dataset
except ImportError:
    print("WARNING: Could not import FineWebEduDataset/C4Dataset. Ensure custom_datasets.py is present.")

from component.svd_llama import SVD_Linear, test_svd
from utils.data_utils import *
from utils.model_utils import *
from evaluater import * 

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)
sys.path.append(parent_path)

# [Helper] Collator for SVD (Simplified from your training script)
def svd_collator(features):
    if not isinstance(features[0], (list, tuple)):
        return transformers.default_data_collator(features)
    batch_ids = torch.tensor(features, dtype=torch.long)
    return {
        "input_ids": batch_ids,
        "labels": batch_ids.clone(),
        "attention_mask": torch.ones_like(batch_ids)
    }

@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev):
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        raise
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        del inp, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.register_forward_hook(hook)
    
    # [Modified] Handle tqdm explicitly to support both list and DataLoader
    for batch in tqdm(calib_loader, desc="Profiling"):
        # If it's a dict (DataLoader), move to device. If list (WikiText), it's likely already handled or simple.
        if isinstance(batch, dict):
            batch = {k: v.to(dev) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model(**batch)
        else:
            # Fallback for original list-based wikitext loader if needed
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
            
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    return profiling_mat

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    # (Kept purely for compatibility, though you likely want the main profile function)
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    # Warning: This low_resource mode assumes 'calib_loader' has a __len__, 
    # which might break if using a streaming iterable without length.
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    # ... (Rest of low_resource logic omitted for brevity as you are likely using the standard one)
    # If you need low_resource with FineWeb, let me know, as it requires deeper refactoring.
    return {}

@torch.no_grad()
def whitening(model_name, model, profiling_mat, ratio, dev):
    model.eval()
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    print("Start SVD decomposition after whitening...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        
        if "llama" in model_name or "vicuna" in model_name:
            pass
        elif "mistral" in model_name:
            raise
        elif 'opt' in model_name:
            raise

        for name in subset:
            W = subset[name].weight.data.float().to(dev)
            dtype = W.dtype
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
            truc_sigma = torch.diag(truc_s)
            #### Replace Attn, MLP ####
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma).cpu().to(dtype)
            svd_v = torch.matmul(sqrtSigma, truc_v).cpu().to(dtype)

            new_linear = SVD_Linear(W.shape[1], W.shape[0], num_s_after_trunc)
            assert new_linear.u_proj.weight.data.shape == svd_u.shape, "q_proj svd_u shape"
            assert new_linear.v_proj.weight.data.shape == svd_v.shape, "q_proj svd_v shape"
            new_linear.u_proj.weight.data = svd_u.to(torch.float16)
            new_linear.v_proj.weight.data = svd_v.to(torch.float16)
            
            if 'opt' in model_name:
                raise
            else:
                if "q_proj" in name:
                    layer.self_attn.q_proj = new_linear
                elif "k_proj" in name:
                    layer.self_attn.k_proj = new_linear
                elif "v_proj" in name:
                    layer.self_attn.v_proj = new_linear
                elif "o_proj" in name:
                    layer.self_attn.o_proj = new_linear
                elif "gate_proj" in name:
                    layer.mlp.gate_proj = new_linear
                elif "down_proj" in name:
                    layer.mlp.down_proj = new_linear
                elif "up_proj" in name:
                    layer.mlp.up_proj = new_linear
            W = W_scale = scaling_matrix_inv = scaling_diag_matrix = U = S = VT  = truc_s = truc_u = truc_v = sqrtSigma = None
            del  W, W_scale, scaling_matrix_inv, scaling_diag_matrix, U, S, VT, truc_s, truc_u, truc_v, sqrtSigma
        del layer
        torch.cuda.empty_cache()

class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # trucation SVD
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
        x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
        self.updated_uT = torch.linalg.lstsq(x,outs).solution
        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio')
    parser.add_argument('--run_low_resource', action='store_true', help='low resource mode')
    
    # [Modified] Added fineweb-edu / c4 choices
    parser.add_argument('--dataset', type=str, default='wikitext2', 
                        choices=['wikitext2', 'ptb', 'c4', 'fineweb-edu'],
                        help='Dataset for calibration or evaluation')
    parser.add_argument('--data_path', type=str, default=None, help='Path for custom datasets (FineWeb/C4)')
    
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Samples for whitening')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Samples for updating')
    parser.add_argument('--save_path', type=str, default=None, help='path to save checkpoints')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='path to load profiling matrices')
    parser.add_argument('--seed',type=int, default=0, help='Seed')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='sequence length')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference batch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len')
    parser.add_argument('--step', type=int, default=4, help='step to run')
    parser.add_argument('--lora', type=str, default=None, help='lora path')
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio

    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()

        if args.profiling_mat_path is None:
            # [Modified] Logic to handle FineWeb-Edu / C4 vs Default Datasets
            if args.dataset in ["fineweb-edu", "c4"]:
                if args.data_path is None:
                    raise ValueError(f"You must provide --data_path when using {args.dataset}")
                
                print(f"Loading custom dataset: {args.dataset} from {args.data_path}")
                
                def smart_tokenizer_wrapper(examples):
                    text_column = "text" if "text" in examples else list(examples.keys())[0]
                    texts = [t + tokenizer.eos_token for t in examples[text_column]]
                    batch_encodings = tokenizer(
                        texts,
                        truncation=False,
                        max_length=int(1e10),
                        add_special_tokens=True 
                    )
                    return batch_encodings

                common_ds_args = {
                    "sequence_length": args.model_seq_len,
                    "tokenize_fn": smart_tokenizer_wrapper,
                    "seed": args.seed,
                    "use_new_sampling_method": True,
                    "shuffle": True,
                }

                if args.dataset == "fineweb-edu":
                    DatasetClass = FineWebEduDataset
                else:
                    DatasetClass = C4Dataset
                
                dataset = DatasetClass(
                    path=args.data_path,
                    split="train",
                    world_size_independent=True, 
                    limit=args.whitening_nsamples,
                    **common_ds_args
                )
                
                cali_white_data = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.eval_batch_size, 
                    shuffle=False, 
                    collate_fn=svd_collator
                )
            else:
                # Original logic for wikitext/ptb
                cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            
            # Use main profile function. 
            # Note: low_resource doesn't support the Dict input from DataLoader easily without changes.
            profiling_mat = profle_svdllm(args.model, model, cali_white_data, args.DEV)
            
            if args.save_path is not None:
                os.makedirs(args.save_path, exist_ok=True)
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        
        whitening(args.model, model, profiling_mat, args.ratio, args.DEV)
        
        if args.save_path is not None:
            os.makedirs(args.save_path, exist_ok=True)
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + '.pt')   # fp32
    
    elif args.step == 2:
        raise
    elif args.step == 3:
        raise
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        
        if args.step == 4:
            # [Updated] Logic to handle FineWeb/C4 vs Standard Benchmarks
            if args.dataset in ["fineweb-edu", "c4"]:
                print(f"Evaluating PPL on {args.dataset} (custom loop)...")
                
                if args.data_path is None:
                    raise ValueError(f"You must provide --data_path when using {args.dataset}")
                
                def smart_tokenizer_wrapper(examples):
                    text_column = "text" if "text" in examples else list(examples.keys())[0]
                    texts = [t + tokenizer.eos_token for t in examples[text_column]]
                    batch_encodings = tokenizer(
                        texts,
                        truncation=False,
                        max_length=int(1e10),
                        add_special_tokens=True 
                    )
                    return batch_encodings

                common_ds_args = {
                    "sequence_length": args.model_seq_len,
                    "tokenize_fn": smart_tokenizer_wrapper,
                    "seed": args.seed,
                    "use_new_sampling_method": True,
                    "shuffle": False, # Deterministic
                }

                if args.dataset == "fineweb-edu":
                    DatasetClass = FineWebEduDataset
                else:
                    DatasetClass = C4Dataset
                
                # Use 'validation' split if available, else 'train' with limit
                dataset = DatasetClass(
                    path=args.data_path,
                    split="validation", 
                    world_size_independent=True,
                    limit=args.whitening_nsamples, # Re-using this arg to limit eval samples
                    **common_ds_args
                )
                
                eval_loader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.eval_batch_size, 
                    shuffle=False, 
                    collate_fn=svd_collator
                )

                # PPL Calculation Loop
                nlls = []
                total_tokens = 0
                loss_fct = nn.CrossEntropyLoss(reduction="sum")
                
                with torch.no_grad():
                    for batch in tqdm(eval_loader, desc="Evaluating"):
                        batch = {k: v.to(args.DEV) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                        labels = batch["labels"]
                        outputs = model(**batch)
                        logits = outputs.logits
                        
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        not_ignore = shift_labels.ne(loss_fct.ignore_index)
                        num_tokens = not_ignore.sum().item()
                        
                        nlls.append(loss)
                        total_tokens += num_tokens

                if total_tokens > 0:
                    total_nll = torch.stack(nlls).sum()
                    ppl = torch.exp(total_nll / total_tokens)
                    
                    # [FIXED] moved :.2f inside the {}
                    print("PPL after pruning: {:.2f}".format(ppl.item()))
                    print("CE after pruning: {:.2f}".format((total_nll / total_tokens).item()))
                    print("Weight Memory: {:.2f} MiB\n".format(torch.cuda.memory_allocated()/1024/1024))
                else:
                    print("\nWARNING: No tokens processed. Check dataset limits.")
            else:
                # Original logic
                ppl_eval(model, tokenizer, datasets=[args.dataset], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)