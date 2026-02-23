import torch
import math

def test_hard_capacity():
    N = 4095
    k = 2
    E = 128
    factor = 1.0
    cap = max(1, int((N * k / E) * factor))
    print(f"N={N}, k={k}, E={E}, cap={cap}, N*k={N*k}, E*cap={E*cap}")
    
    # biased_logits
    biased_logits = torch.randn(N, E)
    topk_weights, topk_indices = torch.topk(biased_logits, k, dim=-1)
    usage = torch.zeros(E, dtype=torch.long)
    for idx in topk_indices.reshape(-1):
        usage[idx] += 1
        
    over = (usage > cap).nonzero(as_tuple=True)[0]
    iters = 0
    while over.numel() > 0:
        iters += 1
        if iters > 10:
            print("Infinite loop detected!")
            break
        e = over[0].item()
        excess = (usage[e] - cap).item()
        t_flat = (topk_indices == e).nonzero(as_tuple=False)
        if t_flat.size(0) < excess:
            excess = t_flat.size(0)
        scores_e = biased_logits[t_flat[:, 0], e].clone()
        _, order = torch.sort(scores_e)
        to_reassign = t_flat[order[:excess]]
        
        reassigned = False
        for ii in range(to_reassign.size(0)):
            t_idx, slot = to_reassign[ii, 0].item(), to_reassign[ii, 1].item()
            chosen = set(topk_indices[t_idx].tolist())
            logits_t = biased_logits[t_idx].clone()
            for c in range(E):
                if c in chosen or usage[c].item() >= cap:
                    logits_t[c] = -1e9
            
            new_e = logits_t.argmax().item()
            if logits_t[new_e] <= -1e9:
                pass # can't reassign
            else:
                topk_indices[t_idx, slot] = new_e
                usage[e] -= 1
                usage[new_e] += 1
                reassigned = True
                
        over = (usage > cap).nonzero(as_tuple=True)[0]
        print(f"Iter {iters}: over.numel()={over.numel()}, max usage={usage.max().item()}")

test_hard_capacity()
