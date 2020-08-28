import torch
import torch.nn as nn

class LinearStateSpaceWrapper():
    def __init__(self, A, B, device="cpu"):
        self.A = A
        self.B = B
        self.device = device

    def to(self, device):
        self.device = device
    
    def rollout(self, z_0, u):
        """
        Args:
            z_0: State (tensor, (batch_size, dim_z))        
            u: Controls (pred_len, batch_size, dim_u)
        Returns:
            z_hat: Predicted future sampled states (pred_len, batch_size, dim_z)
            info: Any extra information needed {
                A: Locally linear transition matrices (pred_len, batch_size, dim_z, dim_z)
                B: Locally linear control matrices (pred_len, batch_size, dim_z, dim_u)
            }
        """
        pred_len, batch_size = u.shape[0], u.shape[1]
        dim_z, dim_u = z_0.shape[-1], u.shape[-1]

        z_hat = torch.zeros((pred_len, batch_size, dim_z)).float().to(device=self.device)
        h_0 = None

        for ll in range(pred_len):
            z_t1_hat = (torch.bmm(self.A[ll].repeat(batch_size, 1, 1), z_0.unsqueeze(-1)) + \
                torch.bmm(self.B[ll].repeat(batch_size, 1, 1), u[ll].unsqueeze(-1))).squeeze(-1)
            z_hat[ll] = z_t1_hat

        info = {
            "A": self.A,
            "B": self.B
        }

        return z_hat, info

class LinearMixWrapper():
    def __init__(self, dyn_model, device="cpu"):
        self.dyn_model = dyn_model
        self.device = device
    
    def to(self, device):
        self.device = device
        self.dyn_model.to(device=device)
    
    def rollout(self, z_0, u):
        """
        Args:
            z_0: State distribution (dict, {
                z:(batch_size, dim_z), 
                mu:(batch_size, dim_z), 
                cov:(batch_size, dim_z, dim_z)
            })              
            u: Controls (pred_len, batch_size, dim_u)
        Returns:
            z_hat: Predicted future sampled states (pred_len, batch_size, dim_z)
            info: Any extra information needed {
                A: Locally linear transition matrices (pred_len, batch_size, dim_z, dim_z)
                B: Locally linear control matrices (pred_len, batch_size, dim_z, dim_u)
            }
        """
        z_0_sample, mu_0, var_0 = z_0["z"], z_0["mu"], z_0["cov"]

        pred_len, batch_size = u.shape[0], u.shape[1]
        dim_z, dim_u = z_0_sample.shape[-1], u.shape[-1]

        z_hat = torch.zeros((pred_len, batch_size, dim_z)).float().to(device=self.device)
        mu_hat = torch.zeros((pred_len, batch_size, dim_z)).float().to(device=self.device)
        var_hat = torch.zeros((pred_len, batch_size, dim_z, dim_z)).float().to(device=self.device)
        A = torch.zeros((pred_len, batch_size, dim_z, dim_z)).float().to(device=self.device)
        B = torch.zeros((pred_len, batch_size, dim_z, dim_u)).float().to(device=self.device)
        h_0 = None

        for ll in range(pred_len):
            z_t1_hat, mu_z_t1_hat, var_z_t1_hat, h_t1, A_t1, B_t1 = \
                self.dyn_model(
                    z_t=z_0_sample, 
                    mu_t=mu_0, 
                    var_t=var_0, 
                    u=u[ll],
                    h_0=h_0,
                    single=True,
                    return_matrices=True
                )

            z_hat[ll] = z_t1_hat
            mu_hat[ll] = mu_z_t1_hat
            var_hat[ll] = var_z_t1_hat
            A[ll] = A_t1
            B[ll] = B_t1
            z_0_sample, mu_0, var_0, h_0 = z_t1_hat, mu_z_t1_hat, var_z_t1_hat, h_t1    

        info = {
            "A": A,
            "B": B
        }

        #XXX: Track based on sample for now
        return z_hat, info