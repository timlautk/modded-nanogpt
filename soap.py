from itertools import chain
import torch

# Parts of the code are modifications of Pytorch's AdamW optimizer
# Parts of the code are modifications of code from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py


class SOAP(torch.optim.Optimizer):
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321).

    Parameters:
        params (`Iterable[nn.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.003):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
            Adam's betas parameters (b1, b2).
        shampoo_beta (`float`, *optional*, defaults to -1):
            If >= 0, use this beta for the preconditioner (L and R in paper, state['GG'] below) moving average instead of betas[1].
        eps (`float`, *optional*, defaults to 1e-08):
            Adam's epsilon for numerical stability.
        precondition_frequency (`int`, *optional*, defaults to 10):
            How often to update the preconditioner.
        normalize_grads (`bool`, *optional*, defaults to `False`):
            Whether or not to normalize gradients per layer. 
            Helps at large precondition_frequency (~100 in our experiments), 
            but hurts performance at small precondition_frequency (~10 in our experiments).
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas=(0.95, 0.95),
        shampoo_beta: float= -1,
        eps: float = 1e-8,
        precondition_frequency: int=10,
        normalize_grads: bool = False,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "precondition_frequency": precondition_frequency,
            "normalize_grads": normalize_grads,
        }
        super().__init__(params, defaults)
        
    def step(self):
        """
        Performs a single optimization step.
        """
        for group in self.param_groups:

            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                state['step'] = state.get('step', 0) + 1
                if state['step'] == 1:
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    state['GG'] = [] # Will hold all the preconditioner matrices (L and R in the paper).
                    for sh in grad.shape:
                        state['GG'].append(torch.zeros(sh, sh, device=grad.device))
                    state['Q'] = None # Will hold all the eigenbases of the preconditioner.
                    state['precondition_frequency'] = group['precondition_frequency']
                    state['shampoo_beta'] = group['shampoo_beta'] if group['shampoo_beta'] >= 0 else group["betas"][1]
                    self.update_preconditioner(grad, state)
                    continue # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner 
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = self.project(grad, state)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))

                # Projecting the exponential moving average of gradients to the eigenbases of Shampoo's preconditioner 
                # i.e. projecting to the eigenbases of matrices in state['GG']
                exp_avg_projected = self.project(exp_avg, state)

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                norm_grad = self.project_back(exp_avg_projected / denom, state)

                step_size = group["lr"]
                #if group["correct_bias"]:
                bias_correction1 = 1.0 - beta1 ** (state["step"])
                bias_correction2 = 1.0 - beta2 ** (state["step"])
                step_size = step_size * (bias_correction2 ** .5) / bias_correction1
                if group["normalize_grads"]:
                    norm_grad = norm_grad / (1e-30+torch.mean(norm_grad**2)**0.5)
                p.add_(norm_grad, alpha=-step_size)

                # Update is done after the gradient step to avoid using current gradients in the projection.
                self.update_preconditioner(grad, state)

    def project(self, grad, state):
        """
        Projects the gradient to the eigenbases of the preconditioner.
        """
        original_shape = grad.shape
        for mat in state['Q']:
            grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [0]],
                )
        return grad

    def update_preconditioner(self, grad, state):
        """
        Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
        """
        for idx, sh in enumerate(grad.shape):
            outer_product = torch.tensordot(
                    grad,
                    grad,
                    # Contracts across all dimensions except for k.
                    dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                )
            state['GG'][idx].lerp_(outer_product, 1-state['shampoo_beta'])
                     
        if state['Q'] is None:
            state['Q'] = self.get_orthogonal_matrix(state['GG'])
        if state['step'] > 0 and state['step'] % state['precondition_frequency'] == 0:
            state['Q'] = self.get_orthogonal_matrix_QR(state)           

    def project_back(self, grad, state):
        """
        Projects the gradient back to the original space.
        """
        for mat in state['Q']:
            grad = torch.tensordot(
                    grad,
                    mat,
                    dims=[[0], [1]],
                )
        return grad

    def get_orthogonal_matrix(self, mat):
        """
        Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
        """
        final = []
        for m in mat:
            try:
                _, Q = torch.linalg.eigh(m+1e-30*torch.eye(m.shape[0], device=m.device))
            except:
                _, Q = torch.linalg.eigh(m.to(torch.float64)+1e-30*torch.eye(m.shape[0], device=m.device))
                Q = Q.to(m.dtype)
            Q = torch.flip(Q, [1])
            final.append(Q)
        return final

    def get_orthogonal_matrix_QR(self, state):
        """
        Computes the eigenbases of the preconditioner using one round of power iteration 
        followed by torch.linalg.qr decomposition.
        """
        precond_list = state['GG']
        orth_list = state['Q']

        orig_shape = state['exp_avg_sq'].shape
        exp_avg_sq = state['exp_avg_sq']

        final = []
        for ind, (m, o) in enumerate(zip(precond_list, orth_list)):
            est_eig = torch.diag(o.T @ m @ o)
            sort_idx = torch.argsort(est_eig, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            o = o[:,sort_idx]
            power_iter = m @ o
            Q, _ = torch.linalg.qr(power_iter)
            final.append(Q)
        
        state['exp_avg_sq'] = exp_avg_sq
        return final

