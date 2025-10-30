from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch_scatter import scatter_sum, scatter_mean
from models.transition import ContigousTransition, GeneralCategoricalTransition
from models.graph import NodeEdgeNet
from models.common import *
from models.diffusion import *
from utils.transforms import *

VDWRADII = {
    1: 1.1,
    5: 2.1,
    6: 1.90,
    7: 1.8,
    8: 1.7,
    16: 2.0,
    15: 2.1,
    9: 1.5,
    17: 1.8,
    35: 2.0,
    53: 2.2,
    30: 1.2,
    25: 1.2,
    26: 1.2,
    27: 1.2,
    12: 1.2,
    28: 1.2,
    20: 1.2,
    29: 1.2,
}

class DiffGen(Module):
    def __init__(self,
        config,
        protein_node_types,
        ligand_node_types,
        num_edge_types,  # explicit bond type: 0, 1, 2, 3, 4
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.protein_node_types = protein_node_types
        self.ligand_node_types = ligand_node_types
        self.num_edge_types = num_edge_types
        self.k = config.knn
        self.cutoff_mode = config.cutoff_mode
        self.center_pos_mode = config.center_pos_mode
        self.bond_len_loss = getattr(config, 'bond_len_loss', False)
        self.ligand_atom_mode = 'aromatic'
        # # define beta and alpha
        self.define_betas_alphas(config.diff)

        # # embedding

        node_dim = config.node_dim
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim

        self.protein_node_embedder = nn.Linear(protein_node_types, node_dim, bias=False) # protein element type
        self.protein_edge_embedder = nn.Linear(num_edge_types, edge_dim, bias=False) # protein bond type
        self.ligand_node_embedder = nn.Linear(ligand_node_types, node_dim - time_dim, bias=False)  # ligand element type
        self.ligand_edge_embedder = nn.Linear(num_edge_types, edge_dim - time_dim, bias=False) # ligand bond type
        self.time_emb = nn.Sequential(
            GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear'),
        )

        # # denoiser
        if config.denoiser.backbone == 'EGNN':
            self.denoiser = NodeEdgeNet(config.node_dim, config.edge_dim, **config.denoiser)
        else:
            raise NotImplementedError(config.denoiser.backbone)

        # # decoder
        self.ligand_node_decoder = MLP(config.node_dim, ligand_node_types, config.node_dim)
        self.ligand_edge_decoder = MLP(config.edge_dim, num_edge_types, config.edge_dim)


    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')
        
        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 1., 1.])
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (defined for compatibility)

        # # diffusion for pos
        pos_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_pos
        )
        assert self.scaling[0] == 1, 'scaling for pos should be 1'
        self.pos_transition = ContigousTransition(pos_betas)

        # # diffusion for node type
        node_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_atom
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_atom.init_prob
            self.node_transition = GeneralCategoricalTransition(node_betas, self.ligand_node_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.ligand_node_types, scaling_node)
        else:
            raise ValueError(self.categorical_space)

        # # diffusion for edge type
        edge_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_bond
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_bond.init_prob
            self.edge_transition = GeneralCategoricalTransition(edge_betas, self.num_edge_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_edge = self.scaling[2]
            self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)
        else:
            raise ValueError(self.categorical_space)

    def sample_time(self, num_graphs, device, **kwargs):
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt
    
    def fix_zero_time(self, num_graphs, device, **kwargs): 
        time_step = torch.zeros(num_graphs, dtype=torch.long, device=device)   
        pt = torch.ones_like(time_step).float() / self.num_timesteps  
        return time_step, pt

    def _get_edge_index(self, x, batch, ligand_mask):
        if self.cutoff_mode == "knn":
            edge_index = knn_graph(x, k=self.k, batch=batch, flow="target_to_source")
        else:
            raise ValueError(
                f"Unsupported cutoff mode: {self.cutoff_mode}! Please select cutoff mode among knn, hybrid."
            )
        return edge_index

    def _get_edge_type(self, edge_index, ligand_mask):
        src, dst = edge_index
        edge_type = torch.zeros(len(src), dtype=torch.int64).to(edge_index.device)
        n_src = ligand_mask[src] == 1
        n_dst = ligand_mask[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3

        nonzero_indices = torch.nonzero(edge_type).flatten()
        edge_type = torch.index_select(edge_type, dim=0, index=nonzero_indices)
        edge_type = torch.zeros_like(edge_type)
        edge_index = torch.index_select(edge_index, dim=1, index=nonzero_indices)
        edge_type = F.one_hot(edge_type, num_classes=self.num_edge_types)
        return edge_type, edge_index

    def forward(
        self, protein_node, protein_pos, protein_batch, 
        ligand_node_pert, ligand_pos_pert, ligand_batch,
        ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
        t
    ):
        """
        Predict Ligand at step `0` given perturbed Ligand at step `t` with hidden dims and time step
        """
        # 1 node, edge and time embedding
        time_embed_node = self.time_emb(t.index_select(0, ligand_batch))
        time_embed_edge = self.time_emb(t.index_select(0, ligand_edge_batch))
        ligand_node_h_pert = torch.cat([self.ligand_node_embedder(ligand_node_pert), time_embed_node], dim=-1)
        ligand_edge_h_pert = torch.cat([self.ligand_edge_embedder(ligand_edge_pert), time_embed_edge], dim=-1)
        protein_h = self.protein_node_embedder(protein_node)

        # 2 combine protein and ligand input
        all_node_h, all_node_pos, all_node_batch, ligand_mask = compose(
            protein_h, protein_pos, protein_batch, ligand_node_h_pert, ligand_pos_pert, ligand_batch
        )

        sub_edge_index = self._get_edge_index(all_node_pos, all_node_batch, ligand_mask)
        sub_edge_type, sub_edge_index = self._get_edge_type(sub_edge_index, ligand_mask)
        sub_edge_batch = all_node_batch[sub_edge_index[0]]
        sub_edge_h = self.protein_edge_embedder(sub_edge_type.to(torch.float32))
        node_batch_counts = torch.bincount(all_node_batch)
        ligand_node_batch_counts = torch.bincount(ligand_batch)
        cumulative_nodes = torch.cat([torch.tensor([0]).to(all_node_batch.device), torch.cumsum(node_batch_counts, dim=0)[:-1]])
        cumulative_ligand_nodes = torch.cat([torch.tensor([0]).to(ligand_batch.device), torch.cumsum(ligand_node_batch_counts, dim=0)[:-1]])
        new_ligand_edge_index = ligand_edge_index + cumulative_nodes[ligand_edge_batch] - cumulative_ligand_nodes[ligand_edge_batch]
        all_edge_h, all_edge_index, all_edge_batch, ligand_edge_mask = edge_compose(
            sub_edge_h, sub_edge_index, sub_edge_batch, ligand_edge_h_pert, new_ligand_edge_index, ligand_edge_batch
        )

        # 3 diffuse to get the updated node embedding and bond embedding
        node_h, node_pos, edge_h = self.denoiser(
            node_h=all_node_h,
            node_pos=all_node_pos, 
            edge_h=all_edge_h, 
            edge_index=all_edge_index,
            node_time=t.index_select(0, all_node_batch).unsqueeze(-1) / self.num_timesteps,
            edge_time=t.index_select(0, all_edge_batch).unsqueeze(-1) / self.num_timesteps,
            ligand_mask=ligand_mask
        )
        
        ligand_node_h = node_h[ligand_mask]
        ligand_node_pos = node_pos[ligand_mask]
        ligand_edge_h = edge_h[ligand_edge_mask]
        n_halfedges = ligand_edge_h.shape[0] // 2
        pred_ligand_node = self.ligand_node_decoder(ligand_node_h)
        pred_ligand_halfedge = self.ligand_edge_decoder(ligand_edge_h[:n_halfedges] + ligand_edge_h[n_halfedges:])
        pred_ligand_pos = ligand_node_pos
        
        return {
            'pred_ligand_node': pred_ligand_node,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_halfedge': pred_ligand_halfedge
        }  # ligand at step 0

    def get_loss(
        self, protein_node, protein_pos, protein_batch, 
        ligand_node, ligand_pos, ligand_batch,
        halfedge_type, halfedge_index, halfedge_batch,
        num_mol, train=True 
    ):
        num_graphs = num_mol
        device = ligand_pos.device
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, protein_batch, ligand_batch, mode=self.center_pos_mode
        )

        # 1. sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

        # 2. perturb pos, node, edge
        pos_pert = self.pos_transition.add_noise(ligand_pos, time_step, ligand_batch)
        node_pert = self.node_transition.add_noise(ligand_node, time_step, ligand_batch)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, halfedge_batch)
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        if self.categorical_space == 'discrete':
            ligand_node_pert, log_node_t, log_node_0 = node_pert
            ligand_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
        else:
            ligand_node_pert, ligand_node_0 = node_pert
            ligand_halfedge_pert, ligand_halfedge_0 = halfedge_pert
        
        ligand_edge_pert = torch.cat([ligand_halfedge_pert, ligand_halfedge_pert], dim=0)
        ligand_pos_pert = pos_pert

        # 3. forward to denoise
        preds = self(
            protein_node, protein_pos, protein_batch,
            ligand_node_pert, ligand_pos_pert, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
            time_step
        )
        pred_ligand_node = preds['pred_ligand_node']
        pred_ligand_pos = preds['pred_ligand_pos']
        pred_ligand_halfedge = preds['pred_ligand_halfedge']

        # 4. loss
        # 4.1 pos loss
        loss_pos = F.mse_loss(pred_ligand_pos, ligand_pos)
        if self.bond_len_loss == True:
            bond_index = halfedge_index[:, halfedge_type > 0]
            true_length = torch.norm(ligand_pos[bond_index[0]] - ligand_pos[bond_index[1]], dim=-1)
            pred_length = torch.norm(pred_ligand_pos[bond_index[0]] - pred_ligand_pos[bond_index[1]], dim=-1)
            loss_len = F.mse_loss(pred_length, true_length)
    
        if self.categorical_space == 'discrete':
            # 4.2 node type loss
            log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
            log_node_post_true = self.node_transition.q_v_posterior(log_node_0, log_node_t, time_step, ligand_batch, v0_prob=True)
            log_node_post_pred = self.node_transition.q_v_posterior(log_node_recon, log_node_t, time_step, ligand_batch, v0_prob=True)
            kl_node = self.node_transition.compute_v_Lt(log_node_post_true, log_node_post_pred, log_node_0, t=time_step, batch=ligand_batch)
            loss_node = torch.mean(kl_node) * 100
            # 4.3 edge type loss
            log_halfedge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
            log_edge_post_true = self.edge_transition.q_v_posterior(log_halfedge_0, log_halfedge_t, time_step, halfedge_batch, v0_prob=True)
            log_edge_post_pred = self.edge_transition.q_v_posterior(log_halfedge_recon, log_halfedge_t, time_step, halfedge_batch, v0_prob=True)
            kl_edge = self.edge_transition.compute_v_Lt(log_edge_post_true, log_edge_post_pred, log_halfedge_0, t=time_step, batch=halfedge_batch)
            loss_edge = torch.mean(kl_edge)  * 100
        else:
            loss_node = F.mse_loss(pred_ligand_node, ligand_node_0)  * 30
            loss_edge = F.mse_loss(pred_ligand_halfedge, ligand_halfedge_0) * 30
        # 4.4 energy calculation
        if self.config.pinn and train:
            dm_min = 0.5
            total_energy = 0
            for batch_idx in range(num_graphs):
                ligand_coor = pred_ligand_pos[ligand_batch==batch_idx]
                protein_coor = protein_pos[protein_batch==batch_idx]
                p1_repeat = ligand_coor.unsqueeze(1).repeat(1, protein_coor.size(0), 1)
                p2_repeat = protein_coor.unsqueeze(0).repeat(ligand_coor.size(0), 1, 1)
                dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
                replace_vec = torch.ones_like(dm) * 1e10
                dm = torch.where(dm < dm_min, replace_vec, dm)

                # ligand_recon_atom_type = log_sample_categorical(pred_ligand_node[ligand_batch==batch_idx])
                # pred_log_atom_type = get_atomic_number_from_index(ligand_recon_atom_type,self.ligand_atom_mode)
                # atom_type_list = get_atom_type(pred_log_atom_type)
                
                # ligand_vdw_radii = torch.tensor([
                #     VDWRADII[atom_idx] for atom_idx in pred_log_atom_type
                # ])       
      
                # dm_0 = ligand_vdw_radii.unsqueeze(1).repeat(1, protein_coor.size(0)).cuda()

                N = 6
                # vdw_term1 = torch.pow(dm_0 / dm, 2 * N)
                # vdw_term2 = -2 * torch.pow(dm_0 / dm, N)
                vdw_term1 = torch.pow(1 / dm, 2 * N)
                vdw_term2 = -2 * torch.pow(1 / dm, N)
                
                energy = vdw_term1 + vdw_term2
                energy = energy.clamp(max=100)
                
                # der = torch.zeros_like(ligand_coor)
                # for x in torch.sum(energy, -1):
                #     der += torch.autograd.grad(x, ligand_coor, retain_graph=True, create_graph=True)[0]
                # der = torch.pow(der.sum(1), 2).mean()
                
                der = torch.autograd.grad(energy.sum(), dm, retain_graph=True, create_graph=True)[0]
                der = der.sum()
                total_energy += der
        else:
            total_energy = torch.tensor(0.).to(device)
        # total loss
        if self.config.train_mode in ('ori'):
            loss_total = loss_pos + loss_node + loss_edge + (loss_len if self.bond_len_loss else 0)
            loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge,
            'energy'  : total_energy,
        }
        elif self.config.train_mode in ('no_pi'):
            loss_total = loss_pos + loss_node + loss_edge + (loss_len if self.bond_len_loss else 0)
            loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge
        }
        if self.bond_len_loss == True:
            loss_dict['loss_len'] = loss_len

        pred_dict = {
            'pred_ligand_node': F.softmax(pred_ligand_node, dim=-1),
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_halfedge': F.softmax(pred_ligand_halfedge, dim=-1)
        }
        return loss_dict, pred_dict

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_eps = extract(self.pos_transition.sqrt_recip_alphas_bar, t, batch) * xt - \
                      extract(self.pos_transition.sqrt_recipm1_alphas_bar, t, batch) * eps
        return pos0_from_eps

    def _predict_eps_from_x0(self, xt, t, pred_x0, batch):
        return (
            (extract(self.pos_transition.sqrt_recip_alphas_bar, t, batch) * xt - pred_x0) /
            extract(self.pos_transition.sqrt_recipm1_alphas_bar, t, batch)
        )

    @torch.no_grad()
    def sample(
        self, n_graphs, 
        protein_node, protein_pos, protein_batch, 
        ligand_batch, halfedge_index, halfedge_batch, 
        bond_predictor=None, guidance=None
    ):
        device = ligand_batch.device
        # # 1. get the init values (position, node and edge types)
        n_nodes_all = len(ligand_batch)
        n_halfedges_all = len(halfedge_batch)
        
        node_init = self.node_transition.sample_init(n_nodes_all)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
        if self.categorical_space == 'discrete':
            _, ligand_node_h_init, log_node_type = node_init
            _, ligand_halfedge_h_init, log_halfedge_type = halfedge_init
        else:
            ligand_node_h_init = node_init
            ligand_halfedge_h_init = halfedge_init
            
        pocket_center_pos = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_center_pos = pocket_center_pos[ligand_batch]
        ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
        protein_pos, ligand_pos_init, offset = center_pos(protein_pos, ligand_pos_init, protein_batch, ligand_batch, self.center_pos_mode)

        # # 1.1 log init
        ligand_node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_all, ligand_node_h_init.shape[-1]],
                                dtype=ligand_node_h_init.dtype).to(device)
        ligand_pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_all, 3], dtype=ligand_pos_init.dtype).to(device)
        ligand_halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_all, ligand_halfedge_h_init.shape[-1]],
                                    dtype=ligand_halfedge_h_init.dtype).to(device)
        ligand_node_traj[0] = ligand_node_h_init
        ligand_pos_traj[0] = ligand_pos_init + offset[ligand_batch]
        ligand_halfedge_traj[0] = ligand_halfedge_h_init

        # # 2. sample loop
        ligand_node_h_pert = ligand_node_h_init
        ligand_pos_pert = ligand_pos_init
        ligand_halfedge_h_pert = ligand_halfedge_h_init
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            ligand_edge_h_pert = torch.cat([ligand_halfedge_h_pert, ligand_halfedge_h_pert], dim=0)
            
            preds = self(
                protein_node, protein_pos, protein_batch,
                ligand_node_h_pert, ligand_pos_pert, ligand_batch,
                ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                time_step
            )
            pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = preds['pred_ligand_pos'], preds['pred_ligand_node'], preds['pred_ligand_halfedge']
            
            # # 2.2 get the t - 1 state
            # pos 
            ligand_pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=ligand_pos_pert, x_recon=pred_ligand_pos, t=time_step, batch=ligand_batch
            )
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, ligand_batch, v0_prob=True)
                ligand_node_type_prev = log_sample_categorical(log_node_type)
                ligand_node_h_prev = self.node_transition.onehot_encode(ligand_node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, halfedge_batch, v0_prob=True)
                ligand_halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                ligand_halfedge_h_prev = self.edge_transition.onehot_encode(ligand_halfedge_type_prev)
                
            else:
                ligand_node_h_prev = self.node_transition.get_prev_from_recon(
                    x_t=ligand_node_h_pert, x_recon=pred_ligand_node, t=time_step, batch=ligand_batch)
                ligand_halfedge_h_prev = self.edge_transition.get_prev_from_recon(
                    x_t=ligand_halfedge_h_pert, x_recon=pred_ligand_halfedge, t=time_step, batch=halfedge_batch)

            # # 2.3 use guidance to modify pos
            if self.config.train_mode not in ('no_bond', 'no_both'):
                if guidance is not None:
                    gui_type, gui_scale = guidance
                    if (gui_scale > 0):
                        with torch.enable_grad():
                            ligand_node_h_in = ligand_node_h_pert.detach()
                            ligand_pos_in = ligand_pos_pert.detach().requires_grad_(True)
                            pred_bondpredictor = bond_predictor(
                                protein_node, protein_pos, protein_batch,
                                ligand_node_h_in, ligand_pos_in, ligand_batch,
                                ligand_edge_index, ligand_edge_batch, time_step)
                            delta = self.bond_guidance(gui_type, gui_scale, pred_bondpredictor, ligand_pos_in, ligand_halfedge_type_prev, log_halfedge_type)
                        ligand_pos_prev = ligand_pos_prev + delta

            # 2.4 log update
            ligand_node_traj[i+1] = ligand_node_h_prev
            ligand_pos_traj[i+1] = ligand_pos_prev + offset[ligand_batch]
            ligand_halfedge_traj[i+1] = ligand_halfedge_h_prev

            # # 2.5 update t-1
            ligand_pos_pert = ligand_pos_prev
            ligand_node_h_pert = ligand_node_h_prev
            ligand_halfedge_h_pert = ligand_halfedge_h_prev

        pred_ligand_pos = pred_ligand_pos + offset[ligand_batch] 
        # # 3. get the final positions
        return {
            'pred': [pred_ligand_node, pred_ligand_pos, pred_ligand_halfedge],
            'traj': [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj]
        }

    @torch.no_grad()
    def sample_frag(
        self, n_graphs, 
        protein_node, protein_pos, protein_batch, 
        frag_node, frag_pos, frag_batch,
        frag_halfedge_type, frag_halfedge_index, frag_halfedge_batch,
        ligand_batch, halfedge_index, halfedge_batch, 
        gui_strength=None, 
        bond_predictor=None, guidance=None, gen_mode=None
    ):
        device = ligand_batch.device
        # # 1. get the init values (position, node and edge types)
        n_nodes_ligand = len(ligand_batch)
        n_halfedges_ligand = len(halfedge_batch)        
        node_init = self.node_transition.sample_init(n_nodes_ligand)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_ligand)
        if self.categorical_space == 'discrete':
            _, ligand_node_h_init, log_node_type = node_init
            _, ligand_halfedge_h_init, log_halfedge_type = halfedge_init
        else:
            ligand_node_h_init = node_init
            ligand_halfedge_h_init = halfedge_init

        frag_node_mask = get_fragment_mask(ligand_batch, frag_batch)
        frag_halfedge_mask = get_fragment_mask(halfedge_batch, frag_halfedge_batch)
        pocket_center_pos = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_center_pos = pocket_center_pos[ligand_batch]
        ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
        protein_pos, ligand_pos_init, offset = center_pos(protein_pos, ligand_pos_init, protein_batch, ligand_batch, self.center_pos_mode)
        frag_pos = frag_pos - offset[frag_batch]

        # # 1.1 init trajectory
        ligand_node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, ligand_node_h_init.shape[-1]],
                                dtype=ligand_node_h_init.dtype).to(device)
        ligand_pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, 3], dtype=ligand_pos_init.dtype).to(device)
        ligand_halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_ligand, ligand_halfedge_h_init.shape[-1]],
                                    dtype=ligand_halfedge_h_init.dtype).to(device)
        ligand_node_traj[0] = ligand_node_h_init
        ligand_pos_traj[0] = ligand_pos_init + offset[ligand_batch]
        ligand_halfedge_traj[0] = ligand_halfedge_h_init

        # # 2. sample loop
        ligand_node_h_pert = ligand_node_h_init
        ligand_pos_pert = ligand_pos_init
        ligand_halfedge_h_pert = ligand_halfedge_h_init
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)

            # # 2.1 get the init values for fragment
            if gen_mode == 'frag_cond':
                frag_pos_pert = frag_pos
                log_frag_node_type = index_to_log_onehot(frag_node, self.ligand_node_types)
                frag_node_pert = F.one_hot(log_sample_categorical(index_to_log_onehot(frag_node, self.ligand_node_types)), self.ligand_node_types).float()
                log_frag_halfedge_type = index_to_log_onehot(frag_halfedge_type, self.num_edge_types)
                frag_halfedge_pert = F.one_hot(log_sample_categorical(index_to_log_onehot(frag_halfedge_type, self.num_edge_types)), self.num_edge_types).float()
            elif gen_mode == 'frag_diff':
                pos_pert = self.pos_transition.add_noise(frag_pos, time_step, frag_batch)
                node_pert = self.node_transition.add_noise(frag_node, time_step, frag_batch)
                halfedge_pert = self.edge_transition.add_noise(frag_halfedge_type, time_step, frag_halfedge_batch)
                
                if self.categorical_space == 'discrete':
                    frag_node_pert, log_frag_node_type, _ = node_pert
                    frag_halfedge_pert, log_frag_halfedge_type, _ = halfedge_pert
                else:
                    frag_node_pert, _ = node_pert
                    frag_halfedge_pert, _ = halfedge_pert
                frag_pos_pert = pos_pert

            # # 2.2 combine fragment and ligand
            ligand_pos_pert[frag_node_mask] = frag_pos_pert
            ligand_node_h_pert[frag_node_mask] = frag_node_pert
            ligand_halfedge_h_pert[frag_halfedge_mask] = frag_halfedge_pert
            log_node_type[frag_node_mask] = log_frag_node_type
            log_halfedge_type[frag_halfedge_mask] = log_frag_halfedge_type
            ligand_edge_h_pert = torch.cat([ligand_halfedge_h_pert, ligand_halfedge_h_pert], dim=0)
            
            preds = self(
                protein_node, protein_pos, protein_batch,
                ligand_node_h_pert, ligand_pos_pert, ligand_batch,
                ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                time_step
            )
            pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = preds['pred_ligand_pos'], preds['pred_ligand_node'], preds['pred_ligand_halfedge']

            # # 2.4 get the t - 1 state
            # pos 
            ligand_pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=ligand_pos_pert, x_recon=pred_ligand_pos, t=time_step, batch=ligand_batch
            )
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, ligand_batch, v0_prob=True)
                ligand_node_type_prev = log_sample_categorical(log_node_type)
                ligand_node_h_prev = self.node_transition.onehot_encode(ligand_node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, halfedge_batch, v0_prob=True)
                ligand_halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                ligand_halfedge_h_prev = self.edge_transition.onehot_encode(ligand_halfedge_type_prev)
                
            else:
                ligand_node_h_prev = self.node_transition.get_prev_from_recon(
                    x_t=ligand_node_h_pert, x_recon=pred_ligand_node, t=time_step, batch=ligand_batch)
                ligand_halfedge_h_prev = self.edge_transition.get_prev_from_recon(
                    x_t=ligand_halfedge_h_pert, x_recon=pred_ligand_halfedge, t=time_step, batch=halfedge_batch)

            # # 2.5 use guidance to modify pos
            if self.config.train_mode not in ('no_bond', 'no_both'):
                if guidance is not None:
                    gui_type, gui_scale = guidance
                    if (gui_scale > 0):
                        with torch.enable_grad():
                            ligand_node_h_in = ligand_node_h_pert.detach()
                            ligand_pos_in = ligand_pos_pert.detach().requires_grad_(True)
                            pred_bondpredictor = bond_predictor(
                                protein_node, protein_pos, protein_batch,
                                ligand_node_h_in, ligand_pos_in, ligand_batch,
                                ligand_edge_index, ligand_edge_batch, time_step)
                            delta = self.bond_guidance(gui_type, gui_scale, pred_bondpredictor, ligand_pos_in, ligand_halfedge_type_prev, log_halfedge_type)
                        ligand_pos_prev = ligand_pos_prev + delta

            # 2.6 update trajectory
            ligand_node_traj[i+1] = ligand_node_h_prev
            ligand_pos_traj[i+1] = ligand_pos_prev + offset[ligand_batch]
            ligand_halfedge_traj[i+1] = ligand_halfedge_h_prev

            # # 2.7 update t-1
            ligand_pos_pert = ligand_pos_prev
            ligand_node_h_pert = ligand_node_h_prev
            ligand_halfedge_h_pert = ligand_halfedge_h_prev

        pred_ligand_pos = pred_ligand_pos + offset[ligand_batch]

        # # 3. get the final positions
        return {
            'pred': [pred_ligand_node, pred_ligand_pos, pred_ligand_halfedge],
            'traj': [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj]
        }

    def bond_guidance(self, gui_type, gui_scale, pred_bondpredictor, ligand_pos_in, halfedge_type_prev, log_halfedge_type):
        if gui_type == 'entropy':
            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'uncertainty':
            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
            uncertainty = uncertainty.log().sum()
            delta = - torch.autograd.grad(uncertainty, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'uncertainty_bond':  # only for the predicted real bond (not no bond)
            prob = torch.softmax(pred_bondpredictor, dim=-1)
            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
            uncertainty = uncertainty.log()
            uncertainty = (uncertainty * prob[:, 1:].detach().sum(dim=-1)).sum()
            delta = - torch.autograd.grad(uncertainty, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'entropy_bond':
            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
            entropy = entropy.log()
            entropy = (entropy * prob_halfedge[:, 1:].detach().sum(dim=-1)).sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'logit_bond':
            ind_real_bond = ((halfedge_type_prev >= 1) & (halfedge_type_prev <= 4))
            idx_real_bond = ind_real_bond.nonzero().squeeze(-1)
            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
            pred = pred_real_bond.sum()
            delta = + torch.autograd.grad(pred, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'logit':
            ind_bond_notmask = (halfedge_type_prev <= 4)
            idx_real_bond = ind_bond_notmask.nonzero().squeeze(-1)
            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
            pred = pred_real_bond.sum()
            delta = + torch.autograd.grad(pred, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'crossent':
            prob_halfedge_type = log_halfedge_type.exp()[:, :-1]  # the last one is masked bond (not used in predictor)
            entropy = F.cross_entropy(pred_bondpredictor, prob_halfedge_type, reduction='none')
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'crossent_bond':
            prob_halfedge_type = log_halfedge_type.exp()[:, 1:-1]  # the last one is masked bond. first one is no bond
            entropy = F.cross_entropy(pred_bondpredictor[:, 1:], prob_halfedge_type, reduction='none')
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        else:
            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
        
        return delta

def get_atom_type(pred_log_atom_type): # pred_log_atom_type : List
    type_list = []
    for index in pred_log_atom_type:
        atom = map_index_to_atom_type_only[index]
        type_list.append(atom)
    return type_list