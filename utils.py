import jax
import jax.numpy as jnp
import numpy as np
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.model import model

#########################
# rmsd
#########################
def jnp_rmsdist(true, pred):
  def pw_dist(a):
    a_norm = jnp.square(a).sum(-1)
    return jnp.sqrt(jax.nn.relu(a_norm[:,None] + a_norm[None,:] - 2 * a @ a.T) + 1e-8)
  return jnp.sqrt(jnp.square(pw_dist(true)-pw_dist(pred)).mean() + 1e-8)

def jnp_rmsd(true, pred, add_dist=False):
  def kabsch(P, Q):
    V, S, W = jnp.linalg.svd(P.T @ Q, full_matrices=False)
    flip = jax.nn.sigmoid(-10 * jnp.linalg.det(V) * jnp.linalg.det(W))
    S = flip * S.at[-1].set(-S[-1]) + (1-flip) * S
    V = flip * V.at[:,-1].set(-V[:,-1]) + (1-flip) * V
    return V@W

  p = true - true.mean(0,keepdims=True)
  q = pred - pred.mean(0,keepdims=True)
  p = p @ kabsch(p,q)
  loss = jnp.sqrt(jnp.square(p-q).sum(-1).mean() + 1e-8)
  if add_dist:
    loss = (loss + jnp_rmsdist(true, pred))/2
  return loss

####################
# loss functions
####################
def get_dgram_loss(batch, outputs):
  pb, pb_mask = model.modules.pseudo_beta_fn(batch["aatype"],
                                             batch["all_atom_positions"],
                                             batch["all_atom_mask"])
  
  dgram_loss = model.modules._distogram_log_loss(outputs["distogram"]["logits"],
                                                 outputs["distogram"]["bin_edges"],
                                                 batch={"pseudo_beta":pb,"pseudo_beta_mask":pb_mask},
                                                 num_bins=64)
  return dgram_loss["loss"]

####################
# update sequence
####################
def soft_seq(seq_logits, hard=True):
  seq_soft = jax.nn.softmax(seq_logits)
  if hard:
    seq_hard = jax.nn.one_hot(seq_soft.argmax(-1),20)
    return jax.lax.stop_gradient(seq_hard - seq_soft) + seq_soft
  else:
    return seq_soft

def update_seq(seq, inputs, msa_input=False):
  msa_feat = jnp.zeros_like(inputs["msa_feat"]).at[...,0:20].set(seq).at[...,25:45].set(seq)
  if msa_input:
    target_feat = jnp.zeros_like(inputs["target_feat"]).at[...,1:21].set(seq[0])
  else:
    target_feat = jnp.zeros_like(inputs["target_feat"]).at[...,1:21].set(seq)
  inputs.update({"target_feat":target_feat,"msa_feat":msa_feat})

def update_aatype(aatype, inputs):
  if jnp.issubdtype(aatype.dtype, jnp.integer):
    inputs.update({"aatype":aatype,
                   "atom14_atom_exists":residue_constants.restype_atom14_mask[aatype],
                   "atom37_atom_exists":residue_constants.restype_atom37_mask[aatype],
                   "residx_atom14_to_atom37":residue_constants.restype_atom14_to_atom37[aatype],
                   "residx_atom37_to_atom14":residue_constants.restype_atom37_to_atom14[aatype]})
  else:
    restype_atom14_to_atom37 = jax.nn.one_hot(residue_constants.restype_atom14_to_atom37,37)
    restype_atom37_to_atom14 = jax.nn.one_hot(residue_constants.restype_atom37_to_atom14,14)
    inputs.update({"aatype":aatype,
                   "atom14_atom_exists":jnp.einsum("...a,am->...m", aatype, residue_constants.restype_atom14_mask),
                   "atom37_atom_exists":jnp.einsum("...a,am->...m", aatype, residue_constants.restype_atom37_mask),
                   "residx_atom14_to_atom37":jnp.einsum("...a,abc->...bc", aatype, restype_atom14_to_atom37),
                   "residx_atom37_to_atom14":jnp.einsum("...a,abc->...bc", aatype, restype_atom37_to_atom14)})
  
####################
# utils
####################
def set_dropout(model_config, dropout=0.0):
  model_config.model.embeddings_and_evoformer.evoformer.msa_row_attention_with_pair_bias.dropout_rate = dropout
  model_config.model.embeddings_and_evoformer.evoformer.triangle_attention_ending_node.dropout_rate = dropout
  model_config.model.embeddings_and_evoformer.evoformer.triangle_attention_starting_node.dropout_rate = dropout
  model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_incoming.dropout_rate = dropout
  model_config.model.embeddings_and_evoformer.evoformer.triangle_multiplication_outgoing.dropout_rate = dropout
  model_config.model.embeddings_and_evoformer.template.template_pair_stack.triangle_attention_ending_node.dropout_rate = dropout
  model_config.model.embeddings_and_evoformer.template.template_pair_stack.triangle_attention_starting_node.dropout_rate = dropout
  model_config.model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_incoming.dropout_rate = dropout
  model_config.model.embeddings_and_evoformer.template.template_pair_stack.triangle_multiplication_outgoing.dropout_rate = dropout
  model_config.model.heads.structure_module.dropout = dropout
  return model_config

def pdb_to_string(pdb_file):
  lines = []
  for line in open(pdb_file,"r"):
    if line[:6] == "HETATM" and line[17:20] == "MSE":
      line = "ATOM  "+line[6:17]+"MET"+line[20:]
    if line[:4] == "ATOM":
      lines.append(line)
  return "".join(lines)

def save_pdb(outs, filename="tmp.pdb"):
  seq = outs["seq"].argmax(-1)
  b_factors = np.zeros_like(outs["outputs"]['final_atom_mask'])
  p = protein.Protein(
        aatype=seq,
        atom_positions=outs["outputs"]["final_atom_positions"],
        atom_mask=outs["outputs"]['final_atom_mask'],
        residue_index=jnp.arange(len(seq))+1,
        b_factors=b_factors)
  pdb_lines = protein.to_pdb(p)
  with open(filename, 'w') as f:
    f.write(pdb_lines)

order_restype = {v: k for k, v in residue_constants.restype_order.items()}
idx_to_resname = dict((v,k) for k,v in residue_constants.resname_to_idx.items())
