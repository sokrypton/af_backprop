import jax
import jax.numpy as jnp
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import numpy as np
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.model import model
from alphafold.model import folding
from alphafold.model import all_atom
from alphafold.model.tf import shape_placeholders

#######################
# reshape inputs
#######################

def make_fixed_size(feat, model_runner, length, batch_axis=True):
  '''pad input features'''
  cfg = model_runner.config
  if batch_axis:
    shape_schema = {k:[None]+v for k,v in dict(cfg.data.eval.feat).items()}
  else:
    shape_schema = {k:v for k,v in dict(cfg.data.eval.feat).items()}

  pad_size_map = {
      shape_placeholders.NUM_RES: length,
      shape_placeholders.NUM_MSA_SEQ: cfg.data.eval.max_msa_clusters,
      shape_placeholders.NUM_EXTRA_SEQ: cfg.data.common.max_extra_msa,
      shape_placeholders.NUM_TEMPLATES: cfg.data.eval.max_templates
  }
  for k, v in feat.items():
    # Don't transfer this to the accelerator.
    if k == 'extra_cluster_assignment':
      continue
    shape = list(v.shape)
    schema = shape_schema[k]
    assert len(shape) == len(schema), (
        f'Rank mismatch between shape and shape schema for {k}: '
        f'{shape} vs {schema}')
    pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]
    padding = [(0, p - tf.shape(v)[i]) for i, p in enumerate(pad_size)]
    if padding:
      feat[k] = tf.pad(v, padding, name=f'pad_to_fixed_{k}')
      feat[k].set_shape(pad_size)
  return {k:np.asarray(v) for k,v in feat.items()}

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

###############
# weighted rmsd
###############
def jnp_kabsch_w(a, b, weights):
  u, s, vh = jnp.linalg.svd(weights * a.T @ b, full_matrices=False)
  u = jnp.where(jnp.linalg.det(u @ vh) < 0, u.at[:,-1].set(-u[:,-1]), u)
  return u @ vh

def jnp_rmsd_w(true, pred, weights):
  p = true - (true * weights[:,None]).sum(0,keepdims=True)/weights.sum()
  q = pred - (pred * weights[:,None]).sum(0,keepdims=True)/weights.sum()
  p = p @ jnp_kabsch_w(p, q, weights)
  return jnp.sqrt((weights*jnp.square(p-q).sum(-1)).sum()/weights.sum() + 1e-8)

def get_rmsd_loss_w(batch, outputs, copies=1):
  weights = batch["all_atom_mask"][:,1]
  true = batch["all_atom_positions"][:,1,:]
  pred = outputs["structure_module"]["final_atom_positions"][:,1,:]
  if copies == 1:
    return jnp_rmsd_w(true, pred, weights)
  else:
    # TODO add support for weights
    I = copies - 1
    L = true.shape[0] // copies
    p = true - true[:L].mean(0)
    q = pred - pred[:L].mean(0)
    p = p @ jnp_kabsch_w(p[:L], q[:L], jnp.ones(L))
    rm = jnp.square(p[:L]-q[:L]).sum(-1).mean()
    p,q = p[L:].reshape(I,1,L,-1),q[L:].reshape(1,I,L,-1)
    rm += jnp.square(p-q).sum(-1).mean(-1).min(-1).sum()
    return jnp.sqrt(rm / copies)

####################
# confidence metrics
####################
def get_plddt(outputs):
  logits = outputs["predicted_lddt"]["logits"]
  num_bins = logits.shape[-1]
  bin_width = 1.0 / num_bins
  bin_centers = jnp.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)
  probs = jax.nn.softmax(logits, axis=-1)
  return jnp.sum(probs * bin_centers[None, :], axis=-1)

def get_pae(outputs):
  prob = jax.nn.softmax(outputs["predicted_aligned_error"]["logits"],-1)
  breaks = outputs["predicted_aligned_error"]["breaks"]
  step = breaks[1]-breaks[0]
  bin_centers = breaks + step/2
  bin_centers = jnp.append(bin_centers,bin_centers[-1]+step)
  return (prob*bin_centers).sum(-1)

####################
# loss functions
####################
def get_rmsd_loss(batch, outputs):
  true = batch["all_atom_positions"][:,1,:]
  pred = outputs["structure_module"]["final_atom_positions"][:,1,:]
  return jnp_rmsd(true, pred)

def _distogram_log_loss(logits, bin_edges, batch, num_bins, copies=1):
  """Log loss of a distogram."""
  pos,mask = batch['pseudo_beta'],batch['pseudo_beta_mask']
  sq_breaks = jnp.square(bin_edges)
  dist2 = jnp.square(pos[:,None] - pos[None,:]).sum(-1,keepdims=True)
  true_bins = jnp.sum(dist2 > sq_breaks, axis=-1)
  true = jax.nn.one_hot(true_bins, num_bins)

  if copies == 1:
    errors = -(true * jax.nn.log_softmax(logits)).sum(-1)
    sq_mask = mask[:,None] * mask[None,:]
    avg_error = (errors * sq_mask).sum()/(1e-6 + sq_mask.sum())
    return avg_error
  else:
    # TODO add support for masks
    L = pos.shape[0] // copies
    I = copies - 1
    true_, pred_ = true[:L,:L], logits[:L,:L]
    errors = -(true_ * jax.nn.log_softmax(pred_)).sum(-1)
    avg_error = errors.mean()

    true_, pred_ = true[:L,L:], logits[:L,L:]
    true_, pred_ = true_.reshape(L,I,1,L,-1), pred_.reshape(L,1,I,L,-1)
    errors = -(true_ * jax.nn.log_softmax(pred_)).sum(-1)
    avg_error += errors.mean((0,-1)).min(-1).sum()

    return avg_error / copies

def get_dgram_loss(batch, outputs, model_config, logits=None, copies=1):
  # get cb features (ca in case of glycine)
  pb, pb_mask = model.modules.pseudo_beta_fn(batch["aatype"],
                                             batch["all_atom_positions"],
                                             batch["all_atom_mask"])
  if logits is None: logits = outputs["distogram"]["logits"]
  dgram_loss = _distogram_log_loss(logits,
                                   outputs["distogram"]["bin_edges"],
                                   batch={"pseudo_beta":pb,"pseudo_beta_mask":pb_mask},
                                   num_bins=model_config.model.heads.distogram.num_bins,
                                   copies=copies)
  return dgram_loss

def get_fape_loss(batch, outputs, model_config, use_clamped_fape=False):
  sub_batch = jax.tree_map(lambda x: x, batch)
  sub_batch["use_clamped_fape"] = use_clamped_fape
  loss = {"loss":0.0}    
  folding.backbone_loss(loss, sub_batch, outputs["structure_module"], model_config.model.heads.structure_module)
  return loss["loss"]

####################
# loss functions (restricted to idx and/or sidechains)
####################
def get_dgram_loss_idx(batch, outputs, idx, model_config):
  idx_ref = batch["idx"]
  pb, pb_mask = model.modules.pseudo_beta_fn(batch["aatype"][idx_ref],
                                             batch["all_atom_positions"][idx_ref],
                                             batch["all_atom_mask"][idx_ref])
  
  dgram_loss = model.modules._distogram_log_loss(outputs["distogram"]["logits"][:,idx][idx,:],
                                                 outputs["distogram"]["bin_edges"],
                                                 batch={"pseudo_beta":pb,"pseudo_beta_mask":pb_mask},
                                                 num_bins=model_config.model.heads.distogram.num_bins)
  return dgram_loss["loss"]

def get_fape_loss_idx(batch, outputs, idx, model_config, backbone=False, sidechain=True, use_clamped_fape=False):
  idx_ref = batch["idx"]
  
  sub_batch = batch.copy()
  sub_batch.pop("idx")
  sub_batch = jax.tree_map(lambda x: x[idx_ref,...],sub_batch)
  sub_batch["use_clamped_fape"] = use_clamped_fape
  
  value = jax.tree_map(lambda x: x, outputs["structure_module"])
  loss = {"loss":0.0}
  
  if sidechain:
    value.update(folding.compute_renamed_ground_truth(sub_batch, value['final_atom14_positions'][idx,...]))
    value['sidechains']['frames'] = jax.tree_map(lambda x: x[:,idx,:], value["sidechains"]["frames"])
    value['sidechains']['atom_pos'] = jax.tree_map(lambda x: x[:,idx,:], value["sidechains"]["atom_pos"])
    loss.update(folding.sidechain_loss(sub_batch, value, model_config.model.heads.structure_module))
  
  if backbone:
    value["traj"] = value["traj"][...,idx,:]
    folding.backbone_loss(loss, sub_batch, value, model_config.model.heads.structure_module)

  return loss["loss"]

def get_sidechain_rmsd_idx(batch, outputs, idx, model_config, include_ca=False):
  idx_ref = batch["idx"]
  bb_atoms_to_exclude = ["N","O"] if include_ca else ["N","CA","O"]

  def kabsch(P, Q):
    V, S, W = jnp.linalg.svd(P.T @ Q, full_matrices=False)
    flip = jax.nn.sigmoid(-10 * jnp.linalg.det(V) * jnp.linalg.det(W))
    S = flip * S.at[-1].set(-S[-1]) + (1-flip) * S
    V = flip * V.at[:,-1].set(-V[:,-1]) + (1-flip) * V
    return V@W

  true_aa_idx = batch["aatype"][idx_ref]
  true_pos = all_atom.atom37_to_atom14(batch["all_atom_positions"],batch)[idx_ref,:,:]
  pred_pos = outputs["structure_module"]["final_atom14_positions"][idx,:,:]

  i,j,j_alt = [],[],[]
  i_non,j_non = [],[]
  for n,aa_idx in enumerate(true_aa_idx):
    aa = idx_to_resname[aa_idx]
    atoms = residue_constants.residue_atoms[aa].copy()
    for atom in atoms:
      if atom not in bb_atoms_to_exclude:
        i.append(n)
        j.append(residue_constants.restype_name_to_atom14_names[aa].index(atom))
        if aa in residue_constants.residue_atom_renaming_swaps:
          swaps = residue_constants.residue_atom_renaming_swaps[aa]
          swaps_rev = {v:k for k,v in swaps.items()}
          if atom in swaps:
            j_alt.append(residue_constants.restype_name_to_atom14_names[aa].index(swaps[atom]))
          elif atom in swaps_rev:
            j_alt.append(residue_constants.restype_name_to_atom14_names[aa].index(swaps_rev[atom]))
          else:
            j_alt.append(j[-1])
            i_non.append(i[-1])
            j_non.append(j[-1])
        else:
          j_alt.append(j[-1])
          i_non.append(i[-1])
          j_non.append(j[-1])

  # align non-ambigious atoms
  true_pos_non = true_pos[i_non,j_non,:]  
  pred_pos_non = pred_pos[i_non,j_non,:]
  true_pos = (true_pos - true_pos_non.mean(0)) @ kabsch(true_pos_non - true_pos_non.mean(0), pred_pos_non - pred_pos_non.mean(0))
  pred_pos = pred_pos - pred_pos_non.mean(0)

  true_pos_a = true_pos[i,j,:]
  pred_pos_a = pred_pos[i,j,:]
  pred_pos_b = pred_pos[i,j_alt,:]

  rms_a = jnp.square(true_pos_a - pred_pos_a).sum(-1)
  rms_b = jnp.square(true_pos_a - pred_pos_b).sum(-1)

  return jnp.sqrt(jnp.minimum(rms_a,rms_b).mean() + 1e-8)

####################
# update sequence
####################
def soft_seq(seq_logits, temp=1.0, hard=True):
  seq_soft = jax.nn.softmax(seq_logits / temp)
  if hard:
    seq_hard = jax.nn.one_hot(seq_soft.argmax(-1),20)
    return jax.lax.stop_gradient(seq_hard - seq_soft) + seq_soft
  else:
    return seq_soft

def update_seq(seq, inputs, seq_1hot=None, seq_pssm=None, msa_input=None):
  '''update the sequence features'''
  
  if seq_1hot is None: seq_1hot = seq 
  if seq_pssm is None: seq_pssm = seq
  msa_feat = jnp.zeros_like(inputs["msa_feat"]).at[...,0:20].set(seq_1hot).at[...,25:45].set(seq_pssm)
  if seq.ndim == 3:
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
  while seq.ndim > 1: seq = seq[0]
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
template_aa_map = np.eye(20)[[residue_constants.HHBLITS_AA_TO_ID[order_restype[i]] for i in range(20)]].T

###########################
# MISC
###########################
jalview_color_list = {"Clustal":           ["#80a0f0","#f01505","#00ff00","#c048c0","#f08080","#00ff00","#c048c0","#f09048","#15a4a4","#80a0f0","#80a0f0","#f01505","#80a0f0","#80a0f0","#ffff00","#00ff00","#00ff00","#80a0f0","#15a4a4","#80a0f0"],
                      "Zappo":             ["#ffafaf","#6464ff","#00ff00","#ff0000","#ffff00","#00ff00","#ff0000","#ff00ff","#6464ff","#ffafaf","#ffafaf","#6464ff","#ffafaf","#ffc800","#ff00ff","#00ff00","#00ff00","#ffc800","#ffc800","#ffafaf"],
                      "Taylor":            ["#ccff00","#0000ff","#cc00ff","#ff0000","#ffff00","#ff00cc","#ff0066","#ff9900","#0066ff","#66ff00","#33ff00","#6600ff","#00ff00","#00ff66","#ffcc00","#ff3300","#ff6600","#00ccff","#00ffcc","#99ff00"],
                      "Hydrophobicity":    ["#ad0052","#0000ff","#0c00f3","#0c00f3","#c2003d","#0c00f3","#0c00f3","#6a0095","#1500ea","#ff0000","#ea0015","#0000ff","#b0004f","#cb0034","#4600b9","#5e00a1","#61009e","#5b00a4","#4f00b0","#f60009","#0c00f3","#680097","#0c00f3"],
                      "Helix Propensity":  ["#e718e7","#6f906f","#1be41b","#778877","#23dc23","#926d92","#ff00ff","#00ff00","#758a75","#8a758a","#ae51ae","#a05fa0","#ef10ef","#986798","#00ff00","#36c936","#47b847","#8a758a","#21de21","#857a85","#49b649","#758a75","#c936c9"],
                      "Strand Propensity": ["#5858a7","#6b6b94","#64649b","#2121de","#9d9d62","#8c8c73","#0000ff","#4949b6","#60609f","#ecec13","#b2b24d","#4747b8","#82827d","#c2c23d","#2323dc","#4949b6","#9d9d62","#c0c03f","#d3d32c","#ffff00","#4343bc","#797986","#4747b8"],
                      "Turn Propensity":   ["#2cd3d3","#708f8f","#ff0000","#e81717","#a85757","#3fc0c0","#778888","#ff0000","#708f8f","#00ffff","#1ce3e3","#7e8181","#1ee1e1","#1ee1e1","#f60909","#e11e1e","#738c8c","#738c8c","#9d6262","#07f8f8","#f30c0c","#7c8383","#5ba4a4"],
                      "Buried Index":      ["#00a35c","#00fc03","#00eb14","#00eb14","#0000ff","#00f10e","#00f10e","#009d62","#00d52a","#0054ab","#007b84","#00ff00","#009768","#008778","#00e01f","#00d52a","#00db24","#00a857","#00e619","#005fa0","#00eb14","#00b649","#00f10e"]}

