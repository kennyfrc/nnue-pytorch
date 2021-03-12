import math
import chess
import ranger
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pdb

# 3 layer fully connected network
L1 = 256
L2 = 32
L3 = 32

class NNUE(pl.LightningModule):
  """
  This model attempts to directly represent the nodchip Stockfish trainer methodology.

  lambda_ = 0.0 - purely based on game results
  lambda_ = 1.0 - purely based on search scores

  It is not ideal for training a Pytorch quantized model directly.
  """
  def __init__(self, feature_set, lambda_=1.0, lrs_=[1e-3,1e-3,1e-3,1e-4], finetune=False):
    super(NNUE, self).__init__()
    self.input = nn.Linear(feature_set.num_features, L1)
    
    weights = self.input.weight.clone()
    kMaxActiveDimensions = 30 # kings don't count
    kSigma = 0.1 / math.sqrt(kMaxActiveDimensions)
    weights = weights.normal_(0.0, kSigma)
    biases = self.input.bias
    biases = biases.clone().fill_(0.5)
    self.input.weight = nn.Parameter(weights)
    self.input.bias = nn.Parameter(biases)

    self.feature_set = feature_set
    self.l1 = nn.Linear(2 * L1, L2)
    self.l2 = nn.Linear(L2, L3)
    self.output = nn.Linear(L3, 1)
    self.lambda_ = lambda_
    self.lrs_ = lrs_
    self.finetune = finetune

    self._zero_virtual_feature_weights()

  '''
  We zero all virtual feature weights because during serialization to .nnue
  we compute weights for each real feature as being the sum of the weights for
  the real feature in question and the virtual features it can be factored to.
  This means that if we didn't initialize the virtual feature weights to zero
  we would end up with the real features having effectively unexpected values
  at initialization - following the bell curve based on how many factors there are.
  '''
  def _zero_virtual_feature_weights(self):
    weights = self.input.weight
    
    with torch.no_grad():
        for a, b in self.feature_set.get_virtual_feature_ranges(): 
            weights[:, a:b] = 0.0
    self.input.weight = nn.Parameter(weights)

  '''
  This method attempts to convert the model from using the self.feature_set
  to new_feature_set.
  '''
  def set_feature_set(self, new_feature_set):
    if self.feature_set.name == new_feature_set.name:
      return

    # TODO: Implement this for more complicated conversions.
    #       Currently we support only a single feature block.
    if len(self.feature_set.features) > 1:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

    # Currently we only support conversion for feature sets with
    # one feature block each so we'll dig the feature blocks directly
    # and forget about the set.
    old_feature_block = self.feature_set.features[0]
    new_feature_block = new_feature_set.features[0]

    # next(iter(new_feature_block.factors)) is the way to get the
    # first item in a OrderedDict. (the ordered dict being str : int
    # mapping of the factor name to its size).
    # It is our new_feature_factor_name.
    # For example old_feature_block.name == "HalfKP"
    # and new_feature_factor_name == "HalfKP^"
    # We assume here that the "^" denotes factorized feature block
    # and we would like feature block implementers to follow this convention.
    # So if our current feature_set matches the first factor in the new_feature_set
    # we only have to add the virtual feature on top of the already existing real ones.
    if old_feature_block.name == next(iter(new_feature_block.factors)):
      # We can just extend with zeros since it's unfactorized -> factorized
      weights = self.input.weight
      padding = weights.new_zeros((weights.shape[0], new_feature_block.num_virtual_features))
      weights = torch.cat([weights, padding], dim=1)
      self.input.weight = nn.Parameter(weights)
      self.feature_set = new_feature_set
    else:
      raise Exception('Cannot change feature set from {} to {}.'.format(self.feature_set.name, new_feature_set.name))

  def forward(self, us, them, w_in, b_in):
    w = self.input(w_in)
    b = self.input(b_in)
    l0_ = (us * torch.cat([w, b], dim=1)) + (them * torch.cat([b, w], dim=1))
    # clamp here is used as a clipped relu to (0.0, 1.0)
    l0_ = torch.clamp(l0_, 0.0, 1.0)
    l1_ = torch.clamp(self.l1(l0_), 0.0, 1.0)
    l2_ = torch.clamp(self.l2(l1_), 0.0, 1.0)
    x = self.output(l2_)
    return x

  def on_after_backward(self):
    w = self.input.weight
    g = w.grad
    a = self.feature_set.features[0].get_factor_base_feature('HalfK')
    b = self.feature_set.features[0].get_factor_base_feature('P')
    g[:, a:b] /= 30.0

  def step_(self, batch, batch_idx, loss_type):    
    us, them, white, black, outcome, score = batch

    # 600 is the kPonanzaConstant scaling factor needed to convert the training net output to a score.
    # This needs to match the value used in the serializer
    # It just works
    nnue2score = 600
    pawnValueEg = 208
    scaling = pawnValueEg * 4 * math.log(10)

    # # MSE Loss for debugging
    # # predicted, outcome, teacher scores (in order)
    # p_score = (self(us, them, white, black) * nnue2score / scaling).sigmoid()
    # z_score = outcome
    # q_score = (score / scaling).sigmoid()

    # # teacher score
    # # only care about z when the outcome is a win/loss
    # # if there is a win/loss, only then shall you train against it
    # if self.lambda_ < 1:
    #   t_score = torch.where(
    #     torch.logical_or(z_score.eq(1.0),z_score.eq(0.0)),
    #     (q_score * self.lambda_) + (z_score * (1.0 - self.lambda_)),
    #     q_score
    #   )
    # else:
    #   t_score = q_score

    # loss = F.mse_loss(p_score, t_score)
    # self.log(loss_type, "mse_loss")

    # Cross-Entropy Loss
    q = self(us, them, white, black) * nnue2score / scaling
    t = outcome
    p = (score / scaling).sigmoid()

    epsilon = 1e-12
    teacher_entropy = -(p * (p + epsilon).log() + (1.0 - p) * (1.0 - p + epsilon).log())
    teacher_loss = -(p * F.logsigmoid(q) + (1.0 - p) * F.logsigmoid(-q))
    
    outcome_entropy = -(t * (t + epsilon).log() + (1.0 - t) * (1.0 - t + epsilon).log())
    outcome_loss = -(t * F.logsigmoid(q) + (1.0 - t) * F.logsigmoid(-q))

    def get_means(lambda_, t_loss, o_loss, t_entropy, o_entropy, drawn_game):
      if drawn_game:
        result  = teacher_loss    
        entropy = teacher_entropy
      else:
        result  = self.lambda_ * teacher_loss    + (1.0 - self.lambda_) * outcome_loss
        entropy = self.lambda_ * teacher_entropy + (1.0 - self.lambda_) * outcome_entropy
      return result.mean(), entropy.mean()

    if self.lambda_ < 1:
      result, entropy = torch.where(
        torch.logical_or(t.eq(1.0),t.eq(0.0)),
         get_means(self.lambda_, teacher_loss, outcome_loss,
                                    teacher_entropy, outcome_entropy, drawn_game=False),
         get_means(self.lambda_, teacher_loss, outcome_loss,
                                    teacher_entropy, outcome_entropy, drawn_game=True)
      )
    else:
      result  = teacher_loss.mean()
      entropy = teacher_entropy.mean()
    
    loss = result - entropy
    self.log(loss_type, loss)

    return loss

  def training_step(self, batch, batch_idx):
    return self.step_(batch, batch_idx, 'train_loss')

  def validation_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'val_loss')

  def test_step(self, batch, batch_idx):
    self.step_(batch, batch_idx, 'test_loss')

  def on_load_checkpoint(self, checkpoint):
    if self.finetune:
      LRs = self.lrs_

      lr_schedulers = checkpoint['lr_schedulers'][0]

      # reset
      checkpoint['epoch'] = 0
      checkpoint['global_step'] = 0
      lr_schedulers['base_lrs'] = LRs
      lr_schedulers['last_lrs'] = LRs 
      lr_schedulers['last_epoch'] = 0

      # assume cosine annealing LR
      lr_schedulers['T_cur'] = 0
      lr_schedulers['T_0'] = 10

      param_groups = checkpoint['optimizer_states'][0]['param_groups']
      
      for idx, param_group in enumerate(param_groups):
          param_group['lr'] = LRs[idx]
          param_group['initial_lr'] = LRs[idx]


  def configure_optimizers(self):
    LRs = self.lrs_

    train_params = [
      {'params': self.get_layers(lambda x: self.input == x), 'lr': LRs[0] },
      {'params': self.get_layers(lambda x: self.l1 == x), 'lr': LRs[1] },
      {'params': self.get_layers(lambda x: self.l2 == x), 'lr': LRs[2] },
      {'params': self.get_layers(lambda x: self.output == x), 'lr': LRs[3] },
    ]

    optimizer = ranger.Ranger(train_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, verbose=True)

    return [optimizer], [scheduler]

  def get_layers(self, filt):
    """
    Returns a list of layers.
    filt: Return true to include the given layer.
    """
    for i in self.children():
      if filt(i):
        if isinstance(i, nn.Linear):
          for p in i.parameters():
            if p.requires_grad:
              yield p
