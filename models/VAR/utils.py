import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kld(recog_mu, recog_std, prior_mu, prior_std):
    kld = -0.5 * torch.sum(torch.log(torch.div(torch.pow(recog_std, 2), torch.pow(prior_std, 2)) + 1e-6)
                           - torch.div(torch.pow(recog_std, 2), torch.pow(prior_std, 2))
                           - torch.div(torch.pow(recog_mu - prior_mu, 2), torch.pow(prior_std, 2)) + 1, dim=-1)
    return kld


def gen_bow_loss(scores, bow_mask, tgt):
    """
    scores    torch.Size([batch_size, seq_len, vocab_size])
    bow_mask  torch.Size([batch_size, seq_len, seq_len])
    tgt       torch.Size([batch_size, seq_len])
    """
    voc_size = scores.shape[-1]
    new_tgt = tgt.unsqueeze(1).expand_as(bow_mask).clone()  # tgt[0][bow_mask[0,0]]
    new_tgt[~bow_mask] = 1  # torch.Size([137, 119, 119]) [bs, seq_len, seq_len]
    # y = torch.full(size=(1, 1, voc_size), fill_value=-1., device=scores.device).expand_as(scores).clone()
    #     # torch.Size([137, 119, 53])
    # # y[new_tgt] = 1.
    # bs, seq_len = tgt.shape
    # for i in range(bs):
    #     for j in range(seq_len):
    #         y[i, j][new_tgt[i, j]] = 1.
    # aux_loss = torch.log(1 + torch.exp(-y * scores))
    device = scores.device
    y_size = (new_tgt.shape[0], new_tgt.shape[1], max(new_tgt.shape[2], voc_size))
    tmp = torch.zeros(y_size, dtype=scores.dtype, device=device).scatter_add(2, new_tgt, torch.ones(y_size, dtype=scores.dtype, device=device))
    y_new = tmp[..., :voc_size]  # count based bag of words
    y_new[..., 1] = -1.  # token '1'--> -1.
    # import ipdb; ipdb.set_trace()
    y_new[y_new > 0.5] = 1.  # freqs --> 1
    y_new[y_new < 0.5] = -1.  # 0--> -1.
    bow_loss = torch.log(1 + torch.exp(-y_new * scores))
    # bow_loss = bow_loss.mean()  # * 4e2
    return bow_loss


def gen_tgt_z(tgt_mask, tgt):  # tgt_mask torch.Size([64, 1, 112])  tgt torch.Size([64, 112])
    bs, seq_len = tgt.shape
    pads = torch.full(size=(bs, 1), fill_value=0, dtype=torch.int64, device=tgt.device) # padding 0
    mask = tgt_mask.squeeze(1)
    tgt_lens = (~mask).sum(dim=1)  # bs torch.Size([64])
    pads[tgt_lens == seq_len] = 3  # </s> _END
    return torch.cat([tgt[:, 1:], pads], dim=-1)


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    #print(logits[0],"a")
    #print(len(argmax_acs),argmax_acs[0])
    if eps == 0.0:
        return argmax_acs

# # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
# def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
#     """Sample from Gumbel(0, 1)"""
#     # U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
#     U = nn.parameter.Parameter(tens_type(*shape).uniform_(), requires_grad=False)
#     return -torch.log(-torch.log(U + eps) + eps)

# # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
# def gumbel_softmax_sample(logits, temperature):
#     """ Draw a sample from the Gumbel-Softmax distribution"""
#     # y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(device=logits.device)
#     y = logits #+ sample_gumbel(logits.shape, tens_type=type(logits.data)).to(device=logits.device)
#     return F.softmax(y / temperature, dim=1)

# # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
# def gumbel_softmax(logits, temperature=1.0, hard=False):
#     """Sample from the Gumbel-Softmax distribution and optionally discretize.
#     Args:
#       logits: [batch_size, n_class] unnormalized log-probs
#       temperature: non-negative scalar
#       hard: if True, take argmax, but differentiate w.r.t. soft sample y
#     Returns:
#       [batch_size, n_class] sample from the Gumbel-Softmax distribution.
#       If hard=True, then the returned sample will be one-hot, otherwise it will
#       be a probabilitiy distribution that sums to 1 across classes
#     """
#     y = gumbel_softmax_sample(logits, temperature)
#     if hard:
#         y_hard = onehot_from_logits(y)
#         #print(y_hard[0], "random")
#         y = (y_hard - y).detach() + y
#     return y


class layerN(nn.Module):
    """
        Layer Normalization class
    """

    def __init__(self, features, eps=1e-6):
        super(layerN, self).__init__()
        self.a_2 = torch.ones(features)
        self.b_2 = torch.zeros(features)
        self.eps = eps

    def forward(self, x):
        device = x.device
        self.a_2, self.b_2 = self.a_2.to(device=device), self.b_2.to(device=device)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Scaler(nn.Module):
    def __init__(self, d_size, tau=0.5):
        super(Scaler, self).__init__()
        self.tau = tau
        # self.scale = nn.parameter.Parameter(torch.Tensor(d_size, ))
        self.scale = nn.parameter.Parameter(torch.Tensor(1, d_size))
        nn.init.xavier_uniform_(self.scale)

    def forward(self, inputs, mode='positive'):
        if mode == 'positive':
            # import ipdb; ipdb.set_trace()
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        return inputs * torch.sqrt(scale)


class Con_Layer_Norm(nn.Module):
    """
        Conditional Layer Normalization class
    """

    def __init__(self, features, d_size, eps=1e-6):
        super(Con_Layer_Norm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # g
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
        self.emb_g = nn.Linear(d_size, features, bias=False)
        self.emb_b = nn.Linear(d_size, features, bias=False)
        nn.init.zeros_(self.emb_g.weight)
        nn.init.zeros_(self.emb_b.weight)

    def forward(self, x, condition=None):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if condition is not None:
            # if condition.dim() != x.dim():
            #     condition = condition.unsqueeze(1)
            #++++++++++++++++++++
            if condition.size() != x.size():
                condition = torch.mean(condition, dim=1, keepdim=True)
            #++++++++++++++++++++
            gamma = self.emb_g(condition)
            beta = self.emb_b(condition)
            return (self.a_2 + gamma) * (x - mean) / (std + self.eps) + (self.b_2 + beta)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class VarFeedForward(nn.Module):
    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', dropout=0.0):
        super(VarFeedForward, self).__init__()

        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.tanh(x)
                x = self.drop(x)
        return x


class FeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, d_out, dropout=0.1):
        super(FeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        # inter = self.dropout_1(self.tanh(self.w_1(self.layer_norm(x))))
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output


# borrow from onmt
def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def mycriterion(input, batch, mask_id=0):
    ts = batch.multi_tgts
    max_len = len(ts)//len(batch.data_indice)
    rep_inp=tile(input.contiguous(), count=max_len, dim=0)
    loss=nn.CrossEntropyLoss(ignore_index=mask_id, reduction="none")(rep_inp, ts).mean(-1)
    loss[loss<1e-10]=1e3
    return torch.stack(loss.chunk(len(batch.data_indice),dim=0),dim=0).min(dim=0)[0].mean()

# borrow from https://github1s.com/bojone/vae/blob/master/vae_keras_cnn_gs.py
def GumbelSoftmax(logits, tau=.8, noise=1e-20):
    eps = torch.rand(size=logits.shape, device=logits.device) # uniform distribution on the interval [0, 1)
    outputs = logits - torch.log(-torch.log(eps + noise) + noise)
    return torch.softmax(outputs / tau, -1)


class LatFFN(nn.Module):
    def __init__(self, d_model, d_out, dropout=0.1):
        super(LatFFN, self).__init__()
        self.w = nn.Linear(d_model, d_out)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = self.dropout(self.tanh(self.w(self.layer_norm(x))))
        return output


# borrow from https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
def sample_gumbel(shape, device, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = torch.rand(size=shape, device=device) # uniform distribution on the interval [0, 1)
  return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1.0): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(logits.shape, logits.device)
  return torch.softmax( y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    # k = logits.shape[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    # y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    # y = tf.stop_gradient(y_hard - y) + y
    y_hard = onehot_from_logits(y)
    y = (y_hard - y).detach() + y
  return y