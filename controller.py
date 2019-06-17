"""A module with NAS controller-related code."""
import collections
import os

import torch
import torch.nn.functional as F

import utils


class Controller(torch.nn.Module):
    """Based on
    https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    TODO(brendan): RL controllers do not necessarily have much to do with
    language models.

    Base the controller RNN on the GRU from:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
    """
    def __init__(self, args):
        torch.nn.Module.__init__(self)
#         args.cuda = True
        self.args = args
        self.controller_hid = 100
        self.num_tokens = [9, 24, 9]###

        num_total_tokens = self.num_tokens[1]

        self.encoder = torch.nn.Embedding(num_total_tokens, self.controller_hid)
        self.lstm = torch.nn.LSTMCell(self.controller_hid, self.controller_hid)

        self.decoders = []

        for i in range(self.num_tokens[0]):
            decoder = torch.nn.Linear(self.controller_hid, self.num_tokens[1])
            self.decoders.append(decoder)
            decoder = torch.nn.Linear(self.controller_hid, self.num_tokens[2])
            self.decoders.append(decoder)

        self._decoders = torch.nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)

        def _get_default_hidden(key):
            return utils.get_variable(
                torch.zeros(key, self.controller_hid),
                self.args.cuda,
                requires_grad=False)

        self.static_inputs = utils.keydefaultdict(_get_default_hidden)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self,  # pylint:disable=arguments-differ
                inputs,
                hidden,
                block_idx,
                is_embed,
                is_train=True):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        logits /= 5#self.args.softmax_temperature

        # exploration
        if is_train:
            logits = (2.5*F.tanh(logits))#self.args.tanh_c

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None, is_train=True):
        """Samples a set of `args.num_blocks` many computational nodes from the
        controller, where each node is made up of an activation function, and
        each node except the last also includes a previous node.
        """

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        entropies = []
        log_probs = []
        policy = []
        
#         for block_idx in range(self.num_tokens[0]):###
#             logits, hidden = self.forward(inputs,
#                                           hidden,
#                                           block_idx,
#                                           is_embed=(block_idx == 0),
#                                           is_train=(True and is_train))

#             probs = F.softmax(logits, dim=-1)
#             log_prob = F.log_softmax(logits, dim=-1)

#             entropy = -(log_prob * probs).sum(1, keepdim=False)

#             action = probs.multinomial(num_samples=1).data
#             policy.append(action.item())
#             selected_log_prob = log_prob.gather(
#                 1, utils.get_variable(action, self.args.cuda, requires_grad=False))

#             entropies.append(entropy)
#             log_probs.append(selected_log_prob[:, 0])

#             inputs = utils.get_variable(action[:, 0], self.args.cuda, requires_grad=False)
        
        
        for block_idx in range(self.num_tokens[0]):
            block_policy = []
            ### sample transmission policy
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx*2,
                                          is_embed=(block_idx == 0),
                                          is_train=(True and is_train))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)

            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            block_policy.append(action.item())
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, self.args.cuda, requires_grad=False))

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            inputs = utils.get_variable(action[:, 0], self.args.cuda, requires_grad=False)
            
            ### sample reduction policy
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          block_idx*2+1,
                                          is_embed=False,
                                          is_train=(True and is_train))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)

            entropy = -(log_prob * probs).sum(1, keepdim=False)

            action = probs.multinomial(num_samples=1).data
            block_policy.append(action.item())
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, self.args.cuda, requires_grad=False))

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            inputs = utils.get_variable(action[:, 0], self.args.cuda, requires_grad=False)
            
            policy.append(block_policy)

        if with_details:
            return policy, torch.cat(log_probs), torch.cat(entropies)

        return policy

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.args.cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.args.cuda, requires_grad=False))
