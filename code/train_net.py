"""
The `fit` function in this file implements a slightly modified version
of the Keras `model.fit()` API.
"""
import torch
from typing import Callable, List, Union

import torchvision.models

from callbacks import ProgressBarLogger, CallbackList, Callback
from framework_utils import make_cuda
import framework_utils
from torch._six import inf
from models.sequence_learner import SequenceMatchingNetSimple
import framework_utils
# from torchviz import make_dot
import matplotlib.pyplot as plt
import torchvision
EPSILON = 1e-8


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm


def standard_net_step(data, model, loss_fn, optimizer, use_cuda, loader, train):
    logs = {}
    images, labels, more = data
    images = make_cuda(images, use_cuda)
    labels = make_cuda(labels, use_cuda)
    if train:
        optimizer.zero_grad()

    output_batch = model(images)

    # framework_utils.imshow_batch(images, stats=loader.dataset.stats)
    if model.training and isinstance(model, torchvision.models.GoogLeNet):
        output_batch = output_batch.logits
    loss = loss_fn(output_batch,
                   labels)
    logs['output'] = output_batch
    logs.update(more)
    # logs['more'] = more
    # logs['images'] = images

    predicted = torch.argmax(output_batch, -1)
    if train:
        loss.backward()
        optimizer.step()

    return loss, labels, predicted, logs


def sequence_net_Ntrain_1cand(data, model: SequenceMatchingNetSimple, loss_fn, optimizer, use_cuda, loader, train, concatenate=False):
    k = dataset.sampler.k
    nSc = dataset.sampler.nSc
    nSt = dataset.sampler.nSt
    nFc = dataset.sampler.nFc
    nFt = dataset.sampler.nFt
    y_matching_labels = dataset.sampler.labels

    camera_positions = dataset.sampler.camera_positions

    # print(f"IS CUDA: {next(model.parameters()).is_cuda}")
    if train:
        optimizer.zero_grad()

    logs = {}
    assert nSc == 1 and nFc == 1
    x, _, _ = data

    # framework_utils.imshow_batch(x[0:50], stats=dataset.stats, title_lab=y_matching_labels)

    y_matching_correct = torch.tensor([1 if i[0] == i[1] else 0 for i in y_matching_labels], dtype=torch.float)
    logs['images'] = x

    x = make_cuda(x, use_cuda)
    relation_scores = model((x, k, nSt, nFt, use_cuda))

    rs = make_cuda(relation_scores.squeeze(1), use_cuda)
    lb = make_cuda(y_matching_correct, use_cuda)
    loss = loss_fn(rs, lb)
    y_matching_predicted = torch.tensor(rs > 0.5, dtype=torch.int).T

    logs['output'] = relation_scores
    logs['labels'] = y_matching_labels
    logs['camera_positions'] = camera_positions

    if train:
        loss.backward()
        # clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return loss, y_matching_correct, y_matching_predicted, logs


def relation_net_step(data, model, loss_fn, optimizer, use_cuda, loader, train, n_shot, k_way, q_queries, concatenate=False):
    logs = {}
    x, y_real_labels, more = data
    if train:
        optimizer.zero_grad()

    # framework_utils.imshow_batch(x)
    assert len(x) == n_shot * k_way + k_way * q_queries
    y_onehot = torch.zeros(q_queries * k_way, k_way)
    y = torch.arange(0, k_way, 1 / q_queries)
    y_onehot = y_onehot.scatter(1, y.unsqueeze(-1).long(), 1)

    embeddings = model.encoder(make_cuda(x, use_cuda))
    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot * k_way]
    queries = embeddings[n_shot * k_way:]

    batch = make_cuda(torch.tensor([]), use_cuda)

    if concatenate:
        # concatenate all features (used in relation_net_cat)
        summed_supp = support.view(k_way, n_shot, *support.size()[-3:]).reshape(k_way, support.shape[1] * n_shot, *support.shape[-2:])
    else:
        # sum across n_shots as in classic relation_net
        summed_supp = support.view(k_way, n_shot, *support.size()[-3:]).sum(dim=1)

    # this could be done without for loop, but for now it seems to be fast enough
    for idx, q in enumerate(queries):
        # K_WAY x 64 * (K_WAY + 1) x 56 x 56
        episode = torch.cat((summed_supp, q.expand(k_way, -1, -1, -1)), dim=1)
        batch = torch.cat((batch, make_cuda(episode, use_cuda)))

    output = model.relation_net(batch).view((k_way, -1)).T

    loss = loss_fn(make_cuda(output, use_cuda),
                   make_cuda(y_onehot, use_cuda))

    _, predicted = output.max(dim=1)

    selected_classes = y_real_labels[n_shot * k_way::q_queries]
    prediction_real_labels = selected_classes[predicted]
    queries_real_labels = y_real_labels[n_shot * k_way:]
    logs['output'] = output
    logs['more'] = more
    logs['more']['center'] = [i[n_shot * k_way:] for i in logs['more']['center']]

    if train:
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return loss, queries_real_labels, prediction_real_labels, logs


def run(data_loader, use_cuda, net, callbacks: List[Callback] = None, optimizer=None, loss_fn=None, iteration_step=standard_net_step, iteration_step_kwargs=None):
    torch.cuda.empty_cache()

    if iteration_step_kwargs is None:
        iteration_step_kwargs = {}

    make_cuda(net, use_cuda)

    callbacks = CallbackList(callbacks)
    callbacks.set_model(net)
    callbacks.set_optimizer(optimizer)
    callbacks.set_loss_fn(loss_fn)

    logs = {}
    callbacks.on_train_begin()

    tot_iter = 0
    epoch = 0
    logs['tot_iter'] = 0

    while True:
        callbacks.on_epoch_begin(epoch)
        logs['epoch'] = epoch

        for batch_index, data in enumerate(data_loader, 0):

            callbacks.on_batch_begin(batch_index, logs)

            loss, y_true, y_pred, new_batch_logs = iteration_step(data, net, loss_fn, optimizer, use_cuda, **iteration_step_kwargs)

            logs.update({'y_pred': y_pred, 'loss': loss.item(), 'y_true': y_true, 'tot_iter': tot_iter, 'stop': False, **new_batch_logs})

            callbacks.on_training_step_end(batch_index, logs)
            callbacks.on_batch_end(batch_index, logs)
            if logs['stop']:
                break
            tot_iter += 1


        callbacks.on_epoch_end(epoch, logs)
        epoch += 1
        if logs['stop']:
            break

    callbacks.on_train_end(logs)
    return net, logs
