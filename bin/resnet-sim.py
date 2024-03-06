# %%
import math
import typing as ty
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
from torch.utils.tensorboard import SummaryWriter
import zero
from torch import Tensor

import lib

writer = SummaryWriter()


# %%
class ResNet(nn.Module):
    def __init__(
            self,
            *,
            d_numerical: int,
            categories: ty.Optional[ty.List[int]],
            d_embedding: int,
            d: int,
            d_hidden_factor: float,
            n_layers: int,
            activation: str,
            normalization: str,
            hidden_dropout: float,
            residual_dropout: float,
            d_out: int,
    ) -> None:
        super().__init__()

        def make_normalization():
            return {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm}[
                normalization
            ](d)

        self.main_activation = lib.get_activation_fn(activation)
        self.last_activation = lib.get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_in = d_numerical
        d_hidden = int(d * d_hidden_factor)

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        'norm': make_normalization(),
                        'linear0': nn.Linear(
                            d, d_hidden * (2 if activation.endswith('glu') else 1)
                        ),
                        'linear1': nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

        self.K = 0.01
        self.update_method = 'random'
        self.k_similar = 0.1
        self.bits = 1.5
        self.sparsity = 0.9

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer['norm'](z)
            z = layer['linear0'](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer['linear1'](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        x = x.squeeze(-1)
        return x

    def update_weights_in_module(self, m):
        if "Linear" in m.__class__.__name__:
            remove_smallest_weights(m, self.K)
            if self.update_method == 'faiss':
                weight_magic_faiss(m, self.K, self.k_similar, self.bits)
            elif self.update_method == 'random':
                weight_magic_random(m, self.K, self.k_similar)
            elif self.update_method == 'bruteforce':
                weight_magic(m, self.K, self.k_similar)
            else:
                raise RuntimeError(f"unknown update method: {self.update_method}")


def remove_smallest_weights(layer, K):
    K = int(layer.weight.numel() * K)
    weights = np.abs(layer.weight.detach().cpu().numpy().reshape(-1))
    weights[layer.weight_mask.detach().cpu().numpy().reshape(-1) == 0] = np.inf
    indices = np.unravel_index(np.argpartition(weights, K)[:K], layer.weight.shape)
    layer.weight[indices] = 0
    layer.weight_mask[indices] = 0


def weight_magic_random(conv, K, k_similar):
    # print("random")
    delattr(conv, "input")
    delattr(conv, "grad_output")

    K = int(conv.weight.numel() * K)

    indices = torch.nonzero(torch.where(conv.weight_mask == 0, 1, 0))
    p = torch.ones(indices.shape[0], device=device)
    to_randn = tuple(indices[p.multinomial(K, False)].T)

    conv.weight[to_randn] = 0
    conv.weight_mask[to_randn] = 1


def weight_magic_faiss(layer, K, k_similar, bits):
    # print("faiss")
    x = layer.input
    batch_size = x.shape[0]
    out = layer.grad_output

    # print(layer.weight.shape)
    # print(x.shape, out.shape)

    delattr(layer, "input")
    delattr(layer, "grad_output")

    K = int(layer.weight.numel() * K)

    x = x.detach().cpu().numpy().T
    out = out.detach().cpu().numpy().T

    index = faiss.IndexFlatIP(batch_size)  # , int(bits * batch_size)
    index.train(x)
    index.add(x)

    k_similar = int(x.shape[0] * 1)

    D, I = index.search(out, k_similar)
    D = D.reshape(-1)
    D = np.abs(D)

    # print(D, D.shape)
    # print(I, I.shape)

    idx = np.array(
        (np.repeat(np.arange(layer.weight.shape[0]), k_similar).reshape(layer.weight.shape[0], k_similar),
         *np.unravel_index(I, layer.weight.shape[1]))).reshape(2, -1)
    # print(idx, idx.shape)

    D[layer.weight_mask[idx[0], idx[1]].detach().cpu().numpy() == 1] = np.inf

    # print(D.shape, idx.shape, K, layer.weight.numel(), out.shape, x.shape)
    idx = idx.T[np.argpartition(D, -K)[-K:]]
    to_randn = idx.T

    # print(to_randn, to_randn.shape)
    layer.weight[to_randn[0], to_randn[1]] = 0
    layer.weight_mask[to_randn[0], to_randn[1]] = 1


# %%
if __name__ == "__main__":
    args, output = lib.load_config()


    def _backward_hook(module, grad_input, grad_output):  # module, grad_input, grad_output
        with torch.no_grad():
            if not hasattr(module, "grad_output"):
                module.grad_output = torch.zeros([args['training']['batch_size']] + list(grad_output[0].shape)[1:],
                                                 device=device)
            if module.grad_output.shape[0] == grad_output[0].shape[0]:
                module.grad_output += grad_output[0]


    def _forward_hook(module, input, output):
        with torch.no_grad():
            if not hasattr(module, "input"):
                module.input = torch.zeros([args['training']['batch_size']] + list(input[0].shape)[1:], device=device)
            if module.input.shape[0] == input[0].shape[0]:
                module.input += input[0]


    # %%
    zero.set_randomness(args['seed'])
    dataset_dir = lib.get_path(args['data']['path'])
    stats: ty.Dict[str, ty.Any] = {
        'dataset': dataset_dir.name,
        'algorithm': Path(__file__).stem,
        **lib.load_json(output / 'stats.json'),
    }
    timer = zero.Timer()
    timer.run()

    D = lib.Dataset.from_dir(dataset_dir)
    X = D.build_X(
        normalization=args['data'].get('normalization'),
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy=args['data'].get('cat_policy', 'indices'),
        cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
        seed=args['seed'],
    )
    if not isinstance(X, tuple):
        X = (X, None)

    zero.set_randomness(args['seed'])
    Y, y_info = D.build_y(args['data'].get('y_policy'))
    lib.dump_pickle(y_info, output / 'y_info.pickle')
    X = tuple(None if x is None else lib.to_tensors(x) for x in X)
    Y = lib.to_tensors(Y)
    device = lib.get_device()
    if device.type != 'cpu':
        X = tuple(
            None if x is None else {k: v.to(device) for k, v in x.items()} for x in X
        )
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y
    X_num, X_cat = X
    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    train_size = D.size(lib.TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )
    args["model"]["d_embedding"] = args["model"].get("d_embedding", None)

    model = ResNet(
        d_numerical=0 if X_num is None else X_num['train'].shape[1],
        categories=lib.get_categories(X_cat),
        d_out=D.info['n_classes'] if D.is_multiclass else 1,
        **args['model'],
    ).to(device)


    def init_weights(m):
        if hasattr(m, "weight") and "Linear" in m.__class__.__name__:
            # print(m.__class__.__name__)
            # print(m.weight)
            prune.random_unstructured(m, 'weight', amount=model.sparsity)
            m.register_forward_hook(_forward_hook)
            m.register_backward_hook(_backward_hook)


    model.apply(init_weights)

    stats['n_parameters'] = lib.get_n_parameters(model)
    optimizer = lib.make_optimizer(
        args['training']['optimizer'],
        model.parameters(),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args['training']['patience'])
    training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
    timer = zero.Timer()
    checkpoint_path = output / 'checkpoint.pt'


    def print_epoch_info():
        print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())} | {output}')
        print(
            ' | '.join(
                f'{k} = {v}'
                for k, v in {
                    'lr': lib.get_lr(optimizer),
                    'batch_size': batch_size,
                    'epoch_size': stats['epoch_size'],
                    'n_parameters': stats['n_parameters'],
                }.items()
            )
        )


    @torch.no_grad()
    def evaluate(parts):
        model.eval()
        metrics = {}
        predictions = {}
        for part in parts:
            predictions[part] = (
                torch.cat(
                    [
                        model(
                            None if X_num is None else X_num[part][idx],
                            None if X_cat is None else X_cat[part][idx],
                        )
                        for idx in lib.IndexLoader(
                        D.size(part),
                        args['training']['eval_batch_size'],
                        False,
                        device,
                    )
                    ]
                )
                .cpu()
                .numpy()
            )
            metrics[part] = lib.calculate_metrics(
                D.info['task_type'],
                Y[part].numpy(),  # type: ignore[code]
                predictions[part],  # type: ignore[code]
                'logits',
                y_info,
            )
        for part, part_metrics in metrics.items():
            print(f'[{part:<5}]', lib.make_summary(part_metrics))
        return metrics, predictions


    def save_checkpoint(final):
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stream': stream.state_dict(),
                'random_state': zero.get_random_state(),
                **{
                    x: globals()[x]
                    for x in [
                        'progress',
                        'stats',
                        'timer',
                        'training_log',
                    ]
                },
            },
            checkpoint_path,
        )
        lib.dump_stats(stats, output, final)
        lib.backup_output(output)


    # %%
    timer.run()
    for epoch in stream.epochs(args['training']['n_epochs']):
        print_epoch_info()

        model.train()
        epoch_losses = []
        for batch_idx in epoch:
            optimizer.zero_grad()
            loss = loss_fn(
                model(
                    None if X_num is None else X_num[lib.TRAIN][batch_idx],
                    None if X_cat is None else X_cat[lib.TRAIN][batch_idx],
                ),
                Y_device[lib.TRAIN][batch_idx],
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach())
            writer.add_scalar("loss by minibatch", epoch_losses[-1], stream.iteration)

            if stream.iteration % 20 == 0:
                model.apply(model.update_weights_in_module)

        epoch_losses = torch.stack(epoch_losses).tolist()
        training_log[lib.TRAIN].extend(epoch_losses)
        writer.add_scalar("loss by epoch", sum(epoch_losses) / len(epoch_losses), stream.epoch)
        print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')


        @torch.no_grad()
        def log_sparsity(m):
            print(m.__class__.__name__)
            if hasattr(m, "weight"):
                print("   ", m.__class__.__name__,
                      torch.count_nonzero(m.weight.detach().cpu()) / torch.numel(m.weight.detach().cpu()))
                # print(m.weight)


        # model.apply(log_sparsity)

        metrics, predictions = evaluate([lib.VAL, lib.TEST])
        # print(metrics["val"]["score"], metrics["test"]["score"])
        writer.add_scalar("val score by epoch", metrics["val"]["score"], stream.epoch)
        writer.add_scalar("test score by epoch", metrics["test"]["score"], stream.epoch)
        for k, v in metrics.items():
            training_log[k].append(v)
        progress.update(metrics[lib.VAL]['score'])

        writer.flush()

        if progress.success:
            print('New best epoch!')
            stats['best_epoch'] = stream.epoch
            stats['metrics'] = metrics
            save_checkpoint(False)
            for k, v in predictions.items():
                np.save(output / f'p_{k}.npy', v)

        elif progress.fail:
            break

    # %%
    print('\nRunning the final evaluation...')
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    stats['metrics'], predictions = evaluate(lib.PARTS)
    for k, v in predictions.items():
        np.save(output / f'p_{k}.npy', v)
    stats['time'] = lib.format_seconds(timer())
    save_checkpoint(True)
    print('Done!')
