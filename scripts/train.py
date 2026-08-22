"""Temporary training entry point (M1).

Wires the OLD PaiNN model and post-processing (imported unchanged from the
original project) to the NEW data layer, to reproduce the baseline
test MAE ~= 5.4 meV. This loop is thrown away in M3 when the real Trainer
lands; scripts must never import concrete model classes after that.
"""

import argparse
import sys
from pathlib import Path

from lightning_fabric import seed_everything
import torch
import torch.nn.functional as F
from tqdm import trange

# The old project is a sibling of the repo root: __file__ -> scripts/ ->
# repo root -> Codes/ (three parents up). Anchoring to __file__ makes this
# work regardless of the current working directory.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "02456_painn_project-main"))

from potlab import ROOT
import potlab.config as config
import potlab.data.qm9
import potlab.training as training
from src.models import PaiNN, AtomwisePostProcessing  # type: ignore
from src.utils import EarlyStopping  # type: ignore


def build_parser():
    """Build the argument parser (separate from main so tests can reuse it)."""
    parser = argparse.ArgumentParser(description="Train a model.")
    # Default anchored to ROOT so the script works from any CWD; a
    # user-supplied --config path is used as given (relative to CWD).
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "default.yaml"), help="Path to the config file.")
    parser.add_argument("-o", "--override", action="append", default=[], help="Override config values using dotted paths, e.g., 'training.lr=1e-3'.")
    parser.add_argument("--resume", type=str, default=None, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--subset-size", type=int, default=None, help="Debug: cap the dataset to N molecules.")
    return parser


def compute_mae(painn, post_processing, dataloader, device):
    """Mean absolute error between predictions and targets (sum-then-divide)."""
    # 计算平均绝对误差（MAE）：在验证/测试集上评估模型精度。
    # 注意这里用 torch.no_grad()：评估阶段不需要记录梯度，省显存也更快。
    N = 0  # 累计统计的分子总数
    mae = 0  # 累计的绝对误差之和
    painn.eval()  # 切换到评估模式（关闭 dropout 等训练时特有的行为）
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)  # 把整个 batch 搬到 GPU/CPU 上

            # 模型输出的是"每个原子的贡献值"
            atomic_contributions = painn(
                atoms=batch.z,            # 每个原子的原子序数，形状 [原子总数]
                atom_positions=batch.pos, # 每个原子的三维坐标，形状 [原子总数, 3]
                graph_indexes=batch.batch,# 每个原子属于第几个分子，形状 [原子总数]
            )
            # 后处理：反标准化 + 加上原子参考值 + 按分子求和，得到分子级预测
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            # 累加这个 batch 的绝对误差（reduction='sum' 表示先求和再除总样本数，避免 batch 大小影响）
            mae += F.l1_loss(preds, batch.y, reduction='sum')
            N += len(batch.y)  # 累计分子个数（batch.y 每行是一个分子的标签）
        mae /= N  # 总绝对误差 / 分子总数 = 平均绝对误差

    return mae


def main():
    """Parse CLI args, load + override the config, train, evaluate."""
    parser = build_parser()
    args = parser.parse_args()
    config_data = config.load_config(args.config, args.override)
    run_dir = training.make_run_dir(config_data.run_name)
    print(f"Run directory: {run_dir}")
    print(f"Configuration: {config_data}")

    seed_everything(config_data.seed)  # 固定随机种子，保证结果可复现
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 有 GPU 就用 GPU，否则退回 CPU

    # 第二步：准备数据。QM9DataModule 封装了下载、切分、标准化等全部数据逻辑
    dm = potlab.data.qm9.QM9DataModule(
        target=config_data.data['target'],
        data_dir=config_data.data['data_dir'],
        batch_size_train=config_data.data['batch_size_train'],
        batch_size_eval=config_data.data['batch_size_eval'],
        num_workers=config_data.data['num_workers'],
        splits=config_data.data['splits'],
        seed=config_data.seed,
        subset_size=args.subset_size,  # None = full dataset; pass e.g. 1000 to debug
    )
    dm.prepare_data()  # 下载 QM9 数据（首次运行才会真正下载，之后自动跳过）
    dm.setup()  # 加载数据、打乱、切分成训练/验证/测试集

    train_loader = dm.train_dataloader()  # 训练集：batch 较小、随机打乱
    val_loader = dm.val_dataloader()      # 验证集：不打乱
    test_loader = dm.test_dataloader()    # 测试集：不打乱

    converter = dm.unit_conversion  # 显示用单位换算（eV → meV），property 已按 target 选好
    standardizer = dm.make_standardizer()  # fit 于训练集：得到 mean/std/atom_refs
    y_mean, y_std, atom_refs = standardizer.mean, standardizer.std, standardizer.atom_refs  # 训练集标签的均值/标准差 + 每种原子的参考值
    

    # 第三步：搭建模型。PaiNN 输出每个原子的贡献值，AtomwisePostProcessing 负责
    # 反标准化 + 加原子参考值 + 按分子求和，两者配合才能得到最终的分子性质预测
    painn = PaiNN(
        num_message_passing_layers=config_data.model['num_message_passing_layers'],
        num_features=config_data.model['num_features'],
        num_outputs=config_data.model['num_outputs'],
        num_rbf_features=config_data.model['num_rbf_features'],
        num_unique_atoms=config_data.model['num_unique_atoms'],
        cutoff_dist=config_data.model['cutoff_dist'],
    )
    post_processing = AtomwisePostProcessing(
        config_data.model['num_outputs'], y_mean, y_std, atom_refs  # 用训练集统计量初始化（注意：只能用训练集的统计量，防止信息泄漏）
    )

    painn.to(device)  # 把模型参数搬到 GPU 上
    post_processing.to(device)

    early_stopping = EarlyStopping(  # 早停器：验证集连续 patience 轮不改善就停止，并保存最优模型
        patience=config_data.training['early_stopping']['patience'],
        min_epochs=config_data.training['early_stopping']['min_epochs'],
    )

    optimizer = torch.optim.AdamW(  # AdamW 优化器：Adam + 解耦的权重衰减，训练 GNN 常用
        painn.parameters(),
        lr=config_data.training['lr'],
        weight_decay=config_data.training['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  # 余弦退火学习率调度：学习率随训练按余弦曲线衰减
        optimizer,
        T_max=config_data.training['num_epochs']*len(train_loader)  # 一个完整余弦周期的长度 = 总训练步数
    )

    # 第四步：训练循环。trange 会显示一个进度条，每轮结束更新 Train loss / Val. MAE
    pbar = trange(config_data.training['num_epochs'])
    for epoch in pbar:

        painn.train()  # 切换到训练模式
        loss_epoch_sum = 0.  # 本轮累计的训练损失（reduction='sum' 的原始和）
        for batch in train_loader:
            batch = batch.to(device)

            # --- 前向传播：分子 → 图 → 原子贡献 → 分子预测 ---
            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch
            )
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            loss_sum = F.mse_loss(preds, batch.y, reduction='sum')  # 均方误差（先求和，下面再归一化）

            # --- 反向传播 + 参数更新（PyTorch 标准三步） ---
            loss_mean = loss_sum / len(batch.y)  # 除以分子数 → 平均到每个分子的损失（batch 大小无关）
            optimizer.zero_grad()  # 清空上一轮残留的梯度
            loss_mean.backward()        # 反向传播：计算每个参数的梯度
            optimizer.step()       # 按梯度更新参数
            scheduler.step()       # 学习率调度器随步数更新

            loss_epoch_sum += loss_sum.detach().item()  # 累加本轮损失；detach() 切断计算图避免占用显存

        loss_epoch_mean = loss_epoch_sum / len(dm.data_train)  # 训练损失除以训练集分子总数，得到"每个分子的平均 MSE"
        val_mae = compute_mae(painn, post_processing, val_loader, device)  # 在验证集上评估 MAE

        pbar.set_postfix_str(  # 在进度条右侧实时显示本轮指标
            f'Train loss: {loss_epoch_mean:.3e}, '
            f'Val. MAE: {converter(val_mae):.3f}'  # unit_conversion 把 eV 换算成 meV 便于阅读
        )

        stop = early_stopping.check(painn, val_mae, epoch)  # 检查是否需要早停（并记录最优模型）
        if stop:
            print(f'Early stopping after epoch {epoch}.')
            break

    # 第五步：用训练过程中验证集表现最好的模型做最终评估
    painn = (
        early_stopping.best_model if early_stopping.best_model is not None
        else painn  # 如果从未触发早停记录（或验证集从未改善），就沿用当前模型
    )
    print(f'Best epoch: {early_stopping.best_epoch}')
    print(f'Best val. MAE: {early_stopping.best_loss}')

    test_mae = compute_mae(painn, post_processing, test_loader, device)  # 第六步：测试集评估
    print(f'Test MAE: {converter(test_mae):.3f}')

    return 0


if __name__ == "__main__":
    # True only when executed directly (python scripts/train.py), not when
    # imported. SystemExit carries main's return code out as the exit code.
    raise SystemExit(main())
