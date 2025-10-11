import math


def linear_warmup_decay_with_min_lr(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,  # 例如 0.1 表示最低 lr 为 base_lr 的 10%
):
    """
    生成给 LambdaLR 使用的 lr_lambda。

    - 0 → warmup_steps      线性升温：0 → 1
    - warmup_steps → total_steps  线性衰减：1 → min_lr_ratio

    Parameters
    ----------
    warmup_steps : int
        线性 warm-up 的步数（>0）。
    total_steps : int
        训练总步数（>warmup_steps）。
    base_lr : float
        初始学习率（optimizer 里设定的 lr，会被乘以此函数的输出）。
    min_lr_ratio : float
        衰减到的最小学习率与 base_lr 之比 (0 < min_lr_ratio ≤ 1)。

    Returns
    -------
    Callable[[int], float]
        传给 LambdaLR 的 lr_lambda。
    """
    assert 0 < min_lr_ratio <= 1, "`min_lr_ratio` 必须在 (0, 1] 范围内"
    assert total_steps > warmup_steps > 0, "`total_steps` 必须大于 `warmup_steps`"

    def lr_lambda(current_step: int) -> float:
        # warm-up
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        # decay
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return lr_lambda


def cosine_schedule_with_min_lr(
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,  # 例如 0.1 表示最低 lr 为 base_lr 的 10%
):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (1 - min_lr_ratio) * cosine + min_lr_ratio
    return lr_lambda