"""EMA teacher helpers for the minimal teacher-student SSL baseline."""

from __future__ import annotations

import torch
from torch import nn


def freeze_teacher(teacher: nn.Module) -> None:
    """Ensure teacher parameters never require gradients."""

    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)


def copy_student_to_teacher(student: nn.Module, teacher: nn.Module) -> None:
    """Initialize teacher weights from the current student weights."""

    teacher.load_state_dict(student.state_dict(), strict=True)
    freeze_teacher(teacher)


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    """Update teacher parameters and floating buffers from student via EMA."""

    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"EMA decay must be within [0, 1], got {decay}.")

    student_state = student.state_dict()
    teacher_state = teacher.state_dict()

    for key, teacher_tensor in teacher_state.items():
        student_tensor = student_state[key].detach()
        if teacher_tensor.is_floating_point():
            teacher_tensor.mul_(decay).add_(student_tensor, alpha=1.0 - decay)
        else:
            teacher_tensor.copy_(student_tensor)

    freeze_teacher(teacher)
