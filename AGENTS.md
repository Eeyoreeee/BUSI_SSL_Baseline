# AGENTS.md

## Repository expectations

- This repository is for BUSI breast ultrasound semi-supervised segmentation.
- Keep the codebase small and readable.
- Do not introduce PyTorch Lightning, Hydra, MONAI, registry systems, or hook platforms.
- Preserve the current clean-baseline structure.
- Use standard ImageNet-pretrained ResNet50 encoder wording; do not claim official torchvision equivalence.
- Before changing training logic, keep supervised and SSL paths comparable.
- Prefer minimal diffs and avoid large refactors.

## Response style for coding tasks

- Do not paste full file contents unless I explicitly ask for them.
- Edit files directly in the repository and keep the response concise.
- After coding, return only:
  - which files were changed
  - what was changed at a high level
  - any important design decision
  - exact commands I should run next
  - any risk or follow-up note
- Prefer minimal diffs and avoid large refactors.