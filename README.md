<img width="2563" height="742" alt="Frame 261" src="https://github.com/user-attachments/assets/fc36b68d-80af-4c8d-ab09-1e0e44a2193e" />

# Do as I Do

[**Project Page**](https://do-as-i-do.com/) | [**arXiv**](https://arxiv.org/abs/2606.19333) 

Code release for Do as I Do.

Each part of our pipeline is contained in its own folder. External code references are provided as git submodules with our changes baked in.

- **`reconstruction/`** — object + hand reconstruction and 6-DoF pose tracking from a hand-object
  demo video (SAM3 → SAM3D mesh → MoGe pointmaps → HaWoR → TAPIR → guided diffusion for tracking → (optionally) projection).
  Full details in [`reconstruction/README.md`](reconstruction/README.md).
- **`retargeting/`** — retargets the reconstructed hand-object demo onto a robot hand
  (dataset processing → convex decomposition → MJCF scene generation → IK → sampling-based MPC
  in MuJoCo Warp). Consumes the reconstruction pipeline's output directly.
  Full details in [`retargeting/README.md`](retargeting/README.md).
- **`deployment/`** — replay a retargeted demo on the real robot: a barebones
  MuJoCo replay/IK pass turns a retargeting output into a dual-UR3e joint
  trajectory, which is then streamed to the UR3e arms + Sharpa Wave hands.
  Full details in [`deployment/README.md`](deployment/README.md).

