# NavGen

### Introduction

The NavGen pipeline consists of four parts: generating the LH-VLN task config, recording trajectories, decomposing trajectories, and generating step-by-step tasks. These correspond to `gen_task`, `gen_traj`, `split_traj`, and `gen_step_task` in `main.py`, respectively.

We use GPT as the open-source large model, so please prepare your OpenAI API key in advance. You can replace the model used (default is GPT-4o, with an optional GPT-4o mini) in `task_gen.py` and `split_task.py`.

Although the entire NavGen pipeline is run at once in `main.py`, due to considerations regarding network connectivity and the stability of large model outputs, we recommend running the code step by step in sequence.
