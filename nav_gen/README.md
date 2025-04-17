# NavGen

### Introduction

The NavGen pipeline consists of four steps: generating the LH-VLN task config, recording trajectories, decomposing trajectories, and generating step-by-step tasks. These correspond to `gen_task`, `gen_traj`, `split_traj`, and `gen_step_task` in `main.py`, respectively.

We use GPT as the open-source large model, so please prepare your OpenAI API key in advance. You can replace the model used (default is GPT-4o, with an optional GPT-4o mini) in `task_gen.py` and `split_task.py`.

Although the entire NavGen pipeline is run at once in `main.py`, due to considerations regarding network connectivity and the stability of large model outputs, we recommend running the code step by step in sequence.

### Start

Before you start generating, I would like to introduce some important parameters.

- API_KEY: Your API key, used to call the GPT model. For the complete generation process of a task, it costs about 2000 tokens.
- loop: The number of cycles of the first step(generating LH-VLN task). This depends on how many tasks you want to generate.
- sample_region: Whether to sample rooms based on the Euler distance between rooms. Since the Euler distance cannot well represent the actual distance between rooms, it is set to false by default.
- sample_obj: Whether to randomly sample objects in the room. Since this can reduce the influence of large model preference selection, it is set to true by default.
- max_step: The maximum number of action steps in the second step(record the trajectory). Used to limit the trajectory that is too long. The default setting is 500.
- ram_logs: The log save path of step 3 and step 4. This can help you confirm the actual progress of the NavGen.

Now you can generate tasks with the following command:

```bash
cd nav_gen
python main.py --API_KEY <your_openai_api_key> --loop 100
```
