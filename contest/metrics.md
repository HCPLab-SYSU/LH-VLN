### Evaluation Metrics

To evaluate the model's performance in the LH-VLN task, in addition to standard evaluation metrics, there are task-specific metrics introduced for this task. The standard metrics include: Success Rate (SR), Oracle Success Rate (OSR), Success Rate weighted by path length (SPL), and Navigation Error (NE). Due to the high difficulty of long-range navigation tasks, standard metrics like SR are too coarse-grained. The newly introduced metrics offer more granular insights into the model's performance. The task-specific metrics are as follows:

**Independent Success Rate (ISR)**: ISR is used to quantify the success rate of each sub-task, providing insight into the independent completion of individual sub-tasks. The formula is:

$$
ISR = \sum_{j=0}^{M} \sum_{i=0}^{N} \frac{s_{j,i}}{M \cdot N}
$$

where $M$ represents the number of tasks, and $N$ represents the number of sub-tasks in task $j$.

**Conditional Success Rate (CSR)**: CSR is used to evaluate the success rate of the entire complex task. Since the result of each sub-task affects subsequent tasks, it reflects the interdependence within the task sequence. The formula is:

$$
CSR = \sum_{j=0}^{M}\sum_{i=0}^{N} \frac{s_{j,i} \left(1 + (N - 1)s_{j,i-1}\right)}{M \cdot N^2}
$$

where $s_{j,i}$ represents the success or failure of the $i$-th sub-task in the $j$-th task.

**CGT (Conditioned CSR based on Ground Truth)**: CGT further corrects CSR by introducing ground truth weighting, accounting for biases in path difficulty. The formula is:

$$
CGT = \sum_{j=0}^{M}\sum_{i=0}^{N} \frac{P_i}{P} \cdot \frac{s_{j,i} \left(1 + (N - 1)s_{j,i-1}\right)}{M \cdot N}
$$

where $P_i$ is the weight of the $i$-th sub-task, and $P$ is the sum of all sub-task weights. Weights are sometimes based on the path length required to complete a task.

**Target Achievement Rate (TAR)**: TAR evaluates how close the agent is to the target when the navigation success rate is low, based on NE. The formula is:

$$
TAR = \sum_{j=0}^{M}\sum_{i=0}^{N} (1-\frac{\max(NE_{j,i}-D_s,0)}{\max(NE_{j,i},GT_{j,i})})
$$

where $NE_{j,i}$ and $GT_{j,i}$ represent the NE and ground truth distance for the $i$-th sub-task in the $j$-th task, and $D_s$ is the distance threshold for determining navigation success.

Final ranking: 0.4\*TAR + 0.2\*ISR + 0.2\*CSR + 0.2\*CGT.

