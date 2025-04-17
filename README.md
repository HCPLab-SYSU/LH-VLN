<div align="center" style="font-family: charter;">
<h1>Towards Long-Horizon Vision-Language Navigation:</br> Platform, Benchmark and Method (CVPR-25)</h1>
<img src="static/images/1-intro.png" width="90%"/>
<br />

<a href="https://arxiv.org/abs/2412.09082" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-LH--VLN-red?logo=arxiv" height="20" />
</a>
<a href="https://hcplab-sysu.github.io/LH-VLN/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-LH--VLN-blue.svg" height="20" />
</a>
<a href="https://github.com/HCPLab-SYSU/LH-VLN" target="_blank" style="display: inline-block; margin-right: 10px;">
    <img alt="GitHub Code" src="https://img.shields.io/badge/Code-LH--VLN-white?&logo=github&logoColor=white" />
</a>

<div>
    <a href="songxsh@mail2.sysu.edu.cn" target="_blank">Xinshuai Song</a><sup>1*</sup>,</span>
    <a href="chenwx228@mail2.sysu.edu.cn" target="_blank">Weixing Chen</a><sup>1*</sup>, </span>
    <a href="liuy856@mail.sysu.edu.cn" target="_blank">Yang Liu</a><sup>1,3</sup>,</span>
    <a href="chenwk891@gmail.com" target="_blank">Weikai Chen</a><sup> </sup>,</span>
    <a href="liguanbin@mail.sysu.edu.cn" target="_blank">Guanbin Li</a><sup>1,2,3</sup>,</span>
    <a href="linliang@ieee.org" target="_blank">Liang Lin</a><sup>1,2,3</sup>,</span>
</div>

<div>
    <sup>1</sup>Sun Yat-sen University&emsp;
    <sup>2</sup>Peng Cheng Laboratory&emsp;
    <sup>3</sup>Guangdong Key Laboratory of Big Data Analysis and Processing&emsp;
</div>
<br />
<p align="justify"><i>Existing Vision-Language Navigation (VLN) methods primarily focus on single-stage navigation, limiting their effectiveness in multi-stage and long-horizon tasks within complex and dynamic environments. To address these limitations, we propose a novel VLN task, named Long-Horizon Vision-Language Navigation (LH-VLN), which emphasizes long-term planning and decision consistency across consecutive subtasks. Furthermore, to support LH-VLN, we develop an automated data generation platform NavGen, which constructs datasets with complex task structures and improves data utility through a bidirectional, multi-granularity generation approach. To accurately evaluate complex tasks, we construct the Long-Horizon Planning and Reasoning in VLN (LHPR-VLN) benchmark consisting of 3,260 tasks with an average of 150 task steps, serving
as the first dataset specifically designed for the long-horizon vision-language navigation task. Furthermore, we propose Independent Success Rate (ISR), Conditional Success Rate (CSR), and CSR weight by Ground Truth (CGT) metrics, to provide fine-grained assessments of task completion. To improve model adaptability in complex tasks, we propose a novel Multi-Granularity Dynamic Memory (MGDM) module that integrates short-term memory blurring with long-term memory retrieval to enable flexible navigation in dynamic environments. Our platform, benchmark and method supply LH-VLN with a robust data generation pipeline, comprehensive model evaluation dataset, reasonable metrics, and a novel VLN model, establishing a foundational framework for advancing LH-VLN. </i></p>
</div>

## Preparation

### Environment

This project is developed with Python 3.9. You can use [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://anaconda.org/) to create the environment:

```bash
conda create -n lhvln python=3.9
conda activate lhvln
```

We uses [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/main) as simulator, which can be [built from source](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md#build-from-source) or installed from conda:

```bash
conda install habitat-sim headless -c conda-forge -c aihabitat
```

Then you can install the environment required for the project:

```bash
git clone https://github.com/HCPLab-SYSU/LH-VLN.git
cd LH-VLN
pip install -r requirements.txt
```

### Data

We use [HM3D](https://aihabitat.org/datasets/hm3d/) as the scene dataset. You can download the splits required by following the command below. Note that you need to submit an application to [Matterport](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data) before using it. For more information, please refer to [this link](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d).

```bash
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_train_v0.2
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_val_v0.2
```

In NavGen, we use the pre-trained model of [RAM](https://github.com/xinyu1205/recognize-anything). You can download the model [here](https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth).

Your final directory structure should be like this:

```
LH-VLN
├── data
│   ├── hm3d
│   │   ├── train
│   │   ├── val
│   │   ├── hm3d_annotated_basis.scene_dataset_config.json
│   └── models
│   │   └── ram_plus_swin_large_14m.pth
```

## LHPR-VLN Dataset

Our dataset is available in [huggingface](https://huggingface.co/datasets/Starry123/LHPR-VLN). Thanks a lot for your patience!

## NavGen Pipeline

After completing the preparations, you can now refer to the [guide](https://github.com/sxshco/LH-VLN/tree/master/nav_gen#readme) to generate your LH-VLN task!

## Timeline

- [ ] 2025.5: Full benchmark

## Acknowledgement

We used [RAM](https://github.com/xinyu1205/recognize-anything)'s source code in `nav_gen/recognize_anything`. Thanks for their contribution!!

## Citation

If you find our paper and code useful in your research, please consider giving us a star :star: and citing our work :pencil: :)
```
@inproceedings{song2024towards,
  title={Towards long-horizon vision-language navigation: Platform, benchmark and method},
  author={Song, Xinshuai and Chen, Weixing and Liu, Yang and Chen, Weikai and Li, Guanbin and Lin, Liang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
