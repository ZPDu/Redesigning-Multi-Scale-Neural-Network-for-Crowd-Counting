# Redesigning Multi-Scale Neural Network for Crowd Counting

This is the official implementation of 'Redesigning Multi-Scale Neural Network for Crowd Counting'. (IEEE Transactions on Image Processing) [[arXiv](https://arxiv.org/abs/2208.02894)]

![overview](./exp/overview.png)



## Installation

Python ≥ 3.6.

To install other required packages, run:

``` 
pip install -r requirements.txt
```



## Training

* Fill in the settings in datasets/SHHA/setting.py
* Run 'train.py'



## Evaluation

- Download the processed dataset ShanghaiTech part A([Link](https://drive.google.com/file/d/1QNLhNiUry77a6uY6dp5hLOs_bUQsU3Cd/view?usp=sharing)). For the preprocessing code, please refer to [this](https://github.com/TencentYoutuResearch/CrowdCounting-SASNet/blob/main/prepare_dataset.py).
- Download our SHA models [here](https://drive.google.com/drive/folders/1uL5Nll3H_SEpWFdriCJ5oawA4WpkzHBf?usp=share_link).
- Modify the path to the processed dataset and pretrained model in 'setting.py' in datasets/SHHA
- Run 'test.py'



## Acknowledgements

Part of codes are borrowed from [C^3 Framework](https://github.com/gjy3035/C-3-Framework) and [SASNet](https://github.com/TencentYoutuResearch/CrowdCounting-SASNet). Thanks for their great work!



## Citation

If you find this work useful, please cite

``` citation
@article{du2022redesigning,
  title={Redesigning Multi-Scale Neural Network for Crowd Counting},
  author={Du, Zhipeng and Shi, Miaojing and Deng, Jiankang and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:2208.02894},
  year={2022}
}
```

or

```@article{du2023redesigning,
@article{du2023redesigning,
  title={Redesigning multi-scale neural network for crowd counting},
  author={Du, Zhipeng and Shi, Miaojing and Deng, Jiankang and Zafeiriou, Stefanos},
  journal={IEEE Transactions on Image Processing},
  year={2023},
  publisher={IEEE}
}
```
