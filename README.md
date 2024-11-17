# ECoC
We study the problem of sequential recommendation based on reinforcement learning. More specifically, we propose an **E**fficient **Co**ntinuous **C**ontrol (ECoC) framework to facilitate unified action learning, under the framework of actor-critic.

## Access the dataset
Download the preprocessed datasets [Tmall](https://drive.google.com/file/d/1K8LO0qpvf1hxtvuryFPR1U7gyAD2Pv0p/view?usp=sharing) and [Yelp](https://drive.google.com/file/d/1-BYAuLj3APJrL7fquMxDxkzLsDsoobXs/view?usp=sharing).

## How to run our code 
1. unzip the data, put the data folder in the `data/` directory
2. run the corresponding script within `runs/`

## Main Packages

```         
torch           
numpy  
pandas
tensorboard
tqdm
```

## Citation
If you use our code, please cite the paper
```
@article{wang2024efficient,
  title={An Efficient Continuous Control Perspective for Reinforcement-Learning-based Sequential Recommendation},
  author={Wang, Jun and Wu, Likang and Liu, Qi and Yang, Yu},
  journal={arXiv preprint arXiv:2408.08047},
  year={2024}
}
```