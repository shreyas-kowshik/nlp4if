# NLP4IF

Experiments Code.

Link to task : https://gitlab.com/NLP4IF/nlp4if-2021

### Wandb installation

```
pip install wandb

wandb login
```

Go over to the link in terminal and paste you API key. That should be it.

### Running Instructions

```
python bert_train.py --epochs 100 --wandb_run [run_name]
```

## Further steps
https://arxiv.org/pdf/2009.09796.pdf

Main points:
- Change loss: Loss Weighting (Sec 3.1) (We should start with this as it should be effective + easy to implement)
- Fully-Adaptive Feature Sharing/Grouping tasks (basically introducing a heirchial network structure after shared layers in our case) (Sec 3.1 2.5)
If we are able to identify where negative transfer is occuring, we will be able to group the tasks better. One way is logically determine this grouping, other is brute force. Also running the model with single tasks once might help us to estimate the negative transfer better
- Use soft parameter sharing + Cross-stich network (Might take some work/time to implement this)

Other:
- Try Multihead attention instead of self-attention

## TODO
- Add wandb config for ml_model_train.py [Will do later today]
- Convert ml_model_train.py feature generation and training to proper functions [Will do later today]
