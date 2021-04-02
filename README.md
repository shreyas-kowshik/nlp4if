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
- Change loss: Loss Weighting (A.3.1) (We should start with this as it should be effective + easy to implement)
- Fully-Adaptive Feature Sharing/Grouping tasks (basically introducing a heirchial network structure after shared layers in our case) (A.2.5)
If we are able to identify where negative transfer is occuring, we will be able to group he tasks better. One way is logically determine this grouping, other is brute force. Also running the model with single tasks once might help us to estimate the negative transfer better
- Use soft parameter sharing + Cross-stich network (Might take some work/time to implement this)
- Gradient Modulation might help to counter negative transfer (3.4)

Other:
- Try Multihead attention instead of self-attention

## TODO
- MAP on val set in not logged for indiviual task
