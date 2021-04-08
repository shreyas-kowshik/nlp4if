# TEAM NAME
```

```

# Model Brief Description
```
The base of the architecture is a pretrained roberta-base model.
The final softmax layer of roberta is removed and we add our own task-specific heads on top.
The embeddings of roberta are passed thorugh a sequence of linear layers finally generating 7 different linear outputs, one for each question's task.
A multi-head attention layer is added on top, for each question's linear layer. This is with the intuition that before making a prediction for a particular task, the model will look at information gathered for the predictions of the other tasks, aggregate them, and using attention, filter out relevant information.
The number of attention-heads used is 3.
The roberta embeddings are not frozen i.e. they are also fine-tuned along with the custom task-specific heads we have added on the top.
The Multi-Head attention layers then individually are passed throguh a linear layer to give the task  predictions.
```