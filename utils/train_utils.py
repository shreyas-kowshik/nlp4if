from transformers import AdamW
from utils.losses import *

# Define training loop here #
def train(model, dataloader, device, num_epochs, lr=1e-5, loss_type="classwise_sum"):
	optimizer = AdamW(model.parameters(),lr = lr)
	model.train()

	loss_fn = None
	if loss_type == "classwise_sum":
		loss_fn = classwise_sum
	else:
		print("Loss {} Not Defined".format(loss_type))
		sys.exit(1)

	for epoch in range(num_epochs):
	    for i,batch in enumerate(dataloader):
	        batch = [r.to(device) for r in batch]
	        sent_id, mask, labels = batch
	        model.zero_grad()
	        preds = model(sent_id, mask)
	        loss_val = loss_fn(preds, labels)
	        optimizer.zero_grad()
	        loss_val.backward()
	        optimizer.step()
	        
	    print("Epoch : {} : Loss : {}".format(epoch, loss_val.cpu()))