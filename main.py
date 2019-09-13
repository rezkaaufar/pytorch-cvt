import torch

from src.data_loader import Data
from src.cvt_model import CVTModel
from torch import optim
from src.evaluator import Evaluator
from src.artifacts_manager import ArtifactsManager
from warmup_scheduler import GradualWarmupScheduler
from tensorboardX import SummaryWriter

""" hyperparameter """
train_path = "labeled_data/ner/sentence-train.conll"
dev_path = "labeled_data/ner/sentence-dev.conll"
unlabeled_path = "unlabeled_data/informal/test.txt"
save_to = "models/"
semi_supervised = True  # whether to use auxiliary modules or not
resume_training = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

""" initialize data_loader, artifacts manager, and summary writer """
data = Data(train_path, unlabeled_path, semi_supervised, dev_path=dev_path, device=device)
data.initialize()
pad_token_id = data.get_pad_token_id()
num_words, num_chars, num_tags = data.get_input_sizes()
artifacts_manager = ArtifactsManager(save_to, device=device)
writer = SummaryWriter(log_dir='runs/cvt')

""" saving and loading models. the initialize command automatically creates the main encoder, primary prediction 
module, and the five auxiliary modules """
if not resume_training:
    cvt = CVTModel(num_words, num_tags, num_chars=num_chars, device=device)
    cvt.initialize()
    cvt.to(device)
    # artifacts_manager.save_model_and_artifacts(cvt)
else:
    cvt = artifacts_manager.load_model_and_artifacts()


""" optimizer """
optimizer = optim.Adam(cvt.parameters(), lr=1e-3, weight_decay=5e-3)
# optimizer = optim.SGD(cvt.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
# max_epoch = 1200
# scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_epoch)
# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=800, after_scheduler=scheduler_cosine)

""" evaluator """
evaluator = Evaluator()

""" training """
count = 0
running_loss = 0.0
for mb, mode in data.get_alternating_minibatch():
    cvt.train()
    optimizer.zero_grad()
    word_input = getattr(mb, "words")
    char_input = getattr(mb, "char")
    label = None
    if mode == "labeled":
        label = getattr(mb, "lab")
    loss = cvt.forward(word_input, mode, char_input=char_input, label=label)

    if mode == "labeled":
        running_loss += loss.item()
        writer.add_scalar('data/labeled_loss', loss.item(), count)
    elif mode == "unlabeled":
        writer.add_scalar('data/unlabeled_loss', loss.item(), count)

    loss.backward()
    optimizer.step()
    if count % 101 == 0 and count != 0:
        cvt.eval()
        precision, recall, f1 = evaluator.evaluate_on_data(cvt, data)
        writer.add_scalar('data/precision', precision, count)
        writer.add_scalar('data/recall', recall, count)
        writer.add_scalar('data/f1', f1, count)
        print("precision: {}, recall: {}, f1: {} at training step {}".format(precision, recall, f1, count))
        print("{} mode loss: {}".format(mode, loss))
    if count % 1000 == 0 and count != 0:
        artifacts_manager.save_model_and_artifacts(cvt)
    if count % 20284 == 0 and count != 0: # this equivalent to one epoch
        scheduler.step(running_loss)
        running_loss = 0
    count += 1

writer.close()


""" this part is for tweaking and validating """
# for i in range(len(data.lab.vocab)):
#     print(data.lab.vocab.itos[i])
# print(data.lab.vocab)
# for i, batch in enumerate(data.dev):
#     print(batch)