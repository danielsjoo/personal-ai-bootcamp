# CIFAR 100

from torch.utils.tensorboard import SummaryWriter
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import sys
import os

# Hyperparameters
batch_size = 64
learning_rate=1e-4
epochs = 10

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
device = local_rank # Use local_rank as the device ID
torch.cuda.set_device(device)

if dist.get_rank() == 0:
    run_name = sys.argv[1]
    # NOTE: I reverted this path to the corrected one from our last discussion.
    tb_log_path = f"./tb_logs/{run_name}"
    w = SummaryWriter(tb_log_path)
class NeuralNetwork(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.conv_layer1 = nn.Conv2d(3,6,(3,3), padding='same')
        self.batch_norm1 = nn.BatchNorm2d(6)
        self.max_pool1 = nn.MaxPool2d((2,2))
        self.conv_layer2 = nn.Conv2d(6,12,(3,3), padding='same')
        self.batch_norm2 = nn.BatchNorm2d(12)
        self.conv_layer3 = nn.Conv2d(12,6,(3,3), padding='same')
        self.batch_norm3 = nn.BatchNorm2d(6)
        self.max_pool2 = nn.MaxPool2d((2,2))
        self.dense1 = nn.Linear(384, 256)
        self.dense2 = nn.Linear(256, 100)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x:torch.Tensor):
        x = self.conv_layer1(x) # 6x32x32
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.max_pool1(x) # 6x16x16
        x = self.conv_layer2(x) #12x16x16
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.conv_layer3(x) #16x16x16
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.max_pool2(x) #6x8x8
        x = torch.flatten(x,1) #384 features
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x) # pass logits into loss function
        return x

train_transforms = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.CIFAR100(
    root=('../data'),
    train=True,
    download=True,
    transform=train_transforms
)
test_dataset = datasets.CIFAR100(
    root=('../data'),
    train=False,
    download=True,
    transform=test_transforms
)

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=dist.get_rank())
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=dist.get_rank())

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    sampler = train_sampler,
    num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    sampler = test_sampler,
    num_workers=2
)

# actual model training

single_model = NeuralNetwork(dropout=0.5).to(device)
model = DDP(single_model, device_ids=[device])
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

if dist.get_rank() == 0:
    sample_images, _ = next(iter(train_loader))
    img_grid = torchvision.utils.make_grid(sample_images)
    w.add_image('cifar100_images', img_grid)
    # Use model.module and send the sample images to the correct device (device 0 for rank 0)
    w.add_graph(model.module, sample_images.to(device))

for epoch in range(epochs):
    # train
    model.train()
    total_train_loss = 0
    for images,labels in train_loader:
        # --- 4. SEND DATA TO THE CORRECT DEVICE ---
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    # test
    model.eval()
    correct = 0
    total = 0
    total_test_loss = 0
    for images,labels in test_loader:
        # --- 5. SEND DATA TO THE CORRECT DEVICE ---
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_test_loss += loss.item()
        predicted_classes = torch.argmax(logits, dim=1)
        comparison = (predicted_classes == labels)
        correct += comparison.sum().item()
        total += labels.size(0)
        
    # write to summary writer
    if dist.get_rank() == 0:
        w.add_scalar("Train Loss", total_train_loss, epoch)
        w.add_scalar("Test Loss", total_test_loss, epoch)
        w.add_scalar("Test Accuracy", correct/total, epoch)
        for name, param in model.named_parameters():
            w.add_histogram(f'Weights/{name}', param.data, epoch)
            if param.grad is not None:
                w.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
        
if dist.get_rank() == 0:
    print("training complete")
    MODEL_PATH = f"../models/{run_name}.pt"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.module.state_dict(), MODEL_PATH)
    w.close()
    
dist.destroy_process_group()