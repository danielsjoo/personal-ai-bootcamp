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

# Hyperparameters
batch_size = 64
learning_rate=1e-4
epochs = 10

device = "cuda"
dist.init_process_group("nccl")
rank = dist.get_rank()
world_size= dist.get_world_size()
torch.cuda.set_device(rank)

if rank == 0:
    run_name = sys.argv[1]
    tb_log_path = f"../runs/{run_name}"
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

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

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

single_model = NeuralNetwork(dropout=0.5).to(rank)
model = DDP(single_model, device_ids=[rank])
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

if rank == 0:
    # Only get sample images and write to TensorBoard on the main process
    sample_images, _ = next(iter(train_loader))
    img_grid = torchvision.utils.make_grid(sample_images)
    w.add_image('cifar100_images', img_grid)
    w.add_graph(model, sample_images.to(rank))

for epoch in range(epochs):
    # train
    model.train()
    total_train_loss = 0
    for images,labels in train_loader:
        images = images.to(rank)
        labels = labels.to(rank)
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
        images = images.to(rank)
        labels = labels.to(rank)
        logits = model(images)
        loss = loss_fn(logits, labels)
        total_test_loss += loss.item()
        predicted_classes = torch.argmax(logits, dim=1)
        comparison = (predicted_classes == labels)
        correct += comparison.sum().item()
        total += labels.size(0)
        
    # write to summary writer
    if rank == 0:
        w.add_scalar("Train Loss", total_train_loss, epoch)
        w.add_scalar("Test Loss", total_test_loss, epoch)
        w.add_scalar("Test Accuracy", correct/total, epoch)
        for name, param in model.named_parameters():
            w.add_histogram(f'Weights/{name}', param.data, epoch)
            if param.grad is not None:
                w.add_histogram(f'Gradients/{name}', param.grad.data, epoch)
        
if rank == 0:
    print("training complete")
    MODEL_PATH = f"../models/{run_name}.pt"
    torch.save(model.module.state_dict(), MODEL_PATH)
    w.close()
    
dist.destroy_process_group()