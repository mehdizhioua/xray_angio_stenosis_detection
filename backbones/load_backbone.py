import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.models.detection as detection

# Define the data transformations
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
transform_val = transforms.Compose([
    transforms.ToTensor()
])



# Load the COCO dataset
#train_dataset = datasets.CocoDetection(root='./coco/train2017', annFile='./coco/annotations/instances_train2017.json', transform=transform_train)
#val_dataset = datasets.CocoDetection(root='./coco/val2017', annFile='./coco/annotations/instances_val2017.json', transform=transform_val)

# Define the data loaders
#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
#val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# Define the Faster R-CNN model with a NASNet backbone
backbone = models.resnet152(pretrained=True)
backbone.out_features = 4032
model = detection.fasterrcnn_resnet50_fpn(backbone, num_classes=91)

# Define the loss function and optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Train the model for a few epochs
#num_epochs = 5
#for epoch in range(num_epochs):
#    model.train()
#    for images, targets in train_loader:
#        images = list(image for image in images)
#        targets = [{k: v for k, v in t.items()} for t in targets]
#        loss_dict = model(images, targets)
#        losses = sum(loss for loss in loss_dict.values())
#        optimizer.zero_grad()
#        losses.backward()
#        optimizer.step()
#        lr_scheduler.step()

#    model.eval()
#    with torch.no_grad():
#        for images, targets in val_loader:
#            images = list(image for image in images)
#            targets = [{k: v for k, v in t.items()} for t in targets]
#            outputs = model(images)
#            # Compute validation loss and metrics

# Save the model weights as a .pth file
torch.save(model.state_dict(), 'resnet/faster_rcnn_resnet50_coco.pth')