from torchvision import transforms


train_aug = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

