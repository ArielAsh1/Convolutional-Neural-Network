# load packages
import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
from torch.utils import data


""" the architecture of the  Convolutional neural network:
    Conv layer (10 5x5 Kernels) -> Max Pooling (2x2 kernel) -> Relu -> Conv layer (20 5x5 Kernels) ->
    -> Max Pooling (2x2 kernel) -> Relu -> Hidden layer (320 units) -> Relu -> Hidden layer (50 units) ->
    -> Output layer (10 outputs). """


class ConvolutionalNet(nn.Module):
    def __init__(self):
        super(ConvolutionalNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x) # conv 1
        x = F.max_pool2d(x, 2) # max pooling 1
        x = F.relu(x) # relu
        x = self.conv2(x) # conv 2
        x = F.max_pool2d((x), 2) # max pooling 2
        x = F.relu(x) # relu
        x = x.view(-1, 320) # flatten input
        x = self.fc1(x) # hidden layer 1
        x = F.relu(x) # relu
        x = self.fc2(x) # hidden layer 2
        return F.log_softmax(x, dim=1) #output

cnn_model = ConvolutionalNet()
print(cnn_model)

# now train the model on the train set.

# set hyperparameters
cnn_nepochs = 3
cnn_learning_rate = 0.01

# train the conv model
cnn_model = ConvolutionalNet()
# create sgd optimizer
cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=cnn_learning_rate)
# create negative log likelihood loos
cnn_criterion = nn.NLLLoss()

train_losses, val_losses = train_model(cnn_model, cnn_optimizer, cnn_criterion,
                                       cnn_nepochs, train_loader, val_loader, is_image_input=False)

# evaluate on the validation set
print(f"Validation accuracy: {evaluate_model(cnn_model, val_loader, is_image_input=False)}")

# A convolutional neural network that achieves the best accuracy on the validation set:
# Prepocess
from torch.utils import data

# transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the data
mnist_data = datasets.MNIST(data_path, download=True, train=True, transform=transform)
mnist_dataloader = data.DataLoader(mnist_data, batch_size=64, shuffle=True)

# split train set into train and validation. train-set is 80%, test-set is 20%
train_size = int(0.8 * len(mnist_data))
test_size = len(mnist_data) - train_size
train_dataset, test_dataset = data.random_split(mnist_data, [train_size, test_size])

# create data loader for the trainset (batch_size=64, shuffle=True)
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# create data loader for the valset (batch_size=64, shuffle=False)
val_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# define our Hyperparameters
lr_list = [0.1, 0.01, 0.001]
#optimizer_list = ["Adam", "RMSprop", "SGD"]
optimizer_list = ["Adam", "SGD"]
epoch_list = [2,3,5]

# Find network and hyperparams that achieve best validation accuracy as possible
import itertools

# this dictionary will hold training params and their accuracy output (params:acc)
results = {}

# create all possible combinations of lr, optimizer and epoch (cartesian product)
cartesian_product = itertools.product(lr_list, optimizer_list, epoch_list)
tracker = 1  # to keep track which cartesian_product we run now
for combination in cartesian_product:
  # prints to help keep track
  print("run", tracker)
  tracker += 1
  print(combination)

  cnn_model = ConvolutionalNet()
  cnn_lr, optimizer, cnn_epochs = combination
  cnn_criterion = nn.NLLLoss()
  # use optimizer according to current combination
  if optimizer == "Adam":
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=cnn_lr)
  if optimizer == "RMSprop":
    cnn_optimizer = optim.RMSprop(cnn_model.parameters(), lr=cnn_lr)
  if optimizer == "SGD":
    cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=cnn_lr)
  # train the model according to current combination
  train_losses, val_losses = train_model(cnn_model, cnn_optimizer, cnn_criterion,
                                       cnn_epochs, train_loader, val_loader, is_image_input=False)
  # current accuracy
  accuracy = evaluate_model(cnn_model, val_loader, is_image_input=False)
  print("acc:", accuracy)
  # create key with current params
  curr_key = (cnn_model, cnn_lr, optimizer, cnn_epochs)
  # add it to our dict with respective accuracy
  results[curr_key] = accuracy

# save the best model in this variable
max_accuracy = max(results.values())
print("best accuracy:", max_accuracy.numpy())

# max_model is a tuple (the CNN model, lr, optimizer, epochs)
max_model = max(results, key=results.get)
best_model = max_model[0]
print("best learning rate:",max_model[1])
print("best optimizer:",max_model[2])
print("best epoches number:",max_model[3])


# function to save predicted data
def predict_and_save(model, test_path, file_name):
  # load mnist test data
  mnist_test_data = torch.utils.data.TensorDataset(torch.load(test_path))
  # create a dataloader
  mnist_test_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=32, shuffle=False)
  # make a prediction for each batch and save all predictions in total_preds
  total_preds = torch.empty(0, dtype=torch.long)
  for imgs in mnist_test_loader:
    log_ps = model(imgs[0])
    ps = torch.exp(log_ps)
    _, top_class = ps.topk(1, dim=1)
    total_preds = torch.cat((total_preds, top_class.reshape(-1)))
  total_preds = total_preds.cpu().numpy()
  # write all predictions to a file
  with open(file_name,"w") as pred_f:
    for pred in total_preds:
      pred_f.write(str(pred) + "\n")
