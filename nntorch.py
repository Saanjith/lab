import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv(r"C:\Users\sreea\Downloads\Bank_Personal_Loan_Modelling.csv")
df = df.drop(['ZIP Code', 'ID'], axis=1)
df.columns = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard', 'Personal Loan']
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Scale features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Define neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(11, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

net = Net()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())

# Train network
for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(zip(x_train, y_train), 0):
        inputs, labels = data
        inputs = torch.from_numpy(inputs).float()
        labels = torch.tensor([labels]).float()

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if epoch % 10 == 9:
        print(f'Epoch {epoch + 1}, loss: {running_loss / len(x_train)}')

# Test network
net.eval()
with torch.no_grad():
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.tensor(y_test).float()

    outputs = net(x_test)
    predicted = torch.round(outputs)
    correct = (predicted == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct / total

    print(f'Test accuracy: {accuracy}')
