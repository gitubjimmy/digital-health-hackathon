import torch
import torchvision

from config import PATH
from data_prep_utils import get_data_loaders
from models import Net
from utils import file_output, img_show

classes = (
    'plane', 'car', 'bird',
    'cat', 'deer', 'dog',
    'frog', 'horse', 'ship', 'truck'
)

if __name__ == '__main__':
    _, test_loader = get_data_loaders()

    net = Net()
    net.load_state_dict(torch.load(PATH))

    # sample png
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # print images
    img_show(torchvision.utils.make_grid(images), 'sample.png')
    file_output(
        'GroundTruth: ' + ' '.join('%5s' % classes[labels[j]] for j in range(4)) + '\n'
    )

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        file_output("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
