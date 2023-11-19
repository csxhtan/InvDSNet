import torch
from InvDSNet import InvDDNet
from utils import load_model, set_requires_grad
import os
from PIL import Image
from torchvision import transforms
import cv2

if __name__ == '__main__':
    net = InvDDNet().cuda()
    set_requires_grad(net, False)
    last_epoch = load_model(net, 'save/replicate/0', epoch=None)
    count = 0
    input_path = 'your test data'
    img_names = os.listdir(input_path)
    print(img_names)
    for i in range(len(img_names)):
        name = 'your test data/' + img_names[i]
        input_img = cv2.imread(name, -1)
        input_img = transforms.ToTensor()(input_img).cuda()
        input_img = torch.flip(input_img, dims=[0])
        input_img = input_img * 2 - 1.0
        input_img = input_img.unsqueeze(0)
        with torch.no_grad():
            output_img, _, _, _, _, _ = net(input_img, input_img)
            output_img = torch.clamp(output_img[:, :3], min=-1, max=1)
            output_img = (output_img + 1.0) / 2
            output_img = torch.squeeze(output_img)
        output_img = transforms.ToPILImage()(output_img)
        output_img.save('result/' + img_names[i])
        print(f"Finishing {i + 1}/{len(img_names)}")
