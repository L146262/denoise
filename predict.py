import argparse
import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from PIL import Image
from model import DenoiseAutoEncoderModel

def main(args):
    # model = DenoiseAutoEncoderModel()
    model = torch.load(args.model, map_location=torch.device('cpu'))
    # model.to(device)
    # model = torch.load(args.model, map_location=torch.device('cpu'))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    test_img = Image.open(args.input_file).convert('L')
    test_img = transform(test_img).unsqueeze(1).float()

    model.eval()
    with torch.no_grad():
        test_result = model(test_img)

    save_image(test_result, args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Input noisy image path')
    parser.add_argument('output_file', help='Output path')
    parser.add_argument('-m', '--model', help='Path to model', default='./c.pth', required=False)
    args = parser.parse_args()
    main(args)
