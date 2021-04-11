import argparse
import torch
from torchvision import transforms, datasets
# from model import DenoiseAutoEncoderModel
from cls_model import Net


def print_table(results):
    print('Classification accuracy on noisy and denoised images \nusing different noise variation\n')
    print("{:<10} {:<10} {:<10} \n".format('Noise var', 'Noisy img', 'Denoised img'))
    for var, (noisy_score, denoised_score) in results:
        print("{:10.2f} {:10.2f} {:10.2f}".format(var, noisy_score, denoised_score))


def add_noise(tensor, mean=0, var=1.):
    return tensor + torch.randn(tensor.size()) * var + mean


def test_classification(cls_model, denoise_model, data_loader, device, noise_var=None):
    cls_model.eval()
    denoise_model.eval()
    noisy_correct = 0
    denoised_correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            noisy_data = add_noise(data, mean=0, var=noise_var).to(device)
            denoised_data = denoise_model(noisy_data)
            target = target.to(device)

            noisy_output = cls_model(noisy_data)
            denoised_output = cls_model(denoised_data)

            noisy_pred = noisy_output.data.max(1, keepdim=True)[1]
            denoised_pred = denoised_output.data.max(1, keepdim=True)[1]

            noisy_correct += noisy_pred.eq(target.data.view_as(noisy_pred)).sum()
            denoised_correct += denoised_pred.eq(target.data.view_as(denoised_pred)).sum()

    noisy_accuracy = 100. * noisy_correct / len(data_loader.dataset)
    denoised_accuracy = 100. * denoised_correct / len(data_loader.dataset)
    return noisy_accuracy.item(), denoised_accuracy.item()


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
    testset = datasets.MNIST(root='./data', train=False,
                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    cls_model = Net()
    cls_model.load_state_dict(torch.load(args.classification_model_path, map_location=device))
    # cls_model = torch.load(args.classification_model_path, map_location=device)
    cls_model.to(device)

    # denoise_model = DenoiseAutoEncoderModel()
    # denoise_model.load_state_dict(torch.load(args.denoising_model_path, map_location=device))
    denoise_model = torch.load(args.denoising_model_path, map_location=device)
    denoise_model.to(device)

    noise_vars = torch.linspace(0, 1.5, 7)
    accurasies = [[var, test_classification(cls_model, denoise_model, testloader, device, noise_var=var)]
                  for var in noise_vars]
    print_table(accurasies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classification_model_path', '-c',
                        help='Path to classification model', default='./cls_model_sd.pth')
    parser.add_argument('--denoising_model_path', '-d',
                        help='Path to denoising model', default='./net_model.pth')
    args = parser.parse_args()
    main(args)