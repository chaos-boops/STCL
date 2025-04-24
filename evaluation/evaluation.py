# -*- coding = utf-8 -*-
# @Time:2023/4/7 15:55
# @Author : Zhang Tong
# @File:evaluateNet.py
# @Software:PyCharm

import torch
from TeacherModel import Stegano_Network, DataLoader
import numpy as np
from MS_SSIM import SSIM, MSSSIM, PSNR
import os
import xlsxwriter
from torchvision.utils import save_image
import argparse
msssim = MSSSIM()
ssim = SSIM()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation with configurable thresholds')
    parser.add_argument('--ssim_high', type=float, default=0.98, 
                        help='High threshold for SSIM (default: 0.98)')
    parser.add_argument('--ssim_low', type=float, default=0.85, 
                        help='Low threshold for SSIM (default: 0.85)')
    parser.add_argument('--msssim_high', type=float, default=0.9, 
                        help='High threshold for MSSSIM (default: 0.9)')
    parser.add_argument('--msssim_low', type=float, default=0.85, 
                        help='Low threshold for MSSSIM (default: 0.85)')
    parser.add_argument('--psnr_threshold', type=float, default=35, 
                        help='Threshold for PSNR (default: 35)')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--output_dir', type=str, default='./model', 
                        help='Base directory for output files (default: ./model)')
    return parser.parse_args()


def load_model(save_name, optimizer, model):
    model_data = torch.load(save_name)
    model.load_state_dict(model_data['model_dict'])
    optimizer.load_state_dict(model_data['optimizer_dict'])
    print("Model loaded successfully from", save_name)

def setup_directories(base_dir):
    dirs = [
        f"{base_dir}/easy1/1",
        f"{base_dir}/em2/1",
        f"{base_dir}/medium/1",
        f"{base_dir}/hard/1"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs

if __name__ == '__main__':

    args = parse_args()
    GPU = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('npu:0')
    # torch.npu.set_device(device)
    print(device)
    output_dirs = setup_directories(args.output_dir)


    batch_size = 1
    Teacher_05 = Stegano_Network()
    Teacher_15 = Stegano_Network()
    Teacher_40 = Stegano_Network()
    Teacher_80 = Stegano_Network()

    opt = torch.optim.Adam(Teacher_05.parameters())


    checkpoint_encoder_05 = torch.load(f'{args.output_dir}/teacher_05.pth')
    checkpoint_encoder_15 = torch.load(f'{args.output_dir}/teacher_15.pth')
    checkpoint_encoder_40 = torch.load(f'{args.output_dir}/teacher_40.pth')
    checkpoint_encoder_80 = torch.load(f'{args.output_dir}/teacher_80.pth')
    #checkpoint_decoder = torch.load('./dncoder5.pth')
    #checkpoint_discriminator = torch.load('./discriminator5.pth')
    Teacher_05.load_state_dict(checkpoint_encoder_05['model_dict'])
    Teacher_15.load_state_dict(checkpoint_encoder_15['model_dict'])
    Teacher_40.load_state_dict(checkpoint_encoder_40['model_dict'])
    Teacher_80.load_state_dict(checkpoint_encoder_80['model_dict'])
    # decode_network.load_state_dict(checkpoint_decoder['model_dict'])
    # discriminatorer.load_state_dict(checkpoint_discriminator['model_dict'])

    # Teacher_00.to(device)
    Teacher_05.to(device)
    Teacher_15.to(device)
    Teacher_40.to(device)
    Teacher_80.to(device)

    train = DataLoader("~/train", limit=np.inf, shuffle=False, batch_size=batch_size)
    workbook = xlsxwriter.Workbook(f'{args.output_dir}/evaluation_results.xlsx', {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet('Difficulty score')

    
    worksheet.write(0, 1, "epochs=05")
    worksheet.write(0, 2, "epochs=15")
    worksheet.write(0, 3, "epochs=40")
    worksheet.write(0, 4, "epochs=80")
    worksheet.write(0, 5, "SSIM")
    worksheet.write(0, 6, "epochs=05")
    worksheet.write(0, 7, "epochs=15")
    worksheet.write(0, 8, "epochs=40")
    worksheet.write(0, 9, "epochs=80")
    worksheet.write(0, 10, "PSNR")
    worksheet.write(0, 11, "epochs=05")
    worksheet.write(0, 12, "epochs=15")
    worksheet.write(0, 13, "epochs=40")
    worksheet.write(0, 14, "epochs=80")
    worksheet.write(0, 15, "MSSSIM")

    j = 1
    difficult_sample = 0
    medium_sample = 0
    easy_sample = 0

    ssim_score = []
    msssim_score = []
    psnr_score = []

    stego_image_1 = []
    stego_image_2 = []
    stego_image_3 = []
    stego_image_4 = []
    revealed_message_1 = []
    revealed_message_2 = []
    revealed_message_3 = []
    revealed_message_4 = []

    for batch_i, (inputs, _) in enumerate(train):
        size = inputs.size()
        ssim_score = np.zeros((len(inputs), 4))
        msssim_score = np.zeros((len(inputs), 4))
        psnr_score = np.zeros((len(inputs), 4))

        carrier = torch.empty(int(batch_size), size[1], size[2], size[3])
        secret = torch.empty(int(batch_size), 1, size[2], size[3])
        concatenated_input = torch.empty(int(batch_size), size[1] + 1, size[2], size[3])

        for i in range(len(inputs)):
            print("for sample %d:" % j)
            carrier[i] = inputs[i]
            secret[i][0] = torch.zeros(size[2], size[3]).random_(0, 2)
            # secret[i][1] = torch.zeros(size[2], size[3]).random_(0, 2)
            # secret[i][2] = torch.zeros(size[2], size[3]).random_(0, 2)
            # secret[i][3] = torch.zeros(size[2], size[3]).random_(0, 2)

            concatenated_input[i] = torch.cat((carrier[i], secret[i]), 0)

            concatenated_input = concatenated_input.to(device)
            carrier = carrier.to(device)
            secret = secret.to(device)

            # stego_image_1, revealed_message_1 = Teacher_00.forward(concatenated_input)
            stego_image_1, revealed_message_1 = Teacher_05.forward(concatenated_input, carrier, secret)
            stego_image_2, revealed_message_2 = Teacher_15.forward(concatenated_input, carrier, secret)
            stego_image_3, revealed_message_3 = Teacher_40.forward(concatenated_input, carrier, secret)
            stego_image_4, revealed_message_4 = Teacher_80.forward(concatenated_input, carrier, secret)

            ssim_score[i][0] = ssim(stego_image_1, carrier)
            print(f" epochs=05: SSIM : {ssim_score[i][0]}  ")
            worksheet.write(j, 1, ssim_score[i][0])

            ssim_score[i][1] = ssim(stego_image_2, carrier)
            print(f" epochs=15: SSIM : {ssim_score[i][1]}  ")
            worksheet.write(j, 2, ssim_score[i][1])

            ssim_score[i][2] = ssim(stego_image_3, carrier)
            print(f" epochs=40: SSIM : {ssim_score[i][2]}  ")
            worksheet.write(j, 3, ssim_score[i][2])

            ssim_score[i][3] = ssim(stego_image_4, carrier)
            print(f" epochs=80: SSIM : {ssim_score[i][3]}  ")
            worksheet.write(j, 4, ssim_score[i][3])

            # Calculate PSNR scores
            psnr_score[i][0] = PSNR(stego_image_1, carrier)
            print(f" epochs=05: PSNR : {psnr_score[i][0]}  ")
            worksheet.write(j, 6, psnr_score[i][0])

            psnr_score[i][1] = PSNR(stego_image_2, carrier)
            print(f" epochs=15: PSNR : {psnr_score[i][1]}  ")
            worksheet.write(j, 7, psnr_score[i][1])

            psnr_score[i][2] = PSNR(stego_image_3, carrier)
            print(f" epochs=40: PSNR : {psnr_score[i][2]}  ")
            worksheet.write(j, 8, psnr_score[i][2])

            psnr_score[i][3] = PSNR(stego_image_4, carrier)
            print(f" epochs=80: PSNR : {psnr_score[i][3]}  ")
            worksheet.write(j, 9, psnr_score[i][3])

            # Calculate MSSSIM scores
            msssim_score[i][0] = msssim(stego_image_1, carrier)
            print(f" epochs=05: MSSSIM : {msssim_score[i][0]}  ")
            worksheet.write(j, 11, msssim_score[i][0])

            msssim_score[i][1] = msssim(stego_image_2, carrier)
            print(f" epochs=15: MSSSIM : {msssim_score[i][1]}  ")
            worksheet.write(j, 12, msssim_score[i][1])

            msssim_score[i][2] = msssim(stego_image_3, carrier)
            print(f" epochs=40: MSSSIM : {msssim_score[i][2]}  ")
            worksheet.write(j, 13, msssim_score[i][2])

            msssim_score[i][3] = msssim(stego_image_4, carrier)
            print(f" epochs=80: MSSSIM : {msssim_score[i][3]}  ")
            worksheet.write(j, 14, msssim_score[i][3])

            # Classify samples based on thresholds from args
            if ((ssim_score[i][0] >= args.ssim_high) and 
                (ssim_score[i][1] >= args.ssim_high) and 
                (ssim_score[i][2] >= args.ssim_high) and 
                (ssim_score[i][3] >= args.ssim_high) and
                (msssim_score[i][0] >= args.msssim_high) and 
                (msssim_score[i][1] >= args.msssim_high) and 
                (msssim_score[i][2] >= args.msssim_high) and
                (msssim_score[i][3] >= args.msssim_high) and
                (psnr_score[i][3] >= args.psnr_threshold)):

                easy_sample += 1
                save_path = f'{args.output_dir}/easy/'
                filename = save_path + str(easy_sample) + 'e' + '.jpg'
                save_image(carrier, filename)

                save_path2 = f'{args.output_dir}/em/'
                filename2 = save_path2 + str(easy_sample) + 'e' + '.jpg'
                save_image(carrier, filename2)

                print(f"Sample {j} is classified as easy")

            elif ((ssim_score[i][0] <= args.ssim_low) or 
                  (ssim_score[i][1] <= args.ssim_low) or 
                  (ssim_score[i][2] <= args.ssim_low) or
                  (ssim_score[i][3] <= args.ssim_low) or
                  (msssim_score[i][0] <= args.msssim_low) or
                  (msssim_score[i][1] <= args.msssim_low) or
                  (msssim_score[i][2] <= args.msssim_low) or
                  (msssim_score[i][3] <= args.msssim_low)):

                difficult_sample += 1
                save_path = f'{args.output_dir}/hard/'
                filename = save_path + str(difficult_sample) + 'h' + '.jpg'
                save_image(carrier, filename)
                print(f"Sample {j} is classified as hard")

            else:
                medium_sample += 1
                save_path = f'{args.output_dir}/medium/'
                filename = save_path + str(medium_sample) + 'm' + '.jpg'
                save_image(carrier, filename)
                
                save_path2 = f'{args.output_dir}/em/'
                filename2 = save_path2 + str(medium_sample) + 'm' + '.jpg'
                save_image(carrier, filename2)
                
                print(f"Sample {j} is classified as medium")

            j = j + 1
            print("-----------------------------next----------------------------------")

    print("---------------evaluate over---------------")

    print("easy sample number: %d" % easy_sample)
    print("medium sample number: %d" % medium_sample)
    print("difficult sample number: %d" % difficult_sample)
    workbook.close()

    