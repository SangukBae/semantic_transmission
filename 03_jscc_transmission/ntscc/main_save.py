import os
import time
from datetime import datetime
import sys
import random
import argparse
from net.NTSCC_Hyperior import NTSCC_Hyperprior
import torch.optim as optim
from utils import *
from data.datasets import get_loader, get_test_loader
from config import config
from PIL import Image
import torchvision.transforms as transforms
from glob import glob
import pandas as pd


def train_one_epoch(epoch, net, train_loader, optimizer_G, aux_optimizer, device, logger):
    global global_step
    net.train()
    elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs = [AverageMeter() for _ in range(7)]
    metrics = [elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs]
    for batch_idx, input_image in enumerate(train_loader):
        optimizer_G.zero_grad()
        aux_optimizer.zero_grad()

        start_time = time.time()
        input_image = input_image.to(device)
        global_step += 1
        mse_loss_ntc, bpp_y, bpp_z, mse_loss_ntscc, cbr_y, x_hat_ntc, x_hat_ntscc = net(input_image)
        if config.use_side_info:
            cbr_z = bpp_snr_to_kdivn(bpp_z, 10)
            loss = mse_loss_ntscc + mse_loss_ntc + config.train_lambda * (bpp_y * config.eta + cbr_z)
            cbrs.update(cbr_y + cbr_z)
        else:
            # add ntc_loss to improve the training convergence stability
            ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
            loss = ntc_loss + mse_loss_ntscc
            cbrs.update(cbr_y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
        optimizer_G.step()

        aux_loss = net.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        elapsed.update(time.time() - start_time)
        losses.update(loss.item())
        bppys.update(bpp_y.item())
        bppzs.update(bpp_z.item())

        psnr_jscc = 10 * (torch.log(255. * 255. / mse_loss_ntscc) / np.log(10))
        psnr_jsccs.update(psnr_jscc.item())
        psnr = 10 * (torch.log(255. * 255. / mse_loss_ntc) / np.log(10))
        psnrs.update(psnr.item())

        if (global_step % config.print_step) == 0:
            process = (global_step % train_loader.__len__()) / (train_loader.__len__()) * 100.0
            log = (' | '.join([
                f'Step [{global_step % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR_JSCC {psnr_jsccs.val:.2f} ({psnr_jsccs.avg:.2f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'PSNR_NTC {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
                f'Epoch {epoch}',
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()


def test(net, test_path, logger, db, save=False):
    print("test_path:", test_path)
    with torch.no_grad():
        sampled_dir =  test_path + "_"+ db
        if not os.path.exists(sampled_dir):
            os.makedirs(sampled_dir, exist_ok=True)
        net.eval()
        elapsed, losses, psnrs, bppys, bppzs, psnr_jsccs, cbrs = [AverageMeter() for _ in range(7)]
        PSNR_list = []
        CBR_list = []
        max_key_frame_num = 1
        key_frame_nums = 0
        #遍历test_path目录下所有.png文件
        for img_path in glob(os.path.join(test_path, '*.png')):
            key_frame_nums += 1
            num = int(os.path.basename(img_path).split('.')[0])
            max_key_frame_num = max(max_key_frame_num, num)
            ori_frame = Image.open(img_path).convert('RGB')
            # ori_frame = ori_frame.resize((256, 256))
            ori_frame = transforms.ToTensor()(ori_frame)
            ori_frame = ori_frame.unsqueeze(0)
            input_image = ori_frame.cuda()
            start_time = time.time()
            mse_loss_ntc, bpp_y, bpp_z, mse_loss_ntscc, cbr_y, x_hat_ntc, x_hat_ntscc = net(input_image)
            if config.use_side_info:
                cbr_z = bpp_snr_to_kdivn(bpp_z, 10)
                ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
                ntscc_loss = mse_loss_ntscc + bpp_y * config.eta + cbr_z
                loss = ntc_loss + ntscc_loss
                cbrs.update(cbr_y + cbr_z)
            else:
                ntc_loss = mse_loss_ntc + config.train_lambda * (bpp_y + bpp_z)
                loss = ntc_loss + mse_loss_ntscc
                cbrs.update(cbr_y)
            losses.update(loss.item())
            bppys.update(bpp_y)
            bppzs.update(bpp_z)
            elapsed.update(time.time() - start_time)

            # psnr_jscc = CalcuPSNR_int(input_image, x_hat_ntscc).mean()
            # psnr_jsccs.update(psnr_jscc)
            # psnr = CalcuPSNR_int(input_image, x_hat_ntc).mean()
            # psnrs.update(psnr)

            psnr_jscc = 10 * (torch.log(255. * 255. / mse_loss_ntscc) / np.log(10))
            psnr_jsccs.update(psnr_jscc.item())
            psnr = 10 * (torch.log(255. * 255. / mse_loss_ntc) / np.log(10))
            psnrs.update(psnr.item())
            log = (' | '.join([
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.val:.2f}',
                f'PSNR1 {psnr_jsccs.val:.2f} ({psnr_jsccs.avg:.2f})',
                f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                f'PSNR2 {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'Bpp_y {bppys.val:.2f} ({bppys.avg:.2f})',
                f'Bpp_z {bppzs.val:.4f} ({bppzs.avg:.4f})',
            ]))
            logger.info(log)
            PSNR_list.append(psnr_jscc)
            CBR_list.append(cbr_y)
            filename = os.path.join(sampled_dir, os.path.basename(img_path))
            directory = os.path.dirname(filename)
            if save:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                torchvision.utils.save_image(x_hat_ntscc, filename)
    # Here, the channel bandwidth cost of side info \bar{k} is transmitted by a capacity-achieving channel code. Note
    # that, the side info should be transmitted through entropy coding and channel coding, which will be addressed in
    # future releases.

    # capacity-achieving channel code
    cbr_sideinfo = np.log2(config.multiple_rate.__len__()) / (16 * 16 * 3) / np.log2(
        1 + 10 ** (net.channel.chan_param / 10))

    # 2/3 rate LDPC + 16QAM for AWGN SNR=10dB
    # cbr_sideinfo = np.log2(config.multiple_rate.__len__()) / (16 * 16 * 8)
    
    logger.info(f'Finish test! Average PSNR={psnr_jsccs.avg:.4f}dB, CBR={cbrs.avg + cbr_sideinfo:.4f}')
    cbr_of_the_video = (cbrs.avg+ cbr_sideinfo)*key_frame_nums/(max_key_frame_num+1)
    logger.info(f'cbr_of_the_video={cbr_of_the_video:.5f}')
    return cbr_of_the_video

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training/testing script.")
    parser.add_argument(
        "-p",
        "--phase",
        default='test',  # train
        type=str,
        help="Train or Test",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=5000,
        type=int,
        help="Number of epochs (default: %(default)s)"
    )
    parser.add_argument("--cuda", default=True, action="store_true", help="Use cuda")
    parser.add_argument(
        "--gpu-id",
        type=str,
        default=0,
        help="GPU ids (default: %(default)s)",
    )
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=1024, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        '--name',
        default=datetime.now().strftime('%Y-%m-%d_%H_%M_%S'),
        type=str,
        help='Result dir name',
    )
    parser.add_argument(
        '--save_log', action='store_true', default=True, help='Save log to disk'
    )
    parser.add_argument("--checkpoint",
                        default="checkpoints/ntscc_hyperprior_quality_4_psnr.pth",
                        type=str, help="Path to a checkpoint")
    parser.add_argument("--test_path",
                        default=os.path.join(os.environ.get("DATA_ROOT", ""), "frames"),
                        type=str, help="Path to test data (frames root); defaults to $DATA_ROOT/frames")
    parser.add_argument("--method",
                        default="key_framesinternvl_diff_0.35",
                        type=str, help="Method to extract key frames")
    parser.add_argument("--model",
                        default="10",
                        type=str, help="Model to use")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save reconstructed frames to disk")
    args = parser.parse_args(argv)
    return args

# Checkpoint locations are resolved from $NTSCC_CKPT (defaults to ./checkpoints under the
# NTSCC repo). SNR=10 uses the released quality_4 weight; SNR 0-8 use separately-trained
# weights that are NOT part of the public release (see docs/CODE_WALKTHROUGH.md provenance
# note) — set $NTSCC_CKPT_DIR_SUBxx or edit here if you have them.
_CKPT = os.environ.get("NTSCC_CKPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
_MAYU = os.environ.get("NTSCC_MAYU", os.path.join(os.path.dirname(os.path.abspath(__file__)), "mayu"))
model_dict = {
    '10': os.path.join(_CKPT, 'ntscc_hyperprior_quality_4_psnr.pth'),
    '8': os.path.join(_MAYU, 'SNR=8dB_2_0.2_train/models/best_loss_8dB.model'),
    '6': os.path.join(_MAYU, 'SNR=6dB_2_0.2_train/models/best_loss_6dB.model'),
    '4': os.path.join(_MAYU, 'SNR=4dB_2_0.2_train/models/best_loss_4dB.model'),
    '2': os.path.join(_MAYU, 'SNR=2dB_2_0.2_train/models/best_loss_2dB.model'),
    '0': os.path.join(_MAYU, 'SNR=0dB_2_0.2_train/models/best_loss_0dB.model'),
}


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    config.device = device

    workdir, logger = logger_configuration(args.name, phase=args.phase, save_log=args.save_log)
    config.logger = logger
    logger.info(config.__dict__)

    config.channel['chan_param'] = int(args.model)
    net = NTSCC_Hyperprior(config).cuda()
    # model_path = args.checkpoint
    model_path = model_dict[args.model]
    load_weights(net, model_path)

    video_cbr = AverageMeter()
    if args.phase == 'test':
        root_path = args.test_path
        # 使用os.path.isdir来过滤掉非目录项
        test_data_dir = [os.path.join(root_path, dir) for dir in os.listdir(root_path) 
                     if os.path.isdir(os.path.join(root_path, dir))]
        method = args.method
        test_data_dir = [os.path.join(dir, method) for dir in test_data_dir]
        result_cbr = []
        for dir in test_data_dir:
            name = dir.split('/')[-2]
            print(f'======Current video {name} ======')
            # print(args.save_frames)
            cur_cbr = test(net, dir, logger, args.model, args.save_frames)
            video_cbr.update(cur_cbr)
            # 将name和cur_cbr写入csv文件
            result_cbr.append((name, cur_cbr))
            # test(net, dir, logger, args.model)
        # 将result_cbr写入csv文件
        # 在root_path创建一个新的csv文件，命名为{method}.csv
        csv_path = os.path.join(root_path, f'{method}.csv')
        df = pd.DataFrame(result_cbr, columns=['name', 'cbr'])
        df.to_csv(csv_path, index=False)
        logger.info(f'Average video CBR={video_cbr.avg:.5f}')
    elif args.phase == 'train':
        train_loader, test_loader = get_loader(config)
        global global_step
        G_params = set(p for n, p in net.named_parameters() if not n.endswith(".quantiles"))
        aux_params = set(p for n, p in net.named_parameters() if n.endswith(".quantiles"))
        optimizer_G = optim.Adam(G_params, lr=config.lr)
        aux_optimizer = optim.Adam(aux_params, lr=config.aux_lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[4000, 4500], gamma=0.1)
        tot_epoch = 5000
        global_step = 0
        best_loss = float("inf")
        steps_epoch = global_step // train_loader.__len__()
        for epoch in range(steps_epoch, tot_epoch):
            logger.info('======Current epoch %s ======' % epoch)
            logger.info(f"Learning rate: {optimizer_G.param_groups[0]['lr']}")
            train_one_epoch(epoch, net, train_loader, optimizer_G, aux_optimizer, device, logger)
            lr_scheduler.step()

            loss = test(net, test_loader, logger)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if is_best:
                save_model(net, save_path=workdir + '/models/EP{}_best_loss.model'.format(epoch + 1))
                test(net, test_loader, logger)

            if (epoch + 1) % 100 == 0:
                save_model(net, save_path=workdir + '/models/EP{}.model'.format(epoch + 1))


if __name__ == '__main__':
    main(sys.argv[1:])
#python main_save.py -p test --test_path $DATA_ROOT/frames --method key_frames-baseline --model 10
