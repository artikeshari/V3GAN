from __future__ import absolute_import

import torch
import torch.nn as nn
from trainer import train, vis
from model.networks import Generator_G4, VideoDiscriminator, ImageDiscriminator
import torchvision
import transforms_vid
from dataset import WZM
# from torch.utils.tensorboard import SummaryWriter
import os
import cfg

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():

	args = cfg.parse_args()
	torch.cuda.manual_seed(args.random_seed)

	print(args)

	# create logging folder
	log_path = os.path.join(args.save_path, args.exp_name + '/log')
	model_path = os.path.join(args.save_path, args.exp_name + '/models')
	if not os.path.exists(log_path) and not os.path.exists(model_path):
		os.makedirs(log_path)
		os.makedirs(model_path)
	#writer = SummaryWriter(log_path) # tensorboard

	# load model
	device = torch.device("cuda:0")

	G = Generator_G4(args.d_za, args.d_zm, args.ch_g, args.g_mode, args.use_attention).to(device)
	VD = VideoDiscriminator(args.ch_d).to(device)
	ID = ImageDiscriminator(args.ch_d).to(device)

	G = nn.DataParallel(G)
	VD = nn.DataParallel(VD)
	ID = nn.DataParallel(ID)

	# Load pre-trained model
	pre_trained_epoch = 0 
	if args.pre_trained_model_path != '':
		print(args.pre_trained_model_path)
		pre_trained_epoch = 300
		G.load_state_dict(torch.load(os.path.join(args.pre_trained_model_path, 'G_%d.pth'%(pre_trained_epoch))))
		VD.load_state_dict(torch.load(os.path.join(args.pre_trained_model_path, 'VD_%d.pth'%(pre_trained_epoch))))
		ID.load_state_dict(torch.load(os.path.join(args.pre_trained_model_path, 'ID_%d.pth'%(pre_trained_epoch))))

	# optimizer
	optimizer_G = torch.optim.Adam(G.parameters(), args.g_lr, (0.5, 0.999))
	optimizer_VD = torch.optim.Adam(VD.parameters(), args.d_lr, (0.5, 0.999))
	optimizer_ID = torch.optim.Adam(ID.parameters(), args.d_lr, (0.5, 0.999))

	# loss
	criterion = nn.BCEWithLogitsLoss().to(device)

	# prepare dataset
	print('==> preparing dataset')
	transform = torchvision.transforms.Compose([
		transforms_vid.ClipResize((args.img_height, args.img_width)),
		transforms_vid.ClipCenterCrop(args.img_size),
		transforms_vid.ClipToTensor(),
		transforms_vid.ClipNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
	)

	dataset = WZM(args.data_path, transform=transform)

	dataloader = torch.utils.data.DataLoader(
		dataset = dataset,
		batch_size = args.batch_size,
		num_workers = args.num_workers,
		shuffle = True,
		pin_memory = True,
		drop_last = True
	)

	# for validation
	fixed_zfg = torch.randn(args.n_za_test, args.d_za, 1, 1, 1).to(device)
	fixed_zbg = torch.randn(args.n_za_test, args.d_za, 1, 1, 1).to(device)
	fixed_zm = torch.randn(args.n_zm_test, args.d_zm, 1, 1, 1).to(device)

	print('==> start training')
	for epoch in range(pre_trained_epoch, args.max_epoch):
		train(args, epoch, G, VD, ID, optimizer_G, optimizer_VD, optimizer_ID, criterion, dataloader, device)
		
		'''
		if epoch % args.val_freq == 0:
			vis(epoch, G, fixed_zfg, fixed_zbg, fixed_zm, writer, device)
		'''
		if epoch % args.save_freq == 0:
			torch.save(G.state_dict(), os.path.join(model_path, 'G_%d.pth'%(epoch)))
			torch.save(VD.state_dict(), os.path.join(model_path, 'VD_%d.pth'%(epoch)))
			torch.save(ID.state_dict(), os.path.join(model_path, 'ID_%d.pth'%(epoch)))

	return

if __name__ == '__main__':

	main()
