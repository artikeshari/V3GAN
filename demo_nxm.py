from __future__ import absolute_import

import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
from model.networks import Generator_G4
import cfg
import skvideo.io
import numpy as np
import os
from PIL import Image as im

def save_video_stitched(path, vids, n_zb, n_zm):
    
	for i in range(n_zb): # foreground loop
		for j in range(n_zm): # motion loop
			v = vids[n_zb*i + j].permute(2,0,3,1).cpu().numpy()
			v *= 255
			v = v.astype(np.uint8)							#(16,64,64,3)
			v = v.reshape(v.shape[0], v.shape[1]*v.shape[2], v.shape[3])
			filepath = os.path.join(path, "vid_%d_%d.png"%(i, j)) #'/vid-%d.png' %( t))       
			img = im.fromarray(v)
			img.save(filepath) 
	return


def save_video_frames(path, vids, n_za, n_zm):
    for i in range(n_za): 
        for j in range(n_zm):                               
            v = vids[n_za*i + j].permute(0,2,3,1).cpu().numpy()
            v *= 255
            v = v.astype(np.uint8)                          #(16,64,64,3)
            ##skvideo.io.vwrite(os.path.join(path, "%d_%d.mp4"%(i, j)), v, outputdict={"-vcodec":"libx264"})
            os.mkdir(os.path.join(path, "%d_%d"%(i, j)))
            for t in range(v.shape[0]):
                filepath = os.path.join(os.path.join(path, "%d_%d"%(i, j)) + '/vid-%d.png' %(t))        
                img = im.fromarray(v[t,:,:,:])
                img.save(filepath) 
    return

def save_video(path, vids, n_za, n_zm):

	for i in range(n_za): # appearance loop
		for j in range(n_zm): # motion loop
			v = vids[n_za*i + j].permute(0,2,3,1).cpu().numpy()
			v *= 255
			v = v.astype(np.uint8)
			skvideo.io.vwrite(os.path.join(path, "%d_%d.mp4"%(i, j)), v, outputdict={"-vcodec":"libx264"})

	return


def main():

	args = cfg.parse_args()

	# write into tensorboard
	log_path = os.path.join(args.demo_path, args.demo_name + '/log')
	vid_path = os.path.join(args.demo_path, args.demo_name + '/vids')
	vid_bg_path = os.path.join(args.demo_path, args.demo_name + '/vids_bg')
	vid_mask_path = os.path.join(args.demo_path, args.demo_name + '/vids_mask')
	vid_time_path = os.path.join(args.demo_path, args.demo_name + '/vids_time')
	vid_fg_path = os.path.join(args.demo_path, args.demo_name + '/vids_fg')

	if not os.path.exists(log_path) and not os.path.exists(vid_path):
		os.makedirs(log_path)
		os.makedirs(vid_path)
		os.makedirs(vid_bg_path)
		os.makedirs(vid_mask_path)
		os.makedirs(vid_time_path)
		os.makedirs(vid_fg_path)
	#writer = SummaryWriter(log_path)

	device = torch.device("cuda:0")

	G = Generator_G4().to(device)

	pytorch_total_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
	print(pytorch_total_params)
	
	G = nn.DataParallel(G)
	G.load_state_dict(torch.load(args.model_path))


	with torch.no_grad():
		G.eval()
		torch.manual_seed(2100)
		za = torch.randn(args.n_za_test, args.d_za, 1, 1, 1).to(device)
		zm = torch.randn(args.n_zm_test, args.d_zm, 1, 1, 1).to(device)

		n_za = za.size(0)
		n_zm = zm.size(0)
		za = za.unsqueeze(1).repeat(1, n_zm, 1, 1, 1, 1).contiguous().view(n_za*n_zm, -1, 1, 1, 1)
		zm = zm.repeat(n_za, 1, 1, 1, 1)

		vid_bg, vid_fake, vid_mask, vid_fg = G(za, zm)

		vid_fake = vid_fake.transpose(2,1) # bs x 16 x 3 x 64 x 64
		vid_fake = ((vid_fake - vid_fake.min()) / (vid_fake.max() - vid_fake.min())).data

		vid_bg = vid_bg.transpose(2,1) # bs x 16 x 3 x 64 x 64
		vid_bg = ((vid_bg - vid_bg.min()) / (vid_bg.max() - vid_bg.min())).data

		vid_mask = vid_mask.transpose(2,1) # bs x 16 x 3 x 64 x 64
		vid_mask = vid_mask.repeat(1,1,3,1,1)
		vid_mask = ((vid_mask - vid_mask.min()) / (vid_mask.max() - vid_mask.min())).data

		# vid_time = vid_time.transpose(2,1) # bs x 16 x 3 x 64 x 64
		# vid_time = ((vid_time - vid_time.min()) / (vid_time.max() - vid_time.min())).data

		vid_fg = vid_fg.transpose(2,1) # bs x 16 x 3 x 64 x 64
		vid_fg = ((vid_fg - vid_fg.min()) / (vid_fg.max() - vid_fg.min())).data

		#writer.add_video(tag='generated_videos', global_step=1, vid_tensor=vid_fake)
		#writer.flush()sssssssss

		# save into videos
		print('==> saving videos...')
		save_video(vid_path, vid_fake, n_za, n_zm)
		save_video(vid_bg_path, vid_bg, n_za, n_zm)
		save_video(vid_mask_path, vid_mask, n_za, n_zm)
		# save_video_frames(vid_time_path, vid_time, n_za, n_zm)
		save_video(vid_fg_path, vid_fg, n_za, n_zm)


	return


if __name__ == '__main__':

	main()
