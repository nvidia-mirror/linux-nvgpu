/*
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __NVGPU_IOCTL_CTRL_H__
#define __NVGPU_IOCTL_CTRL_H__

struct gk20a_ctrl_priv;
struct nvgpu_tsg;

int gk20a_ctrl_dev_open(struct inode *inode, struct file *filp);
int gk20a_ctrl_dev_release(struct inode *inode, struct file *filp);
long gk20a_ctrl_dev_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);
int gk20a_ctrl_dev_mmap(struct file *filp, struct vm_area_struct *vma);

void nvgpu_hide_usermode_for_poweroff(struct gk20a *g);
void nvgpu_restore_usermode_for_poweron(struct gk20a *g);

#ifdef CONFIG_NVGPU_TSG_SHARING
u64 nvgpu_gpu_get_device_instance_id(struct gk20a_ctrl_priv *priv);
int nvgpu_gpu_get_share_token(struct gk20a *g,
			      u64 source_device_instance_id,
			      u64 target_device_instance_id,
			      struct nvgpu_tsg *tsg,
			      u64 *share_token);
int nvgpu_gpu_revoke_share_token(struct gk20a *g,
				 u64 source_device_instance_id,
				 u64 target_device_instance_id,
				 u64 share_token,
				 struct nvgpu_tsg *tsg);
int nvgpu_gpu_tsg_revoke_share_tokens(struct gk20a *g,
				      u64 source_device_instance_id,
				      struct nvgpu_tsg *tsg,
				      u32 *out_count);
struct nvgpu_tsg *nvgpu_gpu_open_tsg_with_share_token(struct gk20a *g,
				 u64 source_device_instance_id,
				 u64 target_device_instance_id,
				 u64 share_token);
#endif

#endif
