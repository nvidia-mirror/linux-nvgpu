/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <nvs/log.h>
#include <nvgpu/nvs.h>
#include <nvgpu/lock.h>
#include <nvgpu/kmem.h>
#include <nvgpu/gk20a.h>
#include <nvgpu/list.h>

struct nvgpu_nvs_domain_ctrl_fifo_users {
	/* Flag to reserve exclusive user */
	bool reserved_exclusive_rw_user;
	/* Store the single Read/Write User */
	struct nvgpu_list_node exclusive_user;
	/* Store multiple Read-Only events subscriber e.g. debugger etc. */
	struct nvgpu_list_node list_non_exclusive_user;
	/* Active users available */
	u32 usage_counter;

	struct nvgpu_spinlock user_lock;
};

struct nvgpu_nvs_domain_ctrl_fifo {
	/*
	 * Instance of global struct gk20a;
	 */
	struct gk20a *g;

	struct nvgpu_nvs_domain_ctrl_fifo_users users;
};

void nvgpu_nvs_ctrl_fifo_reset_exclusive_user(
		struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl, struct nvs_domain_ctrl_fifo_user *user)
{
	nvgpu_spinlock_acquire(&sched_ctrl->users.user_lock);
	nvgpu_list_del(&user->sched_ctrl_list);
	nvgpu_list_add_tail(&user->sched_ctrl_list, &sched_ctrl->users.list_non_exclusive_user);
	nvgpu_spinlock_release(&sched_ctrl->users.user_lock);
}

int nvgpu_nvs_ctrl_fifo_reserve_exclusive_user(
		struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl, struct nvs_domain_ctrl_fifo_user *user)
{
	int ret = 0;

	if (!user->has_write_access) {
		return -EPERM;
	}

	nvgpu_spinlock_acquire(&sched_ctrl->users.user_lock);

	if (nvgpu_list_empty(&sched_ctrl->users.exclusive_user)) {
		nvgpu_list_del(&user->sched_ctrl_list);
		nvgpu_list_add_tail(&user->sched_ctrl_list, &sched_ctrl->users.exclusive_user);
	} else {
		ret = -EBUSY;
	}

	nvgpu_spinlock_release(&sched_ctrl->users.user_lock);

	return ret;
}

bool nvgpu_nvs_ctrl_fifo_user_exists(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
	int pid, bool rw)
{
	bool user_exists = false;
	struct nvs_domain_ctrl_fifo_user *user;

	nvgpu_spinlock_acquire(&sched_ctrl->users.user_lock);

	nvgpu_list_for_each_entry(user, &sched_ctrl->users.list_non_exclusive_user,
			nvs_domain_ctrl_fifo_user, sched_ctrl_list) {
		if (user->pid == pid) {
			user_exists = true;
			break;
		}
	}

	if (!user_exists) {
		if (!nvgpu_list_empty(&sched_ctrl->users.exclusive_user)) {
			user = nvgpu_list_first_entry(&sched_ctrl->users.exclusive_user,
					nvs_domain_ctrl_fifo_user, sched_ctrl_list);
			if (user->pid == pid) {
				user_exists = true;
			}
		}
	}

	nvgpu_spinlock_release(&sched_ctrl->users.user_lock);

	return user_exists;
}

bool nvgpu_nvs_ctrl_fifo_is_exclusive_user(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
    struct nvs_domain_ctrl_fifo_user *user)
{
	bool result = false;

	struct nvs_domain_ctrl_fifo_user *exclusive_user = NULL;

	nvgpu_spinlock_acquire(&sched_ctrl->users.user_lock);

	if (!nvgpu_list_empty(&sched_ctrl->users.exclusive_user)) {
		exclusive_user = nvgpu_list_first_entry(&sched_ctrl->users.exclusive_user,
					nvs_domain_ctrl_fifo_user, sched_ctrl_list);

		if (exclusive_user == user) {
			result = true;
		}
	}

	nvgpu_spinlock_release(&sched_ctrl->users.user_lock);

	return result;
}

void nvgpu_nvs_ctrl_fifo_add_user(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
	struct nvs_domain_ctrl_fifo_user *user)
{
	nvgpu_spinlock_acquire(&sched_ctrl->users.user_lock);

	nvgpu_list_add(&user->sched_ctrl_list, &sched_ctrl->users.list_non_exclusive_user);

	sched_ctrl->users.usage_counter++;

	nvgpu_spinlock_release(&sched_ctrl->users.user_lock);
}

bool nvgpu_nvs_ctrl_fifo_user_is_active(struct nvs_domain_ctrl_fifo_user *user)
{
	return user->active_used_queues != 0;
}

void nvgpu_nvs_ctrl_fifo_remove_user(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
	struct nvs_domain_ctrl_fifo_user *user)
{
	nvgpu_spinlock_acquire(&sched_ctrl->users.user_lock);

	nvgpu_list_del(&user->sched_ctrl_list);

	sched_ctrl->users.usage_counter--;

	nvgpu_spinlock_release(&sched_ctrl->users.user_lock);
}

struct nvgpu_nvs_domain_ctrl_fifo *nvgpu_nvs_ctrl_fifo_create(struct gk20a *g)
{
	struct nvgpu_nvs_domain_ctrl_fifo *sched = nvgpu_kzalloc(g, sizeof(*sched));

	if (sched == NULL) {
		return NULL;
	}

	nvgpu_spinlock_init(&sched->users.user_lock);
	nvgpu_init_list_node(&sched->users.exclusive_user);
	nvgpu_init_list_node(&sched->users.list_non_exclusive_user);

	return sched;
}

bool nvgpu_nvs_ctrl_fifo_is_busy(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl)
{
	bool ret = 0;

	nvgpu_spinlock_acquire(&sched_ctrl->users.user_lock);
	ret = (sched_ctrl->users.usage_counter != 0);
	nvgpu_spinlock_release(&sched_ctrl->users.user_lock);

	return ret;
}

void nvgpu_nvs_ctrl_fifo_destroy(struct gk20a *g)
{
	struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl = g->sched_ctrl_fifo;

	if (sched_ctrl == NULL) {
		return;
	}

	nvgpu_assert(!nvgpu_nvs_ctrl_fifo_is_busy(sched_ctrl));

	nvgpu_kfree(g, sched_ctrl);
	g->sched_ctrl_fifo = NULL;
}
