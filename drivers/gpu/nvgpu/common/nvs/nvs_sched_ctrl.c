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
#include <nvgpu/dma.h>

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

struct nvgpu_nvs_domain_ctrl_fifo_queues {
	/*
	 * send indicates a buffer having data(PUT) written by a userspace client
	 * and queried by the scheduler(GET).
	 */
	struct nvgpu_nvs_ctrl_queue send;
	/*
	 * receive indicates a buffer having data(PUT) written by scheduler
	 * and queried by the userspace client(GET).
	 */
	struct nvgpu_nvs_ctrl_queue receive;

	/*
	 * event indicates a buffer that is subscribed to by userspace clients to
	 * receive events. This buffer is Read-Only for the users and only scheduler can
	 * write to it.
	 */
	struct nvgpu_nvs_ctrl_queue event;

	/*
	 * Global mutex for coarse grained access control
	 * of all Queues for all UMD interfaces. e.g. IOCTL/devctls
	 * and mmap calls. Keeping this as coarse-grained for now till
	 * GSP's implementation is complete.
	 */
	struct nvgpu_mutex queue_lock;
};

struct nvgpu_nvs_domain_ctrl_fifo {
	/*
	 * Instance of global struct gk20a;
	 */
	struct gk20a *g;

	struct nvgpu_nvs_domain_ctrl_fifo_users users;

	struct nvgpu_nvs_domain_ctrl_fifo_queues queues;

	struct nvs_domain_ctrl_fifo_capabilities capabilities;
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

	sched->capabilities.scheduler_implementation_hw = NVGPU_NVS_DOMAIN_SCHED_KMD;

	nvgpu_spinlock_init(&sched->users.user_lock);
	nvgpu_mutex_init(&sched->queues.queue_lock);
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

	nvgpu_nvs_ctrl_fifo_erase_all_queues(g);

	nvgpu_kfree(g, sched_ctrl);
	g->sched_ctrl_fifo = NULL;
}

struct nvgpu_nvs_ctrl_queue *nvgpu_nvs_ctrl_fifo_get_queue(
		struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
		enum nvgpu_nvs_ctrl_queue_num queue_num,
		enum nvgpu_nvs_ctrl_queue_direction queue_direction,
		u8 *mask)
{
	struct nvgpu_nvs_ctrl_queue *queue = NULL;

	if (sched_ctrl == NULL) {
		return NULL;
	}

	if (mask == NULL) {
		return NULL;
	}

	if (queue_num == NVGPU_NVS_NUM_CONTROL) {
		if (queue_direction == NVGPU_NVS_DIR_CLIENT_TO_SCHEDULER) {
			queue = &sched_ctrl->queues.send;
			*mask = NVGPU_NVS_CTRL_FIFO_QUEUE_EXCLUSIVE_CLIENT_WRITE;
		} else if (queue_direction == NVGPU_NVS_DIR_SCHEDULER_TO_CLIENT) {
			queue = &sched_ctrl->queues.receive;
			*mask = NVGPU_NVS_CTRL_FIFO_QUEUE_EXCLUSIVE_CLIENT_READ;
		}
	} else if (queue_num == NVGPU_NVS_NUM_EVENT) {
		if (queue_direction == NVGPU_NVS_DIR_SCHEDULER_TO_CLIENT) {
			queue = &sched_ctrl->queues.event;
			*mask = NVGPU_NVS_CTRL_FIFO_QUEUE_CLIENT_EVENTS_READ;
		}
	}

	return queue;
}

struct nvs_domain_ctrl_fifo_capabilities *nvgpu_nvs_ctrl_fifo_get_capabilities(
		struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl)
{
	return &sched_ctrl->capabilities;
}

bool nvgpu_nvs_buffer_is_valid(struct gk20a *g, struct nvgpu_nvs_ctrl_queue *buf)
{
	return buf->valid;
}

int nvgpu_nvs_buffer_alloc(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
		size_t bytes, u8 mask, struct nvgpu_nvs_ctrl_queue *buf)
{
	int err;
	struct gk20a *g = sched_ctrl->g;
	struct vm_gk20a *system_vm = g->mm.pmu.vm;

	(void)memset(buf, 0, sizeof(*buf));
	buf->g = g;

	err = nvgpu_dma_alloc_map_sys(system_vm, bytes, &buf->mem);
	if (err != 0) {
		nvgpu_err(g, "failed to allocate memory for dma");
		goto fail;
	}

	buf->valid = true;
	buf->mask = mask;

	return 0;

fail:
	(void)memset(buf, 0, sizeof(*buf));

	return err;
}

void nvgpu_nvs_buffer_free(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
		struct nvgpu_nvs_ctrl_queue *buf)
{
	struct gk20a *g = sched_ctrl->g;
	struct vm_gk20a *system_vm = g->mm.pmu.vm;

	if (nvgpu_mem_is_valid(&buf->mem)) {
		nvgpu_dma_unmap_free(system_vm, &buf->mem);
	}

	/* Sets buf->valid as false */
	(void)memset(buf, 0, sizeof(*buf));
}

void nvgpu_nvs_ctrl_fifo_lock_queues(struct gk20a *g)
{
	struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl = g->sched_ctrl_fifo;
	nvgpu_mutex_acquire(&sched_ctrl->queues.queue_lock);
}

void nvgpu_nvs_ctrl_fifo_unlock_queues(struct gk20a *g)
{
	struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl = g->sched_ctrl_fifo;
	nvgpu_mutex_release(&sched_ctrl->queues.queue_lock);
}

bool nvgpu_nvs_ctrl_fifo_queue_has_subscribed_users(struct nvgpu_nvs_ctrl_queue *queue)
{
	return queue->ref != 0;
}

void nvgpu_nvs_ctrl_fifo_user_subscribe_queue(struct nvs_domain_ctrl_fifo_user *user,
		struct nvgpu_nvs_ctrl_queue *queue)
{
	user->active_used_queues |= queue->mask;
	queue->ref++;
}
void nvgpu_nvs_ctrl_fifo_user_unsubscribe_queue(struct nvs_domain_ctrl_fifo_user *user,
		struct nvgpu_nvs_ctrl_queue *queue)
{
	user->active_used_queues &= ~queue->mask;
	queue->ref--;
}
bool nvgpu_nvs_ctrl_fifo_user_is_subscribed_to_queue(struct nvs_domain_ctrl_fifo_user *user,
		struct nvgpu_nvs_ctrl_queue *queue)
{
	return (user->active_used_queues & queue->mask);
}

void nvgpu_nvs_ctrl_fifo_erase_all_queues(struct gk20a *g)
{
	struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl = g->sched_ctrl_fifo;

	nvgpu_nvs_ctrl_fifo_lock_queues(g);

	if (nvgpu_nvs_buffer_is_valid(g, &sched_ctrl->queues.send)) {
		nvgpu_nvs_ctrl_fifo_erase_queue(g, &sched_ctrl->queues.send);
	}

	if (nvgpu_nvs_buffer_is_valid(g, &sched_ctrl->queues.receive)) {
		nvgpu_nvs_ctrl_fifo_erase_queue(g, &sched_ctrl->queues.receive);
	}

	if (nvgpu_nvs_buffer_is_valid(g, &sched_ctrl->queues.event)) {
		nvgpu_nvs_ctrl_fifo_erase_queue(g, &sched_ctrl->queues.event);
	}

	nvgpu_nvs_ctrl_fifo_unlock_queues(g);
}

void nvgpu_nvs_ctrl_fifo_erase_queue(struct gk20a *g, struct nvgpu_nvs_ctrl_queue *queue)
{
	if (queue->free != NULL) {
		queue->free(g, queue);
	}
}
