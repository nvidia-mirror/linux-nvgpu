/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef NVGPU_NVS_H
#define NVGPU_NVS_H

#ifdef CONFIG_NVS_PRESENT
#include <nvs/domain.h>
#endif

#include <nvgpu/atomic.h>
#include <nvgpu/lock.h>
#include <nvgpu/worker.h>
#include <nvgpu/timers.h>
#include <nvgpu/nvgpu_mem.h>

/*
 * Max size we'll parse from an NVS log entry.
 */
#define NVS_LOG_BUF_SIZE	128
/*
 * Keep it to page size for now. Can be updated later.
 */
#define NVS_QUEUE_DEFAULT_SIZE (64 * 1024)

struct gk20a;
struct nvgpu_nvs_domain_ioctl;
struct nvgpu_runlist;
struct nvgpu_runlist_domain;
struct nvgpu_nvs_ctrl_queue;
struct nvgpu_nvs_domain_ctrl_fifo;

/* Structure to store user info common to all schedulers */
struct nvs_domain_ctrl_fifo_user {
	/*
	 * Flag to determine whether the user has write access.
	 * User having write access can update Request/Response buffers.
	 */
	bool has_write_access;
	/*
	 * PID of the user. Used to prevent a given user from opening
	 * multiple instances of control-fifo device node.
	 */
	int pid;
	/* Mask of actively used queue */
	u32 active_used_queues;
	/*
	 * Listnode used for keeping references to the user in
	 * the master struct nvgpu_nvs_domain_ctrl_fifo
	 */
	struct nvgpu_list_node sched_ctrl_list;
};

static inline struct nvs_domain_ctrl_fifo_user *
nvs_domain_ctrl_fifo_user_from_sched_ctrl_list(struct nvgpu_list_node *node)
{
	return (struct nvs_domain_ctrl_fifo_user *)
		((uintptr_t)node - offsetof(struct nvs_domain_ctrl_fifo_user, sched_ctrl_list));
};

/*
 * NvGPU KMD domain implementation details for nvsched.
 */
struct nvgpu_nvs_domain {
	u64 id;

	/*
	 * Subscheduler ID to define the scheduling within a domain. These will
	 * be implemented by the kernel as needed. There'll always be at least
	 * one, which is the host HW built in round-robin scheduler.
	 */
	u32 subscheduler;

	/*
	 * Convenience pointer for linking back to the parent object.
	 */
	struct nvs_domain *parent;

	/*
	 * Domains are dynamically used by their participant TSGs and the
	 * runlist HW. A refcount prevents them from getting prematurely freed.
	 *
	 * This is not the usual refcount. The primary owner is userspace via the
	 * ioctl layer and a TSG putting a ref does not result in domain deletion.
	 */
	u32 ref;

	/*
	 * Userspace API on the device nodes.
	 */
	struct nvgpu_nvs_domain_ioctl *ioctl;

	/*
	 * One corresponding to every runlist
	 */
	struct nvgpu_runlist_domain **rl_domains;
};

struct nvgpu_nvs_worker {
	nvgpu_atomic_t nvs_sched_init;
	struct nvgpu_cond wq_init;
	struct nvgpu_worker worker;
	struct nvgpu_timeout timeout;
	u32 current_timeout;
};

struct nvgpu_nvs_scheduler {
	struct nvs_sched *sched;
	nvgpu_atomic64_t id_counter;
	struct nvgpu_nvs_worker worker;
	struct nvgpu_nvs_domain *active_domain;
	struct nvgpu_nvs_domain *shadow_domain;
};

enum nvgpu_nvs_ctrl_queue_num {
	NVGPU_NVS_NUM_CONTROL = 0,
	NVGPU_NVS_NUM_EVENT,
	NVGPU_NVS_INVALID,
};

enum nvgpu_nvs_ctrl_queue_direction {
	NVGPU_NVS_DIR_CLIENT_TO_SCHEDULER = 0,
	NVGPU_NVS_DIR_SCHEDULER_TO_CLIENT,
	NVGPU_NVS_DIR_INVALID,
};

/*
 * The below definitions mirror the nvgpu-nvs(UAPI)
 * headers.
 */

/*
 * Invalid domain scheduler.
 * The value of 'domain_scheduler_implementation'
 * when 'has_domain_scheduler_control_fifo' is 0.
 */
#define NVGPU_NVS_DOMAIN_SCHED_INVALID 0U
/*
 * CPU based scheduler implementation. Intended use is mainly
 * for debug and testing purposes. Doesn't meet latency requirements.
 * Implementation will be supported in the initial versions and eventually
 * discarded.
 */
#define NVGPU_NVS_DOMAIN_SCHED_KMD 1U
/*
 * GSP based scheduler implementation that meets latency requirements.
 * This implementation will eventually replace NVGPU_NVS_DOMAIN_SCHED_KMD.
 */
#define NVGPU_NVS_DOMAIN_SCHED_GSP 2U

/* Queue meant for exclusive client write access. This shared queue will be
 * used for communicating the scheduling metadata between the client(producer)
 * and scheduler(consumer).
 */
#define NVGPU_NVS_CTRL_FIFO_QUEUE_EXCLUSIVE_CLIENT_WRITE 1U

/* Queue meant for exclusive client read access. This shared queue will be
 * used for communicating the scheduling metadata between the scheduler(producer)
 * and client(consumer).
 */
#define NVGPU_NVS_CTRL_FIFO_QUEUE_EXCLUSIVE_CLIENT_READ 2U

/* Queue meant for generic read access. Clients can subscribe to this read-only
 * queue for processing events such as recovery, preemption etc.
 */
#define NVGPU_NVS_CTRL_FIFO_QUEUE_CLIENT_EVENTS_READ 4U

/*
 * Direction of the requested queue is from CLIENT(producer)
 * to SCHEDULER(consumer).
 */
#define NVGPU_NVS_CTRL_FIFO_QUEUE_DIRECTION_CLIENT_TO_SCHEDULER 0

/*
 * Direction of the requested queue is from SCHEDULER(producer)
 * to CLIENT(consumer).
 */
#define NVGPU_NVS_CTRL_FIFO_QUEUE_DIRECTION_SCHEDULER_TO_CLIENT 1

/* Structure to hold control_queues. This can be then passed to GSP or Rm based subscheduler. */
struct nvgpu_nvs_ctrl_queue {
	struct nvgpu_mem	mem;
	struct gk20a		*g;
	/*
	 * Filled in by each OS - this holds the necessary data to export this
	 * buffer to userspace.
	 */
	void			*priv;
	bool			valid;
	u8				mask;
	void (*free)(struct gk20a *g, struct nvgpu_nvs_ctrl_queue *queue);
};

#ifdef CONFIG_NVS_PRESENT
int nvgpu_nvs_init(struct gk20a *g);
int nvgpu_nvs_open(struct gk20a *g);
void nvgpu_nvs_remove_support(struct gk20a *g);
void nvgpu_nvs_get_log(struct gk20a *g, s64 *timestamp, const char **msg);
u32 nvgpu_nvs_domain_count(struct gk20a *g);
int nvgpu_nvs_del_domain(struct gk20a *g, u64 dom_id);
int nvgpu_nvs_add_domain(struct gk20a *g, const char *name, u64 timeslice,
			 u64 preempt_grace, struct nvgpu_nvs_domain **pdomain);
void nvgpu_nvs_print_domain(struct gk20a *g, struct nvgpu_nvs_domain *domain);

struct nvgpu_nvs_domain *
nvgpu_nvs_domain_by_id(struct gk20a *g, u64 domain_id);
struct nvgpu_nvs_domain *
nvgpu_nvs_domain_by_name(struct gk20a *g, const char *name);
void nvgpu_nvs_domain_get(struct gk20a *g, struct nvgpu_nvs_domain *dom);
void nvgpu_nvs_domain_put(struct gk20a *g, struct nvgpu_nvs_domain *dom);
const char *nvgpu_nvs_domain_get_name(struct nvgpu_nvs_domain *dom);
/*
 * Debug wrapper for NVS code.
 */
#define nvs_dbg(g, fmt, arg...)			\
	nvgpu_log(g, gpu_dbg_nvs, fmt, ##arg)

void nvgpu_nvs_ctrl_fifo_lock_queues(struct gk20a *g);
void nvgpu_nvs_ctrl_fifo_unlock_queues(struct gk20a *g);

struct nvgpu_nvs_domain_ctrl_fifo *nvgpu_nvs_ctrl_fifo_create(struct gk20a *g);
bool nvgpu_nvs_ctrl_fifo_user_exists(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
    int pid, bool rw);
bool nvgpu_nvs_ctrl_fifo_is_busy(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl);
void nvgpu_nvs_ctrl_fifo_destroy(struct gk20a *g);
bool nvgpu_nvs_ctrl_fifo_user_is_active(struct nvs_domain_ctrl_fifo_user *user);
void nvgpu_nvs_ctrl_fifo_add_user(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
    struct nvs_domain_ctrl_fifo_user *user);
bool nvgpu_nvs_ctrl_fifo_is_exclusive_user(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
    struct nvs_domain_ctrl_fifo_user *user);
void nvgpu_nvs_ctrl_fifo_reset_exclusive_user(
		struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl, struct nvs_domain_ctrl_fifo_user *user);
int nvgpu_nvs_ctrl_fifo_reserve_exclusive_user(
		struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl, struct nvs_domain_ctrl_fifo_user *user);
void nvgpu_nvs_ctrl_fifo_remove_user(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
		struct nvs_domain_ctrl_fifo_user *user);
struct nvgpu_nvs_ctrl_queue *nvgpu_nvs_ctrl_fifo_get_queue(
		struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
		enum nvgpu_nvs_ctrl_queue_num queue_num,
		enum nvgpu_nvs_ctrl_queue_direction queue_direction,
		u8 *mask);
/* Below methods require nvgpu_nvs_ctrl_fifo_lock_queues() to be held. */
bool nvgpu_nvs_buffer_is_valid(struct gk20a *g, struct nvgpu_nvs_ctrl_queue *buf);
int nvgpu_nvs_buffer_alloc(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
		size_t bytes, u8 mask, struct nvgpu_nvs_ctrl_queue *buf);
void nvgpu_nvs_buffer_free(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
		struct nvgpu_nvs_ctrl_queue *buf);
void nvgpu_nvs_ctrl_fifo_user_subscribe_queue(struct nvs_domain_ctrl_fifo_user *user,
		struct nvgpu_nvs_ctrl_queue *queue);
void nvgpu_nvs_ctrl_fifo_user_unsubscribe_queue(struct nvs_domain_ctrl_fifo_user *user,
		struct nvgpu_nvs_ctrl_queue *queue);
bool nvgpu_nvs_ctrl_fifo_user_is_subscribed_to_queue(struct nvs_domain_ctrl_fifo_user *user,
		struct nvgpu_nvs_ctrl_queue *queue);
void nvgpu_nvs_ctrl_fifo_erase_queue(struct gk20a *g, struct nvgpu_nvs_ctrl_queue *queue);
void nvgpu_nvs_ctrl_fifo_erase_all_queues(struct gk20a *g);

#else
static inline int nvgpu_nvs_init(struct gk20a *g)
{
	(void)g;
	return 0;
}

static inline void nvgpu_nvs_remove_support(struct gk20a *g)
{
	(void)g;
}

static inline struct nvgpu_nvs_domain *
nvgpu_nvs_domain_by_name(struct gk20a *g, const char *name)
{
	(void)g;
	(void)name;
	return NULL;
}

static inline void nvgpu_nvs_domain_put(struct gk20a *g, struct nvgpu_nvs_domain *dom)
{
	(void)g;
	(void)dom;
}

static inline const char *nvgpu_nvs_domain_get_name(struct nvgpu_nvs_domain *dom)
{
	(void)dom;
	return NULL;
}
#endif

#endif
