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
#include <nvgpu/nvs-control-interface-parser.h>

/*
 * Max size we'll parse from an NVS log entry.
 */
#define NVS_LOG_BUF_SIZE	128
/*
 * Keep it to page size for now. Can be updated later.
 */
#define NVS_QUEUE_DEFAULT_SIZE (64U * 1024U)

struct gk20a;
struct nvgpu_nvs_domain_ioctl;
struct nvgpu_runlist;
struct nvgpu_runlist_domain;
struct nvgpu_nvs_ctrl_queue;
struct nvgpu_nvs_domain_ctrl_fifo;
struct nvgpu_nvs_domain;

struct nvs_domain_ctrl_fifo_capabilities {
	/* Store type of scheduler backend */
	uint8_t scheduler_implementation_hw;
};

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
	 * Used to hold the scheduler capabilities.
	 */
	struct nvs_domain_ctrl_fifo_capabilities capabilities;
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

/**
 * NvGPU KMD domain implementation details for nvsched.
 */
struct nvgpu_nvs_domain {
	u64 id;

	/**
	 * Subscheduler ID to define the scheduling within a domain. These will
	 * be implemented by the kernel as needed. There'll always be at least
	 * one, which is the host HW built in round-robin scheduler.
	 */
	u32 subscheduler;

	/**
	 * Convenience pointer for linking back to the parent object.
	 */
	struct nvs_domain *parent;

	/**
	 * Domains are dynamically used by their participant TSGs and the
	 * runlist HW. A refcount prevents them from getting prematurely freed.
	 *
	 * This is not the usual refcount. The primary owner is userspace via the
	 * ioctl layer and a TSG putting a ref does not result in domain deletion.
	 */
	u32 ref;

	/**
	 * Userspace API on the device nodes.
	 */
	struct nvgpu_nvs_domain_ioctl *ioctl;

	/**
	 * One corresponding to every runlist
	 */
	struct nvgpu_runlist_domain **rl_domains;
};

#define NVS_WORKER_STATE_STOPPED 0
#define NVS_WORKER_STATE_RUNNING 1
#define NVS_WORKER_STATE_SHOULD_PAUSE 2
#define NVS_WORKER_STATE_PAUSED 3
#define NVS_WORKER_STATE_SHOULD_RESUME 4

struct nvgpu_nvs_worker {
	nvgpu_atomic_t nvs_sched_state;
	struct nvgpu_cond wq_request;
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
	u8				ref;
	void (*free)(struct gk20a *g, struct nvgpu_nvs_ctrl_queue *queue);
};

#ifdef CONFIG_NVS_PRESENT
/**
 * @brief This API is used to initialize NVS services
 *
 * Initializes \a g->sched_mutex and invokes \a nvgpu_nvs_open
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 *
 * @retval 0 on success.
 * @retval failure codes of \a nvgpu_nvs_open
 */
int nvgpu_nvs_init(struct gk20a *g);

/**
 * @brief Initialize NVS metadata and setup shadow domain.
 *
 * The initialization is done under a critical section defined by a global
 * scheduling lock.
 *
 * 1) Initialize the metadata required for NVS subject to whether
 *    the flag NVGPU_SUPPORT_NVS_CTRL_FIFO is supported. Return early,
 *    if the metadata is already initialized.
 * 2) Initialize NVS's internal ID counter used for tracking Domain IDs.
 * 3) Construct the Control-Fifo master structure and store it as part
 *    of \a g->sched_ctrl_fifo.
 * 4) Construct a global list for storing domains and initialize the counters
 *    associated with it.
 * 5) Generate the global Shadow Domain. The ID of the shadow domain is set to U64_MAX
 *    as well as timeslice set to 100U * NSEC_PER_MSEC. Ensure the Shadow Domain
 *    is linked with the corresponding Shadow Runlist Domains.
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 *
 * @retval 0 on success.
 * @retval ENOMEM Failed to allocate enough memory.
 */
int nvgpu_nvs_open(struct gk20a *g);

/**
 * @brief Remove support for NVS.
 *
 * 1) Erase all existing struct nvgpu_dom in NVS.
 * 2) Erase the Shadow Domain.
 * 3) Release the metadata required for NVS.
 * 4) Remove Control Fifo support if its enabled.
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 */
void nvgpu_nvs_remove_support(struct gk20a *g);
void nvgpu_nvs_get_log(struct gk20a *g, s64 *timestamp, const char **msg);

/**
 * @brief Return the number of active domains in the global list
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 * @return u32 Count of active domains.
 */
u32 nvgpu_nvs_domain_count(struct gk20a *g);

/**
 * @brief Erase a domain metadata corresponding to a given domain id.
 *
 * The removal of the metedata is done under a critical section defined by a global
 * scheduling lock.
 *
 * 1) Check if dom_id is valid i.e. a valid domain metadata(struct nvgpu_nvs_domain) exists.
 * 2) Check if the domain's reference counter is not one to ensure no existing
 *    user exists.
 * 3) Unlink the RL domain metadata corresponding to this domain metadata.
 * 4) Free RL domain metadata specific memory.
 * 5) Set NVS's active domain to the next domain, if no other domain exists,
 *    set the shadow domain as the active domain.
 * 6) Unlink strut nvgpu_nvs_domain and its corresponding nvs_domain
 * 7) Free domain metadata(struct nvgpu_nvs_domain).
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 * @param dom_id Domain Id for which the domain needs to be erased.
 *
 * @retval 0 on success.
 * @retval ENOENT if domain doesn't exist.
 * @retval EBUSY If domain is already being used i.e domain's reference
 *         counter is not 1.
 */
int nvgpu_nvs_del_domain(struct gk20a *g, u64 dom_id);

/**
 * @brief Create Domain Metadata and Link with RL domain
 *
 * The initialization is done under a critical section defined by a global
 * scheduling lock.
 *
 * 1) Verify if name already doesn't exist, otherwise return failure.
 * 2) Generate a struct nvgpu_nvs_domain, an internal struct nvs_domain,
 *    add their corresponding linkages. i.e. associate nvgpu_nvs_domain
 *    as a priv of nvs_domain and set nvs_domain as the parent of nvgpu_nvs_domain.
 * 3) Increment the global domain ID counter and set the domain's ID to the same.
 * 4) Set the corresponding timeslice and preempt_grace values.
 * 5) Create a struct nvgpu_runlist_domain corresponding to each engines and associate
 *    them with the above struct nvgpu_nvs_domain.
 * 6) Link the struct nvs_domain in a global list.
 * 7) Set the struct nvgpu_nvs_domain's address to pdomain.
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 * @param name [in] Name of the domain. Must not be NULL and must not already exist.
 * @param timeslice [in] The default timeslice of the Domain. Function does not perform any
 *	validation of the parameter.
 * @param preempt_grace [in] The default preempt_grace of the Domain. Function does not perform any
 *	validation of the parameter.
 * @param pdomain [out] Placeholder for returning the constructed Domain pointer. Must be non NULL.
 *
 * @retval 0 on success.
 * @retval EINVAL name is NULL or pdomain is NULL.
 * @retval EEXIST If name already exists
 * @retval ENOMEM Memory allocation failure
 */
int nvgpu_nvs_add_domain(struct gk20a *g, const char *name, u64 timeslice,
			 u64 preempt_grace, struct nvgpu_nvs_domain **pdomain);

/**
 * @brief Print domain attributes
 *
 * Print domain attributes such as name, timeslice, preempt_grace
 * and ID.
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 * @param dom Input Domain. Need null check
 */
void nvgpu_nvs_print_domain(struct gk20a *g, struct nvgpu_nvs_domain *domain);

/**
 * @brief Get a pointer to a corresponding domain metadata using ID
 *
 * Within a global scheduling lock, check if the corresponding domain ID mapping
 * exists and increment its reference counter.
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 * @param domain_id Domain Id required for input.
 *
 * @retval NULL If domain_id doesn't exist.
 * @retval correct pointer to struct nvgpu_nvs_domain
 */
struct nvgpu_nvs_domain *
nvgpu_nvs_domain_by_id(struct gk20a *g, u64 domain_id);

/**
 * @brief Search for instance of nvgpu_nvs_domain by name.
 *
 * Within a global scheduling lock, check if the corresponding domain name
 * exists and increment its reference counter and return the instance.
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 * @param name Name of the domain to search for. Must not be NULL.
 *
 * @retval NULL If domain is null or doesn't exist.
 * @retval correct pointer to instance of struct nvgpu_nvs_domain
 */
struct nvgpu_nvs_domain *
nvgpu_nvs_domain_by_name(struct gk20a *g, const char *name);

/**
 * @brief Increment the domain's reference counter
 *
 * Within a global scheduling lock, increment the corresponding domain's
 * reference counter. Warn if zero already before increment as the init
 * value is one.
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 * @param dom Input Domain. Need null check
 */
void nvgpu_nvs_domain_get(struct gk20a *g, struct nvgpu_nvs_domain *dom);

/**
 * @brief Decrement the domain's reference counter
 *
 * Within a global scheduling lock, decrement the corresponding domain's
 * reference counter. Assert that value after decrement stays greater than
 * zero.
 *
 * @param g [in] The GPU super structure. Function does not perform any
 *	validation of the parameter.
 * @param dom Input Domain. Need null check
 */
void nvgpu_nvs_domain_put(struct gk20a *g, struct nvgpu_nvs_domain *dom);

/**
 * @brief Return name corresponding to the domain.
 *
 * @param dom Input Domain, Need validation check
 *
 * @retval Return name of the domain.
 * @retval NULL If domain_id doesn't exist.
 */
const char *nvgpu_nvs_domain_get_name(struct nvgpu_nvs_domain *dom);
/*
 * Debug wrapper for NVS code.
 */
#define nvs_dbg(g, fmt, arg...)			\
	nvgpu_log(g, gpu_dbg_nvs, fmt, ##arg)

void nvgpu_nvs_ctrl_fifo_lock_queues(struct gk20a *g);
void nvgpu_nvs_ctrl_fifo_unlock_queues(struct gk20a *g);

#ifdef CONFIG_KMD_SCHEDULING_WORKER_THREAD
void nvgpu_nvs_worker_pause(struct gk20a *g);
void nvgpu_nvs_worker_resume(struct gk20a *g);
#endif

bool nvgpu_nvs_ctrl_fifo_is_enabled(struct gk20a *g);
struct nvgpu_nvs_domain_ctrl_fifo *nvgpu_nvs_ctrl_fifo_create(struct gk20a *g);
bool nvgpu_nvs_ctrl_fifo_user_exists(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
    int pid, bool rw);
void nvgpu_nvs_ctrl_fifo_idle(struct gk20a *g);
void nvgpu_nvs_ctrl_fifo_unidle(struct gk20a *g);
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
struct nvs_domain_ctrl_fifo_capabilities *nvgpu_nvs_ctrl_fifo_get_capabilities(
		struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl);
struct nvgpu_nvs_ctrl_queue *nvgpu_nvs_ctrl_fifo_get_queue(
		struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
		enum nvgpu_nvs_ctrl_queue_num queue_num,
		enum nvgpu_nvs_ctrl_queue_direction queue_direction,
		u8 *mask);

#ifdef CONFIG_KMD_SCHEDULING_WORKER_THREAD
struct nvs_control_fifo_receiver *nvgpu_nvs_domain_ctrl_fifo_get_receiver(struct gk20a *g);
struct nvs_control_fifo_sender *nvgpu_nvs_domain_ctrl_fifo_get_sender(struct gk20a *g);
void nvgpu_nvs_domain_ctrl_fifo_set_receiver(struct gk20a *g,
		struct nvs_control_fifo_receiver *receiver);
void nvgpu_nvs_domain_ctrl_fifo_set_sender(struct gk20a *g,
		struct nvs_control_fifo_sender *sender);
int nvgpu_nvs_ctrl_fifo_scheduler_handle_requests(struct gk20a *g);
#endif

/* Below methods require nvgpu_nvs_ctrl_fifo_lock_queues() to be held. */
bool nvgpu_nvs_buffer_is_valid(struct gk20a *g, struct nvgpu_nvs_ctrl_queue *buf);
int nvgpu_nvs_buffer_alloc(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
		size_t bytes, u8 mask, struct nvgpu_nvs_ctrl_queue *buf);
void nvgpu_nvs_buffer_free(struct nvgpu_nvs_domain_ctrl_fifo *sched_ctrl,
		struct nvgpu_nvs_ctrl_queue *buf);
bool nvgpu_nvs_ctrl_fifo_queue_has_subscribed_users(struct nvgpu_nvs_ctrl_queue *queue);
void nvgpu_nvs_ctrl_fifo_user_subscribe_queue(struct nvs_domain_ctrl_fifo_user *user,
		struct nvgpu_nvs_ctrl_queue *queue);
void nvgpu_nvs_ctrl_fifo_user_unsubscribe_queue(struct nvs_domain_ctrl_fifo_user *user,
		struct nvgpu_nvs_ctrl_queue *queue);
bool nvgpu_nvs_ctrl_fifo_user_is_subscribed_to_queue(struct nvs_domain_ctrl_fifo_user *user,
		struct nvgpu_nvs_ctrl_queue *queue);
void nvgpu_nvs_ctrl_fifo_erase_queue(struct gk20a *g, struct nvgpu_nvs_ctrl_queue *queue);
void nvgpu_nvs_ctrl_fifo_erase_all_queues(struct gk20a *g);
struct nvgpu_nvs_domain *
nvgpu_nvs_get_shadow_domain_locked(struct gk20a *g);
struct nvgpu_nvs_domain *nvgpu_nvs_domain_by_id_locked(struct gk20a *g, u64 domain_id);

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
static inline struct nvgpu_nvs_domain *
nvgpu_nvs_get_shadow_domain_locked(struct gk20a *g)
{
	(void)g;
	return NULL;
}
static inline struct nvgpu_nvs_domain *nvgpu_nvs_domain_by_id_locked(struct gk20a *g, u64 domain_id)
{
	(void)g;
	return NULL;
	(void)domain_id;
}
#endif

#ifdef CONFIG_NVGPU_GSP_SCHEDULER
s32 nvgpu_nvs_gsp_get_runlist_domain_info(struct gk20a *g, u64 nvgpu_domain_id, u32 *num_entries,
	u64 *runlist_iova, u32 *aperture, u32 index);
s32 nvgpu_nvs_get_gsp_domain_info(struct gk20a *g, u64 nvgpu_domain_id,
		u32 *domain_id, u32 *timeslice_ns);
#endif
#endif
