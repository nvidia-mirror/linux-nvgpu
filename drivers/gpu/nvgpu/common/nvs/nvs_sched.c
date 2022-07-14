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

#include <nvs/log.h>
#include <nvs/sched.h>
#include <nvgpu/types.h>

#include <nvgpu/nvs.h>
#include <nvgpu/kmem.h>
#include <nvgpu/gk20a.h>
#include <nvgpu/runlist.h>

static struct nvs_sched_ops nvgpu_nvs_ops = {
	.preempt = NULL,
	.recover = NULL,
};

/*
 * TODO: make use of worker items when
 * 1) the active domain gets modified
 *    - currently updates happen asynchronously elsewhere
 *    - either resubmit the domain or do the updates later
 * 2) recovery gets triggered
 *    - currently it just locks all affected runlists
 *    - consider pausing the scheduler logic and signaling users
 */

struct nvgpu_nvs_worker_item {
	struct gk20a *g;
	struct nvgpu_runlist *rl;
	struct nvgpu_runlist_domain *rl_domain;
	struct nvgpu_cond cond;
	bool swap_buffer;
	bool wait_for_finish;
	bool locked;
	int status;
	struct nvgpu_list_node list;
	nvgpu_atomic_t state;
};


static struct nvgpu_nvs_domain *
nvgpu_nvs_domain_by_id_locked(struct gk20a *g, u64 domain_id);

static inline struct nvgpu_nvs_worker_item *
nvgpu_nvs_worker_item_from_worker_item(struct nvgpu_list_node *node)
{
	return (struct nvgpu_nvs_worker_item *)
	   ((uintptr_t)node - offsetof(struct nvgpu_nvs_worker_item, list));
};

static inline struct nvgpu_nvs_worker *
nvgpu_nvs_worker_from_worker(struct nvgpu_worker *worker)
{
	return (struct nvgpu_nvs_worker *)
	   ((uintptr_t)worker - offsetof(struct nvgpu_nvs_worker, worker));
};

static void nvgpu_nvs_worker_poll_init(struct nvgpu_worker *worker)
{
	struct nvgpu_nvs_worker *nvs_worker =
		nvgpu_nvs_worker_from_worker(worker);

	/* 100 ms is a nice arbitrary timeout for default status */
	nvs_worker->current_timeout = 100;
	nvgpu_timeout_init_cpu_timer_sw(worker->g, &nvs_worker->timeout,
			nvs_worker->current_timeout);

	nvgpu_atomic_set(&nvs_worker->nvs_sched_init, 1);
	nvgpu_cond_signal(&nvs_worker->worker.wq);
}

static u32 nvgpu_nvs_worker_wakeup_timeout(struct nvgpu_worker *worker)
{
	struct nvgpu_nvs_worker *nvs_worker =
		nvgpu_nvs_worker_from_worker(worker);

	return nvs_worker->current_timeout;
}

static u64 nvgpu_nvs_tick(struct gk20a *g)
{
	struct nvgpu_nvs_scheduler *sched = g->scheduler;
	struct nvgpu_nvs_domain *domain;
	struct nvs_domain *nvs_next;
	struct nvgpu_nvs_domain *nvgpu_domain_next;
	u64 timeslice;

	nvs_dbg(g, "nvs tick");

	nvgpu_mutex_acquire(&g->sched_mutex);

	domain = sched->active_domain;

	/* If active_domain == shadow_domain, then nvs_next is NULL */
	nvs_next = nvs_domain_get_next_domain(sched->sched, domain->parent);
	if (nvs_next == NULL) {
		nvs_next = sched->shadow_domain->parent;
	}

	timeslice = nvs_next->timeslice_ns;
	nvgpu_domain_next = nvs_next->priv;

	nvgpu_runlist_tick(g, nvgpu_domain_next->rl_domains);
	sched->active_domain = nvs_next->priv;

	nvgpu_mutex_release(&g->sched_mutex);

	return timeslice;
}

static void nvgpu_nvs_worker_wakeup_process_item(struct nvgpu_list_node *work_item)
{
	struct nvgpu_nvs_worker_item *work =
			nvgpu_nvs_worker_item_from_worker_item(work_item);
	struct gk20a *g = work->g;
	int ret = 0;
	struct nvgpu_nvs_scheduler *sched = g->scheduler;
	struct nvgpu_nvs_domain *nvgpu_nvs_domain;
	struct nvs_domain *nvs_domain;
	struct nvgpu_runlist *runlist = work->rl;
	struct nvgpu_runlist_domain *rl_domain = work->rl_domain;

	nvgpu_mutex_acquire(&g->sched_mutex);

	if (rl_domain == NULL) {
		nvs_domain = sched->shadow_domain->parent;
		rl_domain = runlist->shadow_rl_domain;
	} else if (rl_domain->domain_id == SHADOW_DOMAIN_ID) {
		nvs_domain = sched->shadow_domain->parent;
	} else {
		nvgpu_nvs_domain = nvgpu_nvs_domain_by_id_locked(g, rl_domain->domain_id);
		if (nvgpu_nvs_domain == NULL) {
			nvgpu_err(g, "Unable to find domain[%llu]", rl_domain->domain_id);
			ret = -EINVAL;
			goto done;
		} else {
			nvs_domain = nvgpu_nvs_domain->parent;
		}
	}

	if (sched->active_domain == nvs_domain->priv) {
		/* Instantly switch domain and force runlist updates */
		ret = nvgpu_rl_domain_sync_submit(g, runlist, rl_domain, work->swap_buffer, work->wait_for_finish);
	} else {
		/* Swap buffers here even if its deferred for correctness */
		if (work->swap_buffer) {
			nvgpu_runlist_swap_mem(g, rl_domain);
		}
		ret = 1;
	}

	nvs_dbg(g, " ");

done:
	nvgpu_mutex_release(&g->sched_mutex);
	work->status = ret;
	nvgpu_atomic_set(&work->state, 1);
	/* Wakeup threads waiting on runlist submit */
	nvgpu_cond_signal(&work->cond);
}

static int nvgpu_nvs_worker_submit(struct gk20a *g, struct nvgpu_runlist *rl,
		struct nvgpu_runlist_domain *rl_domain, bool swap_buffer,
		bool wait_for_finish)
{
	struct nvgpu_nvs_scheduler *sched = g->scheduler;
	struct nvgpu_nvs_worker *worker = &sched->worker;
	struct nvgpu_nvs_worker_item *work;
	int ret = 0;

	if (sched == NULL) {
		return -ENODEV;
	}

	nvs_dbg(g, " ");

	work = nvgpu_kzalloc(g, sizeof(*work));
	if (work == NULL) {
		nvgpu_err(g, "Unable to allocate memory for runlist job");
		ret = -ENOMEM;
		goto free_domain;
	}

	work->g = g;
	work->rl = rl;
	work->rl_domain = rl_domain;
	nvgpu_cond_init(&work->cond);
	nvgpu_init_list_node(&work->list);
	work->swap_buffer = swap_buffer;
	work->wait_for_finish = wait_for_finish;
	nvgpu_atomic_set(&work->state, 0);

	nvs_dbg(g, " enqueueing runlist submit");

	ret = nvgpu_worker_enqueue(&worker->worker, &work->list);
	if (ret != 0) {
		goto fail;
	}

	nvs_dbg(g, " ");

	ret = NVGPU_COND_WAIT(&work->cond, nvgpu_atomic_read(&work->state) == 1, 0U);
	if (ret != 0) {
		nvgpu_err(g, "Runlist submit interrupted while waiting for submit");
		goto fail;
	}

	nvs_dbg(g, " ");

	ret = work->status;

fail:
	nvgpu_cond_destroy(&work->cond);
	nvgpu_kfree(g, work);

free_domain:

	return ret;
}

static void nvgpu_nvs_worker_wakeup_post_process(struct nvgpu_worker *worker)
{
	struct gk20a *g = worker->g;
	struct nvgpu_nvs_worker *nvs_worker =
		nvgpu_nvs_worker_from_worker(worker);

	if (nvgpu_timeout_peek_expired(&nvs_worker->timeout)) {
		u32 next_timeout_ns = nvgpu_nvs_tick(g);

		if (next_timeout_ns != 0U) {
			nvs_worker->current_timeout =
				(next_timeout_ns + NSEC_PER_MSEC - 1) / NSEC_PER_MSEC;
		}

		nvgpu_timeout_init_cpu_timer_sw(g, &nvs_worker->timeout,
				nvs_worker->current_timeout);
	}
}

static const struct nvgpu_worker_ops nvs_worker_ops = {
	.pre_process = nvgpu_nvs_worker_poll_init,
	.wakeup_timeout = nvgpu_nvs_worker_wakeup_timeout,
	.wakeup_process_item = nvgpu_nvs_worker_wakeup_process_item,
	.wakeup_post_process = nvgpu_nvs_worker_wakeup_post_process,
};

static int nvgpu_nvs_worker_init(struct gk20a *g)
{
	int err = 0;
	struct nvgpu_worker *worker = &g->scheduler->worker.worker;
	struct nvgpu_nvs_worker *nvs_worker = &g->scheduler->worker;

	nvgpu_cond_init(&nvs_worker->wq_init);
	nvgpu_atomic_set(&nvs_worker->nvs_sched_init, 0);

	nvgpu_worker_init_name(worker, "nvgpu_nvs", g->name);

	err = nvgpu_worker_init(g, worker, &nvs_worker_ops);
	if (err != 0) {
		/* Ensure that scheduler thread is started as soon as possible to handle
		 * minimal uptime for applications.
		 */
		err = NVGPU_COND_WAIT(&nvs_worker->worker.wq,
				nvgpu_atomic_read(&nvs_worker->nvs_sched_init) == 1, 0);
		if (err != 0) {
			nvgpu_err(g, "Interrupted while waiting for scheduler thread");
		}
	}

	return err;
}

static void nvgpu_nvs_worker_deinit(struct gk20a *g)
{
	struct nvgpu_worker *worker = &g->scheduler->worker.worker;
	struct nvgpu_nvs_worker *nvs_worker = &g->scheduler->worker;

	nvgpu_worker_deinit(worker);

	nvgpu_atomic_set(&nvs_worker->nvs_sched_init, 0);
	nvgpu_cond_destroy(&nvs_worker->wq_init);

	nvs_dbg(g, "NVS worker suspended");
}

static struct nvgpu_nvs_domain *
	nvgpu_nvs_gen_domain(struct gk20a *g, const char *name, u64 id,
		u64 timeslice, u64 preempt_grace)
{
	struct nvgpu_fifo *f = &g->fifo;
	struct nvs_domain *nvs_dom = NULL;
	struct nvgpu_nvs_domain *nvgpu_dom = NULL;
	u32 num_runlists = f->num_runlists;

	nvs_dbg(g, "Adding new domain: %s", name);

	nvgpu_dom = nvgpu_kzalloc(g, sizeof(*nvgpu_dom));
	if (nvgpu_dom == NULL) {
		nvs_dbg(g, "failed to allocate memory for domain %s", name);
		return nvgpu_dom;
	}

	nvgpu_dom->rl_domains = nvgpu_kzalloc(g, sizeof(*nvgpu_dom->rl_domains) * num_runlists);
	if (nvgpu_dom->rl_domains == NULL) {
		nvs_dbg(g, "failed to allocate memory for domain->rl_domains");
		nvgpu_kfree(g, nvgpu_dom);
		nvgpu_dom = NULL;
		return nvgpu_dom;
	}

	nvgpu_dom->id = id;
	nvgpu_dom->ref = 1U;

	nvs_dom = nvs_domain_create(g->scheduler->sched, name,
				timeslice, preempt_grace, nvgpu_dom);

	if (nvs_dom == NULL) {
		nvs_dbg(g, "failed to create nvs domain for %s", name);
		nvgpu_kfree(g, nvgpu_dom->rl_domains);
		nvgpu_kfree(g, nvgpu_dom);
		nvgpu_dom = NULL;
		return nvgpu_dom;
	}

	nvgpu_dom->parent = nvs_dom;

	return nvgpu_dom;
}

static void nvgpu_nvs_link_shadow_rl_domains(struct gk20a *g,
		struct nvgpu_nvs_domain *nvgpu_dom)
{
	struct nvgpu_fifo *f = &g->fifo;
	u32 num_runlists = f->num_runlists;
	u32 i;

	for (i = 0U; i < num_runlists; i++) {
		struct nvgpu_runlist *runlist = &f->active_runlists[i];
		nvgpu_dom->rl_domains[i] = runlist->shadow_rl_domain;
	}
}

static int nvgpu_nvs_gen_shadow_domain(struct gk20a *g)
{
	int err = 0;
	struct nvgpu_nvs_domain *nvgpu_dom;

	if (g->scheduler->shadow_domain != NULL) {
		goto error;
	}

	nvgpu_dom = nvgpu_nvs_gen_domain(g, SHADOW_DOMAIN_NAME, U64_MAX,
		100U * NSEC_PER_MSEC, 0U);
	if (nvgpu_dom == NULL) {
		err = -ENOMEM;
		goto error;
	}

	nvgpu_nvs_link_shadow_rl_domains(g, nvgpu_dom);

	g->scheduler->shadow_domain = nvgpu_dom;

	/* Set active_domain to shadow_domain during Init */
	g->scheduler->active_domain = g->scheduler->shadow_domain;

error:
	return err;
}

static void nvgpu_nvs_remove_shadow_domain(struct gk20a *g)
{
	struct nvgpu_nvs_scheduler *sched = g->scheduler;
	struct nvs_domain *nvs_dom;

	if (sched == NULL) {
		/* never powered on to init anything */
		return;
	}

	if (sched->shadow_domain == NULL) {
		return;
	}

	if (sched->shadow_domain->ref != 1U) {
		nvgpu_warn(g,
				"domain %llu is still in use during shutdown! refs: %u",
				sched->shadow_domain->id, sched->shadow_domain->ref);
	}

	nvs_dom = sched->shadow_domain->parent;
	nvs_domain_destroy(sched->sched, nvs_dom);

	nvgpu_kfree(g, sched->shadow_domain->rl_domains);
	sched->shadow_domain->rl_domains = NULL;
	nvgpu_kfree(g, sched->shadow_domain);
	sched->shadow_domain = NULL;
}

int nvgpu_nvs_init(struct gk20a *g)
{
	int err;

	nvgpu_mutex_init(&g->sched_mutex);

	err = nvgpu_nvs_open(g);
	if (err != 0) {
		return err;
	}

	return 0;
}

void nvgpu_nvs_remove_support(struct gk20a *g)
{
	struct nvgpu_nvs_scheduler *sched = g->scheduler;
	struct nvs_domain *nvs_dom;

	if (sched == NULL) {
		/* never powered on to init anything */
		return;
	}

	nvgpu_nvs_worker_deinit(g);

	nvs_domain_for_each(sched->sched, nvs_dom) {
		struct nvgpu_nvs_domain *nvgpu_dom = nvs_dom->priv;
		if (nvgpu_dom->ref != 1U) {
			nvgpu_warn(g,
				   "domain %llu is still in use during shutdown! refs: %u",
				   nvgpu_dom->id, nvgpu_dom->ref);
		}

		/* runlist removal will clear the rl domains */
		nvgpu_kfree(g, nvgpu_dom);
	}

	nvgpu_nvs_remove_shadow_domain(g);

	nvs_sched_close(sched->sched);
	nvgpu_kfree(g, sched->sched);
	nvgpu_kfree(g, sched);
	g->scheduler = NULL;

	nvgpu_nvs_ctrl_fifo_destroy(g);

	nvgpu_mutex_destroy(&g->sched_mutex);
}

int nvgpu_nvs_open(struct gk20a *g)
{
	int err = 0;

	nvs_dbg(g, "Opening NVS node.");

	nvgpu_mutex_acquire(&g->sched_mutex);

	if (g->scheduler != NULL) {
		/* resuming from railgate */
		goto unlock;
	}

	g->scheduler = nvgpu_kzalloc(g, sizeof(*g->scheduler));
	if (g->scheduler == NULL) {
		err = -ENOMEM;
		goto unlock;
	}

	if (nvgpu_is_enabled(g, NVGPU_SUPPORT_NVS_CTRL_FIFO)) {
		g->sched_ctrl_fifo = nvgpu_nvs_ctrl_fifo_create(g);
		if (g->sched_ctrl_fifo == NULL) {
			err = -ENOMEM;
			goto unlock;
		}
	}

	/* separately allocated to keep the definition hidden from other files */
	g->scheduler->sched = nvgpu_kzalloc(g, sizeof(*g->scheduler->sched));
	if (g->scheduler->sched == NULL) {
		err = -ENOMEM;
		goto unlock;
	}

	nvs_dbg(g, "  Creating NVS scheduler.");
	err = nvs_sched_create(g->scheduler->sched, &nvgpu_nvs_ops, g);
	if (err != 0) {
		goto unlock;
	}

	err = nvgpu_nvs_gen_shadow_domain(g);
	if (err != 0) {
		goto unlock;
	}

	err = nvgpu_nvs_worker_init(g);
	if (err != 0) {
		nvgpu_nvs_remove_shadow_domain(g);
		goto unlock;
	}

	g->nvs_worker_submit = nvgpu_nvs_worker_submit;
unlock:
	if (err) {
		nvs_dbg(g, "  Failed! Error code: %d", err);
		if (g->scheduler) {
			nvgpu_kfree(g, g->scheduler->sched);
			nvgpu_kfree(g, g->scheduler);
			g->scheduler = NULL;
		}
		if (g->sched_ctrl_fifo)
			nvgpu_nvs_ctrl_fifo_destroy(g);
	}

	nvgpu_mutex_release(&g->sched_mutex);

	return err;
}

/*
 * A trivial allocator for now.
 */
static u64 nvgpu_nvs_new_id(struct gk20a *g)
{
	return nvgpu_atomic64_inc_return(&g->scheduler->id_counter);
}

static int nvgpu_nvs_create_rl_domain_mem(struct gk20a *g,
		struct nvgpu_nvs_domain *domain)
{
	struct nvgpu_fifo *f = &g->fifo;
	u32 i, j;
	int err = 0;

	for (i = 0U; i < f->num_runlists; i++) {
		domain->rl_domains[i] = nvgpu_runlist_domain_alloc(g, domain->id);
		if (domain->rl_domains[i] == NULL) {
			err = -ENOMEM;
			break;
		}
	}

	if (err != 0) {
		for (j = 0; j != i; j++) {
			nvgpu_runlist_domain_free(g, domain->rl_domains[j]);
			domain->rl_domains[j] = NULL;
		}
	}

	return err;
}

static void nvgpu_nvs_link_rl_domains(struct gk20a *g,
		struct nvgpu_nvs_domain *domain)
{
	struct nvgpu_fifo *f = &g->fifo;
	u32 i;

	for (i = 0U; i < f->num_runlists; i++) {
		struct nvgpu_runlist *runlist;

		runlist = &f->active_runlists[i];
		nvgpu_runlist_link_domain(runlist, domain->rl_domains[i]);
	}
}

int nvgpu_nvs_add_domain(struct gk20a *g, const char *name, u64 timeslice,
			 u64 preempt_grace, struct nvgpu_nvs_domain **pdomain)
{
	int err = 0;
	struct nvs_domain *nvs_dom;
	struct nvgpu_nvs_domain *nvgpu_dom;
	struct nvgpu_nvs_scheduler *sched = g->scheduler;

	nvgpu_mutex_acquire(&g->sched_mutex);

	if (nvs_domain_by_name(g->scheduler->sched, name) != NULL) {
		err = -EEXIST;
		goto unlock;
	}

	nvgpu_dom = nvgpu_nvs_gen_domain(g, name, nvgpu_nvs_new_id(g),
		timeslice, preempt_grace);
	if (nvgpu_dom == NULL) {
		err = -ENOMEM;
		goto unlock;
	}

	err = nvgpu_nvs_create_rl_domain_mem(g, nvgpu_dom);
	if (err != 0) {
		nvs_domain_destroy(sched->sched, nvgpu_dom->parent);
		nvgpu_kfree(g, nvgpu_dom->rl_domains);
		nvgpu_kfree(g, nvgpu_dom);
		goto unlock;
	}

	nvgpu_nvs_link_rl_domains(g, nvgpu_dom);

	nvs_dom = nvgpu_dom->parent;

	nvs_domain_scheduler_attach(g->scheduler->sched, nvs_dom);

	nvgpu_dom->parent = nvs_dom;

	*pdomain = nvgpu_dom;
unlock:
	nvgpu_mutex_release(&g->sched_mutex);

	return err;
}

static struct nvgpu_nvs_domain *
nvgpu_nvs_domain_by_id_locked(struct gk20a *g, u64 domain_id)
{
	struct nvgpu_nvs_scheduler *sched = g->scheduler;
	struct nvs_domain *nvs_dom;

	nvgpu_log(g, gpu_dbg_nvs, "lookup %llu", domain_id);

	nvs_domain_for_each(sched->sched, nvs_dom) {
		struct nvgpu_nvs_domain *nvgpu_dom = nvs_dom->priv;

		if (nvgpu_dom->id == domain_id) {
			return nvgpu_dom;
		}
	}

	return NULL;
}

struct nvgpu_nvs_domain *
nvgpu_nvs_domain_by_id(struct gk20a *g, u64 domain_id)
{
	struct nvgpu_nvs_domain *dom = NULL;

	nvgpu_log(g, gpu_dbg_nvs, "lookup %llu", domain_id);

	nvgpu_mutex_acquire(&g->sched_mutex);

	dom = nvgpu_nvs_domain_by_id_locked(g, domain_id);
	if (dom == NULL) {
		goto unlock;
	}

	dom->ref++;

unlock:
	nvgpu_mutex_release(&g->sched_mutex);
	return dom;
}

struct nvgpu_nvs_domain *
nvgpu_nvs_domain_by_name(struct gk20a *g, const char *name)
{
	struct nvs_domain *nvs_dom;
	struct nvgpu_nvs_domain *dom = NULL;
	struct nvgpu_nvs_scheduler *sched = g->scheduler;

	nvgpu_log(g, gpu_dbg_nvs, "lookup %s", name);

	nvgpu_mutex_acquire(&g->sched_mutex);

	nvs_dom = nvs_domain_by_name(sched->sched, name);
	if (nvs_dom == NULL) {
		goto unlock;
	}

	dom = nvs_dom->priv;
	dom->ref++;

unlock:
	nvgpu_mutex_release(&g->sched_mutex);
	return dom;
}

void nvgpu_nvs_domain_get(struct gk20a *g, struct nvgpu_nvs_domain *dom)
{
	nvgpu_mutex_acquire(&g->sched_mutex);
	WARN_ON(dom->ref == 0U);
	dom->ref++;
	nvgpu_log(g, gpu_dbg_nvs, "domain %s: ref++ = %u",
			dom->parent->name, dom->ref);
	nvgpu_mutex_release(&g->sched_mutex);
}

void nvgpu_nvs_domain_put(struct gk20a *g, struct nvgpu_nvs_domain *dom)
{
	nvgpu_mutex_acquire(&g->sched_mutex);
	dom->ref--;
	WARN_ON(dom->ref == 0U);
	nvgpu_log(g, gpu_dbg_nvs, "domain %s: ref-- = %u",
			dom->parent->name, dom->ref);
	nvgpu_mutex_release(&g->sched_mutex);
}

static void nvgpu_nvs_delete_rl_domain_mem(struct gk20a *g, struct nvgpu_nvs_domain *dom)
{
	struct nvgpu_fifo *f = &g->fifo;
	u32 i;

	for (i = 0U; i < f->num_runlists; i++) {
		nvgpu_runlist_domain_free(g, dom->rl_domains[i]);
		dom->rl_domains[i] = NULL;
	}
}

static void nvgpu_nvs_unlink_rl_domains(struct gk20a *g, struct nvgpu_nvs_domain *domain)
{
	struct nvgpu_fifo *f = &g->fifo;
	u32 i;

	for (i = 0; i < f->num_runlists; i++) {
		struct nvgpu_runlist *runlist;
		runlist = &f->active_runlists[i];

		nvgpu_runlist_unlink_domain(runlist, domain->rl_domains[i]);
	}
}

int nvgpu_nvs_del_domain(struct gk20a *g, u64 dom_id)
{
	struct nvgpu_nvs_scheduler *s = g->scheduler;
	struct nvgpu_nvs_domain *nvgpu_dom;
	struct nvs_domain *nvs_dom, *nvs_next;
	int err = 0;

	nvgpu_mutex_acquire(&g->sched_mutex);

	nvs_dbg(g, "Attempting to remove domain: %llu", dom_id);

	nvgpu_dom = nvgpu_nvs_domain_by_id_locked(g, dom_id);
	if (nvgpu_dom == NULL) {
		nvs_dbg(g, "domain %llu does not exist!", dom_id);
		err = -ENOENT;
		goto unlock;
	}

	if (nvgpu_dom->ref != 1U) {
		nvs_dbg(g, "domain %llu is still in use! refs: %u",
				dom_id, nvgpu_dom->ref);
		err = -EBUSY;
		goto unlock;
	}

	nvs_dom = nvgpu_dom->parent;

	nvgpu_nvs_unlink_rl_domains(g, nvgpu_dom);
	nvgpu_nvs_delete_rl_domain_mem(g, nvgpu_dom);
	nvgpu_dom->ref = 0U;

	if (s->active_domain == nvgpu_dom) {
		nvs_next = nvs_domain_get_next_domain(s->sched, nvs_dom);
		/* Its the only entry in the list. Set the shadow domain as the active domain */
		if (nvs_next == nvs_dom) {
			nvs_next = s->shadow_domain->parent;
		}
		s->active_domain = nvs_next->priv;
	}

	nvs_domain_unlink_and_destroy(s->sched, nvs_dom);

	nvgpu_kfree(g, nvgpu_dom->rl_domains);
	nvgpu_kfree(g, nvgpu_dom);

unlock:
	nvgpu_mutex_release(&g->sched_mutex);
	return err;
}

u32 nvgpu_nvs_domain_count(struct gk20a *g)
{
	u32 count;

	nvgpu_mutex_acquire(&g->sched_mutex);
	count = nvs_domain_count(g->scheduler->sched);
	nvgpu_mutex_release(&g->sched_mutex);

	return count;
}

const char *nvgpu_nvs_domain_get_name(struct nvgpu_nvs_domain *dom)
{
	struct nvs_domain *nvs_dom = dom->parent;

	return nvs_dom->name;
}

void nvgpu_nvs_get_log(struct gk20a *g, s64 *timestamp, const char **msg)
{
	struct nvs_log_event ev;

	nvs_log_get(g->scheduler->sched, &ev);

	if (ev.event == NVS_EV_NO_EVENT) {
		*timestamp = 0;
		*msg = NULL;
		return;
	}

	*msg       = nvs_log_event_string(ev.event);
	*timestamp = ev.timestamp;
}

void nvgpu_nvs_print_domain(struct gk20a *g, struct nvgpu_nvs_domain *domain)
{
	struct nvs_domain *nvs_dom = domain->parent;

	nvs_dbg(g, "Domain %s", nvs_dom->name);
	nvs_dbg(g, "  timeslice:     %llu ns", nvs_dom->timeslice_ns);
	nvs_dbg(g, "  preempt grace: %llu ns", nvs_dom->preempt_grace_ns);
	nvs_dbg(g, "  domain ID:     %llu", domain->id);
}
