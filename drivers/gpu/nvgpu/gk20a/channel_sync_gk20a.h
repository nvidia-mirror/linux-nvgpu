/*
 * drivers/video/tegra/host/gk20a/channel_sync_gk20a.h
 *
 * GK20A Channel Synchronization Abstraction
 *
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef _GK20A_CHANNEL_SYNC_H_
#define _GK20A_CHANNEL_SYNC_H_

struct gk20a_channel_sync;
struct priv_cmd_entry;
struct channel_gk20a;
struct gk20a_fence;
struct gk20a;

struct gk20a_channel_sync {
	nvgpu_atomic_t refcount;

	/* Generate a gpu wait cmdbuf from syncpoint.
	 * Returns a gpu cmdbuf that performs the wait when executed
	 */
	int (*wait_syncpt)(struct gk20a_channel_sync *s, u32 id, u32 thresh,
			   struct priv_cmd_entry *entry);

	/* Generate a gpu wait cmdbuf from sync fd.
	 * Returns a gpu cmdbuf that performs the wait when executed
	 */
	int (*wait_fd)(struct gk20a_channel_sync *s, int fd,
		       struct priv_cmd_entry *entry);

	/* Increment syncpoint/semaphore.
	 * Returns
	 *  - a gpu cmdbuf that performs the increment when executed,
	 *  - a fence that can be passed to wait_cpu() and is_expired().
	 */
	int (*incr)(struct gk20a_channel_sync *s,
		    struct priv_cmd_entry *entry,
		    struct gk20a_fence *fence,
		    bool need_sync_fence,
		    bool register_irq);

	/* Increment syncpoint/semaphore, preceded by a wfi.
	 * Returns
	 *  - a gpu cmdbuf that performs the increment when executed,
	 *  - a fence that can be passed to wait_cpu() and is_expired().
	 */
	int (*incr_wfi)(struct gk20a_channel_sync *s,
			struct priv_cmd_entry *entry,
			struct gk20a_fence *fence);

	/* Increment syncpoint/semaphore, so that the returned fence represents
	 * work completion (may need wfi) and can be returned to user space.
	 * Returns
	 *  - a gpu cmdbuf that performs the increment when executed,
	 *  - a fence that can be passed to wait_cpu() and is_expired(),
	 *  - a gk20a_fence that signals when the incr has happened.
	 */
	int (*incr_user)(struct gk20a_channel_sync *s,
			 int wait_fence_fd,
			 struct priv_cmd_entry *entry,
			 struct gk20a_fence *fence,
			 bool wfi,
			 bool need_sync_fence,
			 bool register_irq);

	/* Reset the channel syncpoint/semaphore. */
	void (*set_min_eq_max)(struct gk20a_channel_sync *s);

	/* Signals the sync timeline (if owned by the gk20a_channel_sync layer).
	 * This should be called when we notice that a gk20a_fence is
	 * expired. */
	void (*signal_timeline)(struct gk20a_channel_sync *s);

	/* Returns the sync point id or negative number if no syncpt*/
	int (*syncpt_id)(struct gk20a_channel_sync *s);

	/* Returns the sync point address of sync point or 0 if not supported */
	u64 (*syncpt_address)(struct gk20a_channel_sync *s);

	/* Free the resources allocated by gk20a_channel_sync_create. */
	void (*destroy)(struct gk20a_channel_sync *s);
};

void gk20a_channel_sync_destroy(struct gk20a_channel_sync *sync);
struct gk20a_channel_sync *gk20a_channel_sync_create(struct channel_gk20a *c,
	bool user_managed);
bool gk20a_channel_sync_needs_sync_framework(struct gk20a *g);

#endif
