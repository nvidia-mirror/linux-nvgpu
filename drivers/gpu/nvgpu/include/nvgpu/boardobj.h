/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef NVGPU_BOARDOBJ_H
#define NVGPU_BOARDOBJ_H

#include <nvgpu/list.h>
#include <nvgpu/pmu/pmuif/nvgpu_cmdif.h>

struct boardobj;
struct nvgpu_list_node;
struct gk20a;

/*
* check whether the specified BOARDOBJ object implements the queried
* type/class enumeration.
*/
bool boardobj_implements_super(struct gk20a *g, struct boardobj *pboardobj,
					u8 type);

/*
* Fills out the appropriate the nv_pmu_xxxx_device_desc_<xyz> driver->PMU
* description structure, describing this BOARDOBJ board device to the PMU.
*
*/
int boardobj_pmudatainit_super(struct gk20a *g, struct boardobj *pboardobj,
				struct nv_pmu_boardobj *pmudata);

/*
* Constructor for the base Board Object. Called by each device-specific
* implementation of the BOARDOBJ interface to initialize the board object.
*/
int boardobj_construct_super(struct gk20a *g, struct boardobj **ppboardobj,
				size_t size, void *args);

/*
* Destructor for the base board object. Called by each device-Specific
* implementation of the BOARDOBJ interface to destroy the board object.
* This has to be explicitly set by each device that extends from the
* board object.
*/
int boardobj_destruct_super(struct boardobj *pboardobj);

/*
* Base Class for all physical or logical device on the PCB.
* Contains fields common to all devices on the board. Specific types of
* devices may extend this object adding any details specific to that
* device or device-type.
*/

struct boardobj {
	struct gk20a *g;

	u8 type; /*type of the device*/
	u8 idx;  /*index of boardobj within in its group*/
	/* true if allocated in constructor. destructor should free */
	bool allocated;
	u32 type_mask; /*mask of types this boardobjimplements*/
	bool (*implements)(struct gk20a *g, struct boardobj *pboardobj,
			u8 type);
	int (*destruct)(struct boardobj *pboardobj);
	/*
	* Access interface apis which will be overridden by the devices
	* that inherit from BOARDOBJ
	*/
	int (*pmudatainit)(struct gk20a *g, struct boardobj *pboardobj,
			struct nv_pmu_boardobj *pmudata);
	struct nvgpu_list_node node;
};

#define BOARDOBJ_GET_TYPE(pobj) (((struct boardobj *)(pobj))->type)
#define BOARDOBJ_GET_IDX(pobj) (((struct boardobj *)(pobj))->idx)

static inline struct boardobj *
boardobj_from_node(struct nvgpu_list_node *node)
{
	return (struct boardobj *)
		((uintptr_t)node - offsetof(struct boardobj, node));
};

#endif /* NVGPU_BOARDOBJ_H */
