/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

#include <nvgpu/gk20a.h>

#include "perf_tu104.h"

static const u32 hwpm_sys_perfmon_regs[] =
{
	/* This list is autogenerated. Do not edit. */
	0x00240000,
	0x00240004,
	0x00240008,
	0x0024000c,
	0x00240010,
	0x00240014,
	0x00240020,
	0x00240024,
	0x00240028,
	0x0024002c,
	0x00240030,
	0x00240034,
	0x00240040,
	0x00240044,
	0x00240048,
	0x0024004c,
	0x00240050,
	0x00240054,
	0x00240058,
	0x0024005c,
	0x00240060,
	0x00240064,
	0x00240068,
	0x0024006c,
	0x00240070,
	0x00240074,
	0x00240078,
	0x0024007c,
	0x00240080,
	0x00240084,
	0x00240088,
	0x0024008c,
	0x00240090,
	0x00240094,
	0x00240098,
	0x0024009c,
	0x002400a0,
	0x002400a4,
	0x002400a8,
	0x002400ac,
	0x002400b0,
	0x002400b4,
	0x002400b8,
	0x002400bc,
	0x002400c0,
	0x002400c4,
	0x002400c8,
	0x002400cc,
	0x002400d0,
	0x002400d4,
	0x002400d8,
	0x002400dc,
	0x002400e0,
	0x002400e4,
	0x002400e8,
	0x002400ec,
	0x002400f8,
	0x002400fc,
	0x00240104,
	0x00240108,
	0x0024010c,
	0x00240110,
	0x00240120,
	0x00240114,
	0x00240118,
	0x0024011c,
	0x00240124,
	0x00240100,
};

static const u32 hwpm_gpc_perfmon_regs[] =
{
	/* This list is autogenerated. Do not edit. */
	0x00278000,
	0x00278004,
	0x00278008,
	0x0027800c,
	0x00278010,
	0x00278014,
	0x00278020,
	0x00278024,
	0x00278028,
	0x0027802c,
	0x00278030,
	0x00278034,
	0x00278040,
	0x00278044,
	0x00278048,
	0x0027804c,
	0x00278050,
	0x00278054,
	0x00278058,
	0x0027805c,
	0x00278060,
	0x00278064,
	0x00278068,
	0x0027806c,
	0x00278070,
	0x00278074,
	0x00278078,
	0x0027807c,
	0x00278080,
	0x00278084,
	0x00278088,
	0x0027808c,
	0x00278090,
	0x00278094,
	0x00278098,
	0x0027809c,
	0x002780a0,
	0x002780a4,
	0x002780a8,
	0x002780ac,
	0x002780b0,
	0x002780b4,
	0x002780b8,
	0x002780bc,
	0x002780c0,
	0x002780c4,
	0x002780c8,
	0x002780cc,
	0x002780d0,
	0x002780d4,
	0x002780d8,
	0x002780dc,
	0x002780e0,
	0x002780e4,
	0x002780e8,
	0x002780ec,
	0x002780f8,
	0x002780fc,
	0x00278104,
	0x00278108,
	0x0027810c,
	0x00278110,
	0x00278120,
	0x00278114,
	0x00278118,
	0x0027811c,
	0x00278124,
	0x00278100,
};

static const u32 hwpm_fbp_perfmon_regs[] =
{
	/* This list is autogenerated. Do not edit. */
	0x0027c000,
	0x0027c004,
	0x0027c008,
	0x0027c00c,
	0x0027c010,
	0x0027c014,
	0x0027c020,
	0x0027c024,
	0x0027c028,
	0x0027c02c,
	0x0027c030,
	0x0027c034,
	0x0027c040,
	0x0027c044,
	0x0027c048,
	0x0027c04c,
	0x0027c050,
	0x0027c054,
	0x0027c058,
	0x0027c05c,
	0x0027c060,
	0x0027c064,
	0x0027c068,
	0x0027c06c,
	0x0027c070,
	0x0027c074,
	0x0027c078,
	0x0027c07c,
	0x0027c080,
	0x0027c084,
	0x0027c088,
	0x0027c08c,
	0x0027c090,
	0x0027c094,
	0x0027c098,
	0x0027c09c,
	0x0027c0a0,
	0x0027c0a4,
	0x0027c0a8,
	0x0027c0ac,
	0x0027c0b0,
	0x0027c0b4,
	0x0027c0b8,
	0x0027c0bc,
	0x0027c0c0,
	0x0027c0c4,
	0x0027c0c8,
	0x0027c0cc,
	0x0027c0d0,
	0x0027c0d4,
	0x0027c0d8,
	0x0027c0dc,
	0x0027c0e0,
	0x0027c0e4,
	0x0027c0e8,
	0x0027c0ec,
	0x0027c0f8,
	0x0027c0fc,
	0x0027c104,
	0x0027c108,
	0x0027c10c,
	0x0027c110,
	0x0027c120,
	0x0027c114,
	0x0027c118,
	0x0027c11c,
	0x0027c124,
	0x0027c100,
};

const u32 *tu104_perf_get_hwpm_sys_perfmon_regs(u32 *count)
{
	*count = sizeof(hwpm_sys_perfmon_regs) / sizeof(hwpm_sys_perfmon_regs[0]);
	return hwpm_sys_perfmon_regs;
}

const u32 *tu104_perf_get_hwpm_gpc_perfmon_regs(u32 *count)
{
	*count = sizeof(hwpm_gpc_perfmon_regs) / sizeof(hwpm_gpc_perfmon_regs[0]);
	return hwpm_gpc_perfmon_regs;
}

const u32 *tu104_perf_get_hwpm_fbp_perfmon_regs(u32 *count)
{
	*count = sizeof(hwpm_fbp_perfmon_regs) / sizeof(hwpm_fbp_perfmon_regs[0]);
	return hwpm_fbp_perfmon_regs;
}
