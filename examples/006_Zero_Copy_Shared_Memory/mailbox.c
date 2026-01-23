/**
 * mailbox.c - VideoCore IV Mailbox Interface Implementation
 * 
 * Implementation of the mailbox library for Raspberry Pi GPU communication.
 * 
 * This file implements the Linux /dev/vcio interface for communicating with
 * the VideoCore IV firmware via property tags.
 * 
 * References:
 * - https://github.com/raspberrypi/firmware/wiki/Mailbox-property-interface
 * - https://github.com/hermanhermitage/videocoreiv/wiki/
 * 
 * Target: Raspberry Pi 3B (BCM2837)
 * Author: HPC_GPGPU Course
 * License: MIT
 */

#include "mailbox.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

/* ============================================================================
 * Mailbox IOCTL Definitions
 * ============================================================================ */

/** Major number for the mailbox device */
#define MAJOR_NUM 100

/** IOCTL command for mailbox property interface */
#define IOCTL_MBOX_PROPERTY _IOWR(MAJOR_NUM, 0, char *)

/** Device file for mailbox interface */
#define DEVICE_FILE_NAME "/dev/vcio"

/** Device file for physical memory access */
#define DEV_MEM "/dev/mem"

/** Page size for mmap alignment */
#define PAGE_SIZE 4096


/* ============================================================================
 * Mailbox Property Tags
 * ============================================================================
 * 
 * Property tags are the message types used to communicate with the GPU.
 * Each tag has a specific format and response.
 */

/** End of tag list marker */
#define TAG_END                 0x00000000

/** Get firmware revision */
#define TAG_GET_FIRMWARE_REV    0x00000001

/** Get ARM memory region */
#define TAG_GET_ARM_MEMORY      0x00010005

/** Get VideoCore memory region */
#define TAG_GET_VC_MEMORY       0x00010006

/** Allocate GPU memory */
#define TAG_ALLOCATE_MEMORY     0x0003000C

/** Lock memory (get bus address) */
#define TAG_LOCK_MEMORY         0x0003000D

/** Unlock memory */
#define TAG_UNLOCK_MEMORY       0x0003000E

/** Release memory */
#define TAG_RELEASE_MEMORY      0x0003000F

/** Execute QPU code */
#define TAG_EXECUTE_QPU         0x00030011

/** Enable QPU */
#define TAG_ENABLE_QPU          0x00030012


/* ============================================================================
 * Mailbox Property Buffer Format
 * ============================================================================
 * 
 * The property buffer has a specific format:
 * 
 * Word 0: Buffer size (total bytes)
 * Word 1: Request/Response code
 *         0x00000000 = Request
 *         0x80000000 = Success
 *         0x80000001 = Error
 * Word 2+: Tag list
 *         Tag ID (4 bytes)
 *         Value buffer size (4 bytes)
 *         Request/Response indicator (4 bytes)
 *         Value buffer (variable)
 * End: 0x00000000 (end tag)
 * 
 * Buffer must be 16-byte aligned!
 */

#define REQUEST_CODE    0x00000000
#define RESPONSE_OK     0x80000000
#define RESPONSE_ERROR  0x80000001


/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * @brief Round up to next multiple of alignment.
 */
static inline uint32_t align_up(uint32_t value, uint32_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}


/* ============================================================================
 * Mailbox Interface Functions
 * ============================================================================ */

mbox_handle_t mbox_open(void) {
    int fd = open(DEVICE_FILE_NAME, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot open %s: %s\n", 
                DEVICE_FILE_NAME, strerror(errno));
        fprintf(stderr, "Hint: Try running as root or adding user to 'video' group\n");
        return MBOX_INVALID_HANDLE;
    }
    return fd;
}

void mbox_close(mbox_handle_t mbox) {
    if (mbox != MBOX_INVALID_HANDLE) {
        close(mbox);
    }
}

int mbox_property(mbox_handle_t mbox, void *buf) {
    if (mbox == MBOX_INVALID_HANDLE) {
        fprintf(stderr, "Error: Invalid mailbox handle\n");
        return -1;
    }
    
    int ret = ioctl(mbox, IOCTL_MBOX_PROPERTY, buf);
    if (ret < 0) {
        fprintf(stderr, "Error: Mailbox ioctl failed: %s\n", strerror(errno));
        return -1;
    }
    
    return 0;
}


/* ============================================================================
 * Memory Management Functions
 * ============================================================================ */

uint32_t mem_alloc(mbox_handle_t mbox, uint32_t size, uint32_t alignment, 
                   uint32_t flags) {
    /*
     * Allocate Memory Property Tag:
     * 
     * Tag: 0x0003000C
     * Request:
     *   u32: size (bytes)
     *   u32: alignment (bytes)
     *   u32: flags
     * Response:
     *   u32: handle
     */
    
    /* Buffer must be 16-byte aligned */
    uint32_t buf[32] __attribute__((aligned(16)));
    
    buf[0] = 12 * sizeof(uint32_t);  /* Total buffer size */
    buf[1] = REQUEST_CODE;
    
    /* Tag */
    buf[2] = TAG_ALLOCATE_MEMORY;
    buf[3] = 12;                     /* Value buffer size (bytes) */
    buf[4] = 12;                     /* Request size */
    buf[5] = size;                   /* Size to allocate */
    buf[6] = alignment;              /* Alignment */
    buf[7] = flags;                  /* Flags */
    
    buf[8] = TAG_END;
    
    if (mbox_property(mbox, buf) < 0) {
        return 0;
    }
    
    /* Check response */
    if (buf[1] != RESPONSE_OK) {
        fprintf(stderr, "Error: Memory allocation failed (response: 0x%08X)\n", buf[1]);
        return 0;
    }
    
    /* Handle is returned in buf[5] */
    return buf[5];
}

uint32_t mem_lock(mbox_handle_t mbox, uint32_t handle) {
    /*
     * Lock Memory Property Tag:
     * 
     * Tag: 0x0003000D
     * Request:
     *   u32: handle
     * Response:
     *   u32: bus address
     */
    
    uint32_t buf[32] __attribute__((aligned(16)));
    
    buf[0] = 7 * sizeof(uint32_t);
    buf[1] = REQUEST_CODE;
    
    buf[2] = TAG_LOCK_MEMORY;
    buf[3] = 4;                      /* Value buffer size */
    buf[4] = 4;                      /* Request size */
    buf[5] = handle;
    
    buf[6] = TAG_END;
    
    if (mbox_property(mbox, buf) < 0) {
        return 0;
    }
    
    if (buf[1] != RESPONSE_OK) {
        fprintf(stderr, "Error: Memory lock failed (response: 0x%08X)\n", buf[1]);
        return 0;
    }
    
    return buf[5];  /* Bus address */
}

uint32_t mem_unlock(mbox_handle_t mbox, uint32_t handle) {
    /*
     * Unlock Memory Property Tag:
     * 
     * Tag: 0x0003000E
     * Request:
     *   u32: handle
     * Response:
     *   u32: status (0 = success)
     */
    
    uint32_t buf[32] __attribute__((aligned(16)));
    
    buf[0] = 7 * sizeof(uint32_t);
    buf[1] = REQUEST_CODE;
    
    buf[2] = TAG_UNLOCK_MEMORY;
    buf[3] = 4;
    buf[4] = 4;
    buf[5] = handle;
    
    buf[6] = TAG_END;
    
    if (mbox_property(mbox, buf) < 0) {
        return (uint32_t)-1;
    }
    
    return buf[5];
}

uint32_t mem_free(mbox_handle_t mbox, uint32_t handle) {
    /*
     * Release Memory Property Tag:
     * 
     * Tag: 0x0003000F
     * Request:
     *   u32: handle
     * Response:
     *   u32: status (0 = success)
     */
    
    uint32_t buf[32] __attribute__((aligned(16)));
    
    buf[0] = 7 * sizeof(uint32_t);
    buf[1] = REQUEST_CODE;
    
    buf[2] = TAG_RELEASE_MEMORY;
    buf[3] = 4;
    buf[4] = 4;
    buf[5] = handle;
    
    buf[6] = TAG_END;
    
    if (mbox_property(mbox, buf) < 0) {
        return (uint32_t)-1;
    }
    
    return buf[5];
}


/* ============================================================================
 * Memory Mapping Functions
 * ============================================================================ */

void *mapmem(uint32_t base, uint32_t size) {
    return mapmem_uncached(base, size, 0);
}

void *mapmem_uncached(uint32_t base, uint32_t size, int uncached) {
    /*
     * Map physical memory into user space.
     * 
     * We use /dev/mem which provides direct access to physical memory.
     * This requires CAP_SYS_RAWIO capability or root privileges.
     * 
     * The uncached flag controls whether we use O_SYNC which bypasses
     * the CPU cache for writes.
     */
    
    int flags = O_RDWR;
    if (uncached) {
        flags |= O_SYNC;  /* Uncached/unbuffered access */
    }
    
    int fd = open(DEV_MEM, flags);
    if (fd < 0) {
        fprintf(stderr, "Error: Cannot open %s: %s\n", DEV_MEM, strerror(errno));
        fprintf(stderr, "Hint: Try running as root\n");
        return NULL;
    }
    
    /* Calculate page-aligned base and offset */
    uint32_t page_base = base & ~(PAGE_SIZE - 1);
    uint32_t page_offset = base - page_base;
    uint32_t map_size = align_up(size + page_offset, PAGE_SIZE);
    
    /* Map the memory */
    void *mem = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, 
                     fd, page_base);
    
    close(fd);  /* Can close fd after mmap */
    
    if (mem == MAP_FAILED) {
        fprintf(stderr, "Error: mmap failed for 0x%08X (size %u): %s\n", 
                base, size, strerror(errno));
        return NULL;
    }
    
    /* Return pointer to requested address within the mapped region */
    return (char *)mem + page_offset;
}

void unmapmem(void *virt_addr, uint32_t size) {
    if (virt_addr == NULL) {
        return;
    }
    
    /* Calculate page-aligned address */
    uintptr_t addr = (uintptr_t)virt_addr;
    uintptr_t page_addr = addr & ~(PAGE_SIZE - 1);
    uint32_t page_offset = addr - page_addr;
    uint32_t map_size = align_up(size + page_offset, PAGE_SIZE);
    
    if (munmap((void *)page_addr, map_size) < 0) {
        fprintf(stderr, "Warning: munmap failed: %s\n", strerror(errno));
    }
}


/* ============================================================================
 * High-Level GPU Memory API
 * ============================================================================ */

int gpu_mem_alloc(mbox_handle_t mbox, uint32_t size, uint32_t alignment,
                  uint32_t flags, gpu_mem_t *mem) {
    if (mem == NULL) {
        return -1;
    }
    
    /* Initialize structure */
    memset(mem, 0, sizeof(*mem));
    mem->mbox = mbox;
    
    /* Round up size to alignment */
    size = align_up(size, alignment);
    mem->size = size;
    mem->flags = flags;
    
    /* Step 1: Allocate GPU memory */
    mem->mem_handle = mem_alloc(mbox, size, alignment, flags);
    if (mem->mem_handle == 0) {
        fprintf(stderr, "Error: Failed to allocate %u bytes of GPU memory\n", size);
        return -1;
    }
    
    /* Step 2: Lock to get bus address */
    mem->bus_addr = mem_lock(mbox, mem->mem_handle);
    if (mem->bus_addr == 0) {
        fprintf(stderr, "Error: Failed to lock GPU memory\n");
        mem_free(mbox, mem->mem_handle);
        mem->mem_handle = 0;
        return -1;
    }
    
    /* Step 3: Map to user space */
    /* Convert bus address to physical address */
    uint32_t phys_addr = bus_to_phys(mem->bus_addr);
    
    /* Determine if we should use uncached mapping based on flags */
    int use_uncached = (flags & MEM_FLAG_DIRECT) ? 1 : 0;
    
    mem->virt_addr = mapmem_uncached(phys_addr, size, use_uncached);
    if (mem->virt_addr == NULL) {
        fprintf(stderr, "Error: Failed to map GPU memory to user space\n");
        mem_unlock(mbox, mem->mem_handle);
        mem_free(mbox, mem->mem_handle);
        mem->mem_handle = 0;
        mem->bus_addr = 0;
        return -1;
    }
    
    return 0;
}

int gpu_mem_free(gpu_mem_t *mem) {
    if (mem == NULL) {
        return -1;
    }
    
    int ret = 0;
    
    /* Step 1: Unmap from user space */
    if (mem->virt_addr != NULL) {
        unmapmem(mem->virt_addr, mem->size);
        mem->virt_addr = NULL;
    }
    
    /* Step 2: Unlock */
    if (mem->bus_addr != 0) {
        if (mem_unlock(mem->mbox, mem->mem_handle) != 0) {
            fprintf(stderr, "Warning: Failed to unlock GPU memory\n");
            ret = -1;
        }
        mem->bus_addr = 0;
    }
    
    /* Step 3: Free */
    if (mem->mem_handle != 0) {
        if (mem_free(mem->mbox, mem->mem_handle) != 0) {
            fprintf(stderr, "Warning: Failed to free GPU memory\n");
            ret = -1;
        }
        mem->mem_handle = 0;
    }
    
    return ret;
}


/* ============================================================================
 * Query Functions
 * ============================================================================ */

uint32_t get_firmware_version(mbox_handle_t mbox) {
    uint32_t buf[32] __attribute__((aligned(16)));
    
    buf[0] = 7 * sizeof(uint32_t);
    buf[1] = REQUEST_CODE;
    
    buf[2] = TAG_GET_FIRMWARE_REV;
    buf[3] = 4;
    buf[4] = 0;
    buf[5] = 0;
    
    buf[6] = TAG_END;
    
    if (mbox_property(mbox, buf) < 0) {
        return 0;
    }
    
    return buf[5];
}

int get_arm_memory(mbox_handle_t mbox, uint32_t *base, uint32_t *size) {
    uint32_t buf[32] __attribute__((aligned(16)));
    
    buf[0] = 8 * sizeof(uint32_t);
    buf[1] = REQUEST_CODE;
    
    buf[2] = TAG_GET_ARM_MEMORY;
    buf[3] = 8;
    buf[4] = 0;
    buf[5] = 0;  /* base */
    buf[6] = 0;  /* size */
    
    buf[7] = TAG_END;
    
    if (mbox_property(mbox, buf) < 0) {
        return -1;
    }
    
    if (base) *base = buf[5];
    if (size) *size = buf[6];
    
    return 0;
}

int get_vc_memory(mbox_handle_t mbox, uint32_t *base, uint32_t *size) {
    uint32_t buf[32] __attribute__((aligned(16)));
    
    buf[0] = 8 * sizeof(uint32_t);
    buf[1] = REQUEST_CODE;
    
    buf[2] = TAG_GET_VC_MEMORY;
    buf[3] = 8;
    buf[4] = 0;
    buf[5] = 0;
    buf[6] = 0;
    
    buf[7] = TAG_END;
    
    if (mbox_property(mbox, buf) < 0) {
        return -1;
    }
    
    if (base) *base = buf[5];
    if (size) *size = buf[6];
    
    return 0;
}


/* ============================================================================
 * Debug and Information
 * ============================================================================ */

void gpu_mem_print_info(const gpu_mem_t *mem, const char *name) {
    if (mem == NULL) {
        printf("GPU Memory [%s]: NULL\n", name ? name : "unnamed");
        return;
    }
    
    char flags_str[256];
    mem_flags_to_string(mem->flags, flags_str, sizeof(flags_str));
    
    printf("GPU Memory [%s]:\n", name ? name : "unnamed");
    printf("  Handle:       0x%08X\n", mem->mem_handle);
    printf("  Bus Address:  0x%08X (alias: 0x%X)\n", 
           mem->bus_addr, bus_get_alias(mem->bus_addr));
    printf("  Phys Address: 0x%08X\n", bus_to_phys(mem->bus_addr));
    printf("  Virt Address: %p\n", mem->virt_addr);
    printf("  Size:         %u bytes (%.2f MB)\n", 
           mem->size, (double)mem->size / (1024 * 1024));
    printf("  Flags:        0x%08X (%s)\n", mem->flags, flags_str);
}

const char *mem_flags_to_string(uint32_t flags, char *buf, size_t len) {
    if (buf == NULL || len == 0) {
        return "";
    }
    
    buf[0] = '\0';
    
    if (flags & MEM_FLAG_DISCARDABLE) {
        strncat(buf, "DISCARDABLE ", len - strlen(buf) - 1);
    }
    
    uint32_t alias = (flags >> 2) & 0x3;
    switch (alias) {
        case 0: strncat(buf, "NORMAL ", len - strlen(buf) - 1); break;
        case 1: strncat(buf, "DIRECT ", len - strlen(buf) - 1); break;
        case 2: strncat(buf, "COHERENT ", len - strlen(buf) - 1); break;
        case 3: strncat(buf, "L1_NONALLOC ", len - strlen(buf) - 1); break;
    }
    
    if (flags & MEM_FLAG_ZERO) {
        strncat(buf, "ZERO ", len - strlen(buf) - 1);
    }
    
    if (flags & MEM_FLAG_NO_INIT) {
        strncat(buf, "NO_INIT ", len - strlen(buf) - 1);
    }
    
    if (flags & MEM_FLAG_HINT_PERMALOCK) {
        strncat(buf, "PERMALOCK ", len - strlen(buf) - 1);
    }
    
    /* Remove trailing space */
    size_t slen = strlen(buf);
    if (slen > 0 && buf[slen - 1] == ' ') {
        buf[slen - 1] = '\0';
    }
    
    return buf;
}
