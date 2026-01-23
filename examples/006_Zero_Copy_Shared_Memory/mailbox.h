/**
 * mailbox.h - Reusable Mailbox Interface Library for Raspberry Pi VideoCore IV
 * 
 * This library provides a clean C API for communicating with the VideoCore IV
 * GPU through the Linux mailbox interface (/dev/vcio).
 * 
 * Key Concepts:
 * =============
 * 
 * 1. MAILBOX INTERFACE
 *    The mailbox is a message-passing system between ARM CPU and VideoCore GPU.
 *    Messages are structured property tags that request services like memory
 *    allocation, clock management, and framebuffer configuration.
 * 
 * 2. MEMORY ALIASES (BCM2835/BCM2837)
 *    The GPU sees memory through different "aliases" that control caching:
 * 
 *    | Alias | Bus Address Base | Caching Behavior          |
 *    |-------|------------------|---------------------------|
 *    | 0x0   | 0x00000000       | L1 & L2 cached            |
 *    | 0x4   | 0x40000000       | L2 cached only (coherent) |
 *    | 0x8   | 0x80000000       | L2 cached (allocating)    |
 *    | 0xC   | 0xC0000000       | Direct/Uncached           |
 * 
 *    For Zero-Copy GPU access, we use the 0xC alias (MEM_FLAG_DIRECT) to ensure
 *    the CPU writes go directly to RAM without cache pollution.
 * 
 * 3. ADDRESS TYPES
 *    - Physical Address: ARM's view of memory (0x00000000 - 0x3FFFFFFF)
 *    - Bus Address: GPU's view of memory (with alias prefix)
 *    - Virtual Address: User-space pointer after mmap()
 * 
 * Target: Raspberry Pi 3B (BCM2837, VideoCore IV)
 * Author: HPC_GPGPU Course
 * License: MIT
 */

#ifndef MAILBOX_H
#define MAILBOX_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Constants and Flags
 * ============================================================================ */

/**
 * Memory allocation flags for mem_alloc()
 * 
 * These flags control how the GPU firmware allocates memory and how it
 * appears in the bus address space.
 */

/** Memory is discardable (can be resized to 0 at any time) */
#define MEM_FLAG_DISCARDABLE    (1 << 0)

/** Memory can be locked for normal use without being discarded */
#define MEM_FLAG_NORMAL         (0 << 2)

/** Use direct uncached alias (0xC) - bypasses all caches */
#define MEM_FLAG_DIRECT         (1 << 2)

/** Use coherent alias (0x4) - L2 cached, but coherent with ARM */
#define MEM_FLAG_COHERENT       (2 << 2)

/** Use L1-allocating alias (0x8) - May cause cache eviction */
#define MEM_FLAG_L1_NONALLOCATING (MEM_FLAG_DIRECT | MEM_FLAG_COHERENT)

/** Initialize buffer to all zeros */
#define MEM_FLAG_ZERO           (1 << 4)

/** Don't initialize (default) */
#define MEM_FLAG_NO_INIT        (1 << 5)

/** Hint for locking: no existing kernel mapping */
#define MEM_FLAG_HINT_PERMALOCK (1 << 6)

/**
 * Recommended flags for Zero-Copy GPU memory:
 * - MEM_FLAG_DIRECT: Uncached access (CPU writes go directly to RAM)
 * - MEM_FLAG_ZERO: Clear memory (helps with debugging)
 */
#define MEM_FLAG_ZERO_COPY      (MEM_FLAG_DIRECT | MEM_FLAG_ZERO)

/**
 * Recommended flags for cached CPU memory with GPU access:
 * - MEM_FLAG_COHERENT: L2 cached, ARM-coherent
 */
#define MEM_FLAG_CACHED_COHERENT (MEM_FLAG_COHERENT | MEM_FLAG_ZERO)


/* ============================================================================
 * Mailbox Handle
 * ============================================================================ */

/**
 * Opaque handle for mailbox operations.
 * Actually just a file descriptor, but abstracted for potential future changes.
 */
typedef int mbox_handle_t;

/** Invalid handle sentinel */
#define MBOX_INVALID_HANDLE (-1)


/* ============================================================================
 * GPU Memory Allocation Structure
 * ============================================================================ */

/**
 * Structure representing an allocated GPU memory region.
 * 
 * This encapsulates all the information needed to work with a GPU memory
 * allocation: the handle, addresses, size, and user-space mapping.
 */
typedef struct {
    /** Mailbox handle used for this allocation */
    mbox_handle_t mbox;
    
    /** GPU memory handle (returned by mem_alloc, used for mem_lock/mem_unlock) */
    uint32_t mem_handle;
    
    /** Bus address (GPU's view of memory, with alias prefix) */
    uint32_t bus_addr;
    
    /** Size of allocation in bytes */
    uint32_t size;
    
    /** User-space virtual address (after mmap) */
    void *virt_addr;
    
    /** Flags used for allocation */
    uint32_t flags;
} gpu_mem_t;


/* ============================================================================
 * Mailbox Interface Functions
 * ============================================================================ */

/**
 * @brief Open the mailbox interface.
 * 
 * Opens /dev/vcio for communicating with the VideoCore firmware.
 * 
 * @return Valid handle on success, MBOX_INVALID_HANDLE on failure.
 * 
 * @note Requires read/write access to /dev/vcio (typically root or video group).
 */
mbox_handle_t mbox_open(void);

/**
 * @brief Close the mailbox interface.
 * 
 * @param mbox Handle obtained from mbox_open().
 */
void mbox_close(mbox_handle_t mbox);

/**
 * @brief Send a property message to the GPU.
 * 
 * Low-level function to send raw mailbox property messages.
 * Most users should use the higher-level functions below.
 * 
 * @param mbox Handle obtained from mbox_open().
 * @param buf  Property buffer (must be 16-byte aligned).
 * @return 0 on success, -1 on failure.
 */
int mbox_property(mbox_handle_t mbox, void *buf);


/* ============================================================================
 * Memory Management Functions
 * ============================================================================ */

/**
 * @brief Allocate GPU memory.
 * 
 * Requests the GPU firmware to allocate a block of memory from the GPU's
 * memory pool. The memory is not yet accessible to the CPU.
 * 
 * @param mbox      Mailbox handle.
 * @param size      Size in bytes (will be rounded up to alignment).
 * @param alignment Alignment in bytes (must be power of 2, typically 4096).
 * @param flags     Allocation flags (MEM_FLAG_*).
 * @return Memory handle on success, 0 on failure.
 * 
 * @note The returned handle must be locked with mem_lock() before use.
 */
uint32_t mem_alloc(mbox_handle_t mbox, uint32_t size, uint32_t alignment, 
                   uint32_t flags);

/**
 * @brief Lock GPU memory and get bus address.
 * 
 * Locks a previously allocated memory handle and returns the bus address.
 * The bus address is the GPU's view of the memory location.
 * 
 * @param mbox   Mailbox handle.
 * @param handle Memory handle from mem_alloc().
 * @return Bus address on success, 0 on failure.
 * 
 * @note The bus address includes the alias prefix (e.g., 0xC0000000 for direct).
 */
uint32_t mem_lock(mbox_handle_t mbox, uint32_t handle);

/**
 * @brief Unlock GPU memory.
 * 
 * Unlocks a locked memory handle. The bus address is no longer valid.
 * 
 * @param mbox   Mailbox handle.
 * @param handle Memory handle.
 * @return 0 on success, non-zero on failure.
 */
uint32_t mem_unlock(mbox_handle_t mbox, uint32_t handle);

/**
 * @brief Free GPU memory.
 * 
 * Releases a GPU memory allocation back to the firmware.
 * The memory must be unlocked first.
 * 
 * @param mbox   Mailbox handle.
 * @param handle Memory handle.
 * @return 0 on success, non-zero on failure.
 */
uint32_t mem_free(mbox_handle_t mbox, uint32_t handle);


/* ============================================================================
 * Memory Mapping Functions
 * ============================================================================ */

/**
 * @brief Map physical memory to user space.
 * 
 * Uses /dev/mem to map a physical address range into the calling process's
 * virtual address space.
 * 
 * @param base Physical base address (page-aligned).
 * @param size Size in bytes.
 * @return Virtual address on success, NULL on failure.
 * 
 * @note Requires CAP_SYS_RAWIO or root privileges.
 * @note The caller is responsible for calling unmapmem() to release the mapping.
 */
void *mapmem(uint32_t base, uint32_t size);

/**
 * @brief Map physical memory with cache control.
 * 
 * Extended version of mapmem() that allows specifying caching behavior.
 * 
 * @param base     Physical base address (page-aligned).
 * @param size     Size in bytes.
 * @param uncached If non-zero, map as uncached (O_SYNC).
 * @return Virtual address on success, NULL on failure.
 */
void *mapmem_uncached(uint32_t base, uint32_t size, int uncached);

/**
 * @brief Unmap previously mapped memory.
 * 
 * @param virt_addr Virtual address from mapmem().
 * @param size      Size that was mapped.
 */
void unmapmem(void *virt_addr, uint32_t size);


/* ============================================================================
 * Address Conversion Utilities
 * ============================================================================ */

/**
 * @brief Convert bus address to physical address.
 * 
 * Strips the alias bits from a bus address to get the ARM physical address.
 * 
 * @param bus_addr Bus address (with alias prefix).
 * @return Physical address (ARM view).
 */
static inline uint32_t bus_to_phys(uint32_t bus_addr) {
    return bus_addr & 0x3FFFFFFF;
}

/**
 * @brief Get the alias from a bus address.
 * 
 * @param bus_addr Bus address.
 * @return Alias (0x0, 0x4, 0x8, or 0xC).
 */
static inline uint32_t bus_get_alias(uint32_t bus_addr) {
    return (bus_addr >> 30) & 0x3;
}

/**
 * @brief Construct bus address from physical address and alias.
 * 
 * @param phys_addr Physical address.
 * @param alias     Alias (0-3, representing 0x0, 0x4, 0x8, 0xC).
 * @return Bus address.
 */
static inline uint32_t phys_to_bus(uint32_t phys_addr, uint32_t alias) {
    return (phys_addr & 0x3FFFFFFF) | (alias << 30);
}


/* ============================================================================
 * High-Level GPU Memory API
 * ============================================================================ */

/**
 * @brief Allocate and map GPU memory in one call.
 * 
 * This is the recommended high-level API for most use cases. It:
 * 1. Allocates GPU memory
 * 2. Locks it to get the bus address
 * 3. Maps it to user space
 * 
 * @param mbox      Mailbox handle.
 * @param size      Requested size in bytes.
 * @param alignment Alignment (typically 4096 for page alignment).
 * @param flags     Allocation flags (MEM_FLAG_*).
 * @param[out] mem  Structure to fill with allocation details.
 * @return 0 on success, -1 on failure.
 * 
 * @note Use gpu_mem_free() to release the allocation.
 */
int gpu_mem_alloc(mbox_handle_t mbox, uint32_t size, uint32_t alignment,
                  uint32_t flags, gpu_mem_t *mem);

/**
 * @brief Free a GPU memory allocation.
 * 
 * Unmaps, unlocks, and frees a GPU memory allocation.
 * 
 * @param mem GPU memory structure from gpu_mem_alloc().
 * @return 0 on success, -1 on failure.
 */
int gpu_mem_free(gpu_mem_t *mem);


/* ============================================================================
 * Query Functions
 * ============================================================================ */

/**
 * @brief Get VideoCore firmware version.
 * 
 * @param mbox Mailbox handle.
 * @return Firmware version, or 0 on failure.
 */
uint32_t get_firmware_version(mbox_handle_t mbox);

/**
 * @brief Get ARM memory base and size.
 * 
 * @param mbox    Mailbox handle.
 * @param[out] base  ARM memory base address.
 * @param[out] size  ARM memory size in bytes.
 * @return 0 on success, -1 on failure.
 */
int get_arm_memory(mbox_handle_t mbox, uint32_t *base, uint32_t *size);

/**
 * @brief Get VideoCore (GPU) memory base and size.
 * 
 * @param mbox    Mailbox handle.
 * @param[out] base  GPU memory base address.
 * @param[out] size  GPU memory size in bytes.
 * @return 0 on success, -1 on failure.
 */
int get_vc_memory(mbox_handle_t mbox, uint32_t *base, uint32_t *size);


/* ============================================================================
 * Debug and Information
 * ============================================================================ */

/**
 * @brief Print memory region information.
 * 
 * Useful for debugging - prints all details of a GPU memory allocation.
 * 
 * @param mem GPU memory structure.
 * @param name Optional name/label for the region.
 */
void gpu_mem_print_info(const gpu_mem_t *mem, const char *name);

/**
 * @brief Get human-readable string for memory flags.
 * 
 * @param flags Allocation flags.
 * @param buf   Buffer to write string to.
 * @param len   Buffer length.
 * @return Pointer to buf.
 */
const char *mem_flags_to_string(uint32_t flags, char *buf, size_t len);


#ifdef __cplusplus
}
#endif

#endif /* MAILBOX_H */
