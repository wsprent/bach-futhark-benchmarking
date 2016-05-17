#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include <getopt.h>
/* Crash and burn. */

#include <stdarg.h>

static const char *fut_progname;

void panic(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
        fprintf(stderr, "%s: ", fut_progname);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
        exit(eval);
}

/* Some simple utilities for wall-clock timing.

   The function get_wall_time() returns the wall time in microseconds
   (with an unspecified offset).
*/

#ifdef _WIN32

#include <windows.h>

int64_t get_wall_time() {
  LARGE_INTEGER time,freq;
  assert(QueryPerformanceFrequency(&freq));
  assert(QueryPerformanceCounter(&time));
  return ((double)time.QuadPart / freq.QuadPart) * 1000000;
}

#else
/* Assuming POSIX */

#include <time.h>
#include <sys/time.h>

int64_t get_wall_time() {
  struct timeval time;
  assert(gettimeofday(&time,NULL) == 0);
  return time.tv_sec * 1000000 + time.tv_usec;
}

#endif

#define FUT_BLOCK_DIM 16
/* The simple OpenCL runtime framework used by Futhark. */

#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define FUT_KERNEL(s) #s
#define OPENCL_SUCCEED(e) opencl_succeed(e, #e, __FILE__, __LINE__)

static cl_context fut_cl_context;
static cl_command_queue fut_cl_queue;
static const char *cl_preferred_platform = "";
static const char *cl_preferred_device = "";
static int cl_debug = 0;

static size_t cl_group_size = 256;
static size_t cl_num_groups = 128;
static size_t cl_lockstep_width = 1;

struct opencl_device_option {
  cl_platform_id platform;
  cl_device_id device;
  cl_device_type device_type;
  char *platform_name;
  char *device_name;
};

/* This function must be defined by the user.  It is invoked by
   setup_opencl() after the platform and device has been found, but
   before the program is loaded.  Its intended use is to tune
   constants based on the selected platform and device. */
static void post_opencl_setup(struct opencl_device_option*);

static char *strclone(const char *str) {
  size_t size = strlen(str) + 1;
  char *copy = malloc(size);
  if (copy == NULL) {
    return NULL;
  }

  memcpy(copy, str, size);
  return copy;
}

static const char* opencl_error_string(unsigned int err)
{
    switch (err) {
        case CL_SUCCESS:                            return "Success!";
        case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:                   return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
        case CL_MAP_FAILURE:                        return "Map failure";
        case CL_INVALID_VALUE:                      return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
        case CL_INVALID_PLATFORM:                   return "Invalid platform";
        case CL_INVALID_DEVICE:                     return "Invalid device";
        case CL_INVALID_CONTEXT:                    return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
        case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
        case CL_INVALID_SAMPLER:                    return "Invalid sampler";
        case CL_INVALID_BINARY:                     return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
        case CL_INVALID_PROGRAM:                    return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
        case CL_INVALID_KERNEL:                     return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
        case CL_INVALID_EVENT:                      return "Invalid event";
        case CL_INVALID_OPERATION:                  return "Invalid operation";
        case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
        default:                                    return "Unknown";
    }
}

static void opencl_succeed(unsigned int ret,
                    const char *call,
                    const char *file,
                    int line) {
  if (ret != CL_SUCCESS) {
    panic(-1, "%s:%d: OpenCL call\n  %s\nfailed with error code %d (%s)\n",
          file, line, call, ret, opencl_error_string(ret));
  }
}

static char* opencl_platform_info(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED(clGetPlatformInfo(platform, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED(clGetPlatformInfo(platform, param, req_bytes, info, NULL));

  return info;
}

static char* opencl_device_info(cl_device_id device,
                                cl_device_info param) {
  size_t req_bytes;
  char *info;

  OPENCL_SUCCEED(clGetDeviceInfo(device, param, 0, NULL, &req_bytes));

  info = malloc(req_bytes);

  OPENCL_SUCCEED(clGetDeviceInfo(device, param, req_bytes, info, NULL));

  return info;
}

static void opencl_all_device_options(struct opencl_device_option **devices_out,
                                      size_t *num_devices_out) {
  size_t num_devices = 0, num_devices_added = 0;

  cl_platform_id *all_platforms;
  cl_uint *platform_num_devices;

  cl_uint num_platforms;

  // Find the number of platforms.
  OPENCL_SUCCEED(clGetPlatformIDs(0, NULL, &num_platforms));

  // Make room for them.
  all_platforms = calloc(num_platforms, sizeof(cl_platform_id));
  platform_num_devices = calloc(num_platforms, sizeof(cl_uint));

  // Fetch all the platforms.
  OPENCL_SUCCEED(clGetPlatformIDs(num_platforms, all_platforms, NULL));

  // Count the number of devices for each platform, as well as the
  // total number of devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    if (clGetDeviceIDs(all_platforms[i], CL_DEVICE_TYPE_ALL,
                       0, NULL, &platform_num_devices[i]) == CL_SUCCESS) {
      num_devices += platform_num_devices[i];
    } else {
      platform_num_devices[i] = 0;
    }
  }

  // Make room for all the device options.
  struct opencl_device_option *devices =
    calloc(num_devices, sizeof(struct opencl_device_option));

  // Loop through the platforms, getting information about their devices.
  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_platform_id platform = all_platforms[i];
    cl_uint num_platform_devices = platform_num_devices[i];

    if (num_platform_devices == 0) {
      continue;
    }

    char *platform_name = opencl_platform_info(platform, CL_PLATFORM_NAME);
    cl_device_id *platform_devices =
      calloc(num_platform_devices, sizeof(cl_device_id));

    // Fetch all the devices.
    OPENCL_SUCCEED(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL,
                                  num_platform_devices, platform_devices, NULL));

    // Loop through the devices, adding them to the devices array.
    for (cl_uint i = 0; i < num_platform_devices; i++) {
      char *device_name = opencl_device_info(platform_devices[i], CL_DEVICE_NAME);
      devices[num_devices_added].platform = platform;
      devices[num_devices_added].device = platform_devices[i];
      OPENCL_SUCCEED(clGetDeviceInfo(platform_devices[i], CL_DEVICE_TYPE,
                                     sizeof(cl_device_type),
                                     &devices[num_devices_added].device_type,
                                     NULL));
      // We don't want the structs to share memory, so copy the platform name.
      // Each device name is already unique.
      devices[num_devices_added].platform_name = strclone(platform_name);
      devices[num_devices_added].device_name = device_name;
      num_devices_added++;
    }
    free(platform_devices);
    free(platform_name);
  }
  free(all_platforms);
  free(platform_num_devices);

  *devices_out = devices;
  *num_devices_out = num_devices;
}

static struct opencl_device_option get_preferred_device() {
  struct opencl_device_option *devices;
  size_t num_devices;

  opencl_all_device_options(&devices, &num_devices);

  for (size_t i = 0; i < num_devices; i++) {
    struct opencl_device_option device = devices[i];
    if (strstr(device.platform_name, cl_preferred_platform) != NULL &&
        strstr(device.device_name, cl_preferred_device) != NULL) {
      // Free all the platform and device names, except the ones we have chosen.
      for (size_t j = 0; j < num_devices; j++) {
        if (j != i) {
          free(devices[j].platform_name);
          free(devices[j].device_name);
        }
      }
      free(devices);
      return device;
    }
  }

  panic(1, "Could not find acceptable OpenCL device.");
}

static void describe_device_option(struct opencl_device_option device) {
  fprintf(stderr, "Using platform: %s\n", device.platform_name);
  fprintf(stderr, "Using device: %s\n", device.device_name);
}

static cl_build_status build_opencl_program(cl_program program, cl_device_id device, const char* options) {
  cl_int ret_val = clBuildProgram(program, 1, &device, options, NULL, NULL);

  // Avoid termination due to CL_BUILD_PROGRAM_FAILURE
  if (ret_val != CL_SUCCESS && ret_val != CL_BUILD_PROGRAM_FAILURE) {
    assert(ret_val == 0);
  }

  cl_build_status build_status;
  ret_val = clGetProgramBuildInfo(program,
                                  device,
                                  CL_PROGRAM_BUILD_STATUS,
                                  sizeof(cl_build_status),
                                  &build_status,
                                  NULL);
  assert(ret_val == 0);

  if (build_status != CL_SUCCESS) {
    char *build_log;
    size_t ret_val_size;
    ret_val = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    assert(ret_val == 0);

    build_log = malloc(ret_val_size+1);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    assert(ret_val == 0);

    // The spec technically does not say whether the build log is zero-terminated, so let's be careful.
    build_log[ret_val_size] = '\0';

    fprintf(stderr, "Build log:\n%s\n", build_log);

    free(build_log);
  }

  return build_status;
}

static cl_program setup_opencl(const char *prelude_src, const char *src) {

  cl_int error;
  cl_platform_id platform;
  cl_device_id device;
  cl_uint platforms, devices;
  size_t max_group_size;

  struct opencl_device_option device_option = get_preferred_device();

  if (cl_debug) {
    describe_device_option(device_option);
  }

  device = device_option.device;
  platform = device_option.platform;

  OPENCL_SUCCEED(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                 sizeof(size_t), &max_group_size, NULL));

  if (max_group_size < cl_group_size) {
    fprintf(stderr, "Warning: Device limits group size to %zu (setting was %zu)\n",
            max_group_size, cl_group_size);
    cl_group_size = max_group_size;
  }

  cl_context_properties properties[] = {
    CL_CONTEXT_PLATFORM,
    (cl_context_properties)platform,
    0
  };
  // Note that nVidia's OpenCL requires the platform property
  fut_cl_context = clCreateContext(properties, 1, &device, NULL, NULL, &error);
  assert(error == 0);

  fut_cl_queue = clCreateCommandQueue(fut_cl_context, device, 0, &error);
  assert(error == 0);

  /* Make sure this function is defined. */
  post_opencl_setup(&device_option);

  // Build the OpenCL program.  First we have to prepend the prelude to the program source.
  size_t prelude_size = strlen(prelude_src);
  size_t program_size = strlen(src);
  size_t src_size = prelude_size + program_size;
  char *fut_opencl_src = malloc(src_size + 1);
  strncpy(fut_opencl_src, prelude_src, src_size);
  strncpy(fut_opencl_src+prelude_size, src, src_size-prelude_size);
  fut_opencl_src[src_size] = '0';

  cl_program prog;
  error = 0;
  const char* src_ptr[] = {fut_opencl_src};
  prog = clCreateProgramWithSource(fut_cl_context, 1, src_ptr, &src_size, &error);
  assert(error == 0);
  char compile_opts[1024];
  snprintf(compile_opts, sizeof(compile_opts), "-DFUT_BLOCK_DIM=%d -DLOCKSTEP_WIDTH=%d", FUT_BLOCK_DIM, cl_lockstep_width);
  OPENCL_SUCCEED(build_opencl_program(prog, device, compile_opts));
  free(fut_opencl_src);

  return prog;
}

static const char fut_opencl_prelude[] =
                  "typedef char int8_t;\ntypedef short int16_t;\ntypedef int int32_t;\ntypedef long int64_t;\ntypedef uchar uint8_t;\ntypedef ushort uint16_t;\ntypedef uint uint32_t;\ntypedef ulong uint64_t;\nstatic inline int8_t add8(int8_t x, int8_t y)\n{\n    return x + y;\n}\nstatic inline int16_t add16(int16_t x, int16_t y)\n{\n    return x + y;\n}\nstatic inline int32_t add32(int32_t x, int32_t y)\n{\n    return x + y;\n}\nstatic inline int64_t add64(int64_t x, int64_t y)\n{\n    return x + y;\n}\nstatic inline int8_t sub8(int8_t x, int8_t y)\n{\n    return x - y;\n}\nstatic inline int16_t sub16(int16_t x, int16_t y)\n{\n    return x - y;\n}\nstatic inline int32_t sub32(int32_t x, int32_t y)\n{\n    return x - y;\n}\nstatic inline int64_t sub64(int64_t x, int64_t y)\n{\n    return x - y;\n}\nstatic inline int8_t mul8(int8_t x, int8_t y)\n{\n    return x * y;\n}\nstatic inline int16_t mul16(int16_t x, int16_t y)\n{\n    return x * y;\n}\nstatic inline int32_t mul32(int32_t x, int32_t y)\n{\n    return x * y;\n}\nstatic inline int64_t mul64(int64_t x, int64_t y)\n{\n    return x * y;\n}\nstatic inline uint8_t udiv8(uint8_t x, uint8_t y)\n{\n    return x / y;\n}\nstatic inline uint16_t udiv16(uint16_t x, uint16_t y)\n{\n    return x / y;\n}\nstatic inline uint32_t udiv32(uint32_t x, uint32_t y)\n{\n    return x / y;\n}\nstatic inline uint64_t udiv64(uint64_t x, uint64_t y)\n{\n    return x / y;\n}\nstatic inline uint8_t umod8(uint8_t x, uint8_t y)\n{\n    return x % y;\n}\nstatic inline uint16_t umod16(uint16_t x, uint16_t y)\n{\n    return x % y;\n}\nstatic inline uint32_t umod32(uint32_t x, uint32_t y)\n{\n    return x % y;\n}\nstatic inline uint64_t umod64(uint64_t x, uint64_t y)\n{\n    return x % y;\n}\nstatic inline int8_t sdiv8(int8_t x, int8_t y)\n{\n    int8_t q = x / y;\n    int8_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int16_t sdiv16(int16_t x, int16_t y)\n{\n    int16_t q = x / y;\n    int16_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int32_t sdiv32(int32_t x, int32_t y)\n{\n    int32_t q = x / y;\n    int32_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int64_t sdiv64(int64_t x, int64_t y)\n{\n    int64_t q = x / y;\n    int64_t r = x % y;\n    \n    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);\n}\nstatic inline int8_t smod8(int8_t x, int8_t y)\n{\n    int8_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int16_t smod16(int16_t x, int16_t y)\n{\n    int16_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int32_t smod32(int32_t x, int32_t y)\n{\n    int32_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int64_t smod64(int64_t x, int64_t y)\n{\n    int64_t r = x % y;\n    \n    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);\n}\nstatic inline int8_t squot8(int8_t x, int8_t y)\n{\n    return x / y;\n}\nstatic inline int16_t squot16(int16_t x, int16_t y)\n{\n    return x / y;\n}\nstatic inline int32_t squot32(int32_t x, int32_t y)\n{\n    return x / y;\n}\nstatic inline int64_t squot64(int64_t x, int64_t y)\n{\n    return x / y;\n}\nstatic inline int8_t srem8(int8_t x, int8_t y)\n{\n    return x % y;\n}\nstatic inline int16_t srem16(int16_t x, int16_t y)\n{\n    return x % y;\n}\nstatic inline int32_t srem32(int32_t x, int32_t y)\n{\n    return x % y;\n}\nstatic inline int64_t srem64(int64_t x, int64_t y)\n{\n    return x % y;\n}\nstatic inline uint8_t shl8(uint8_t x, uint8_t y)\n{\n    return x << y;\n}\nstatic inline uint16_t shl16(uint16_t x, uint16_t y)\n{\n    return x << y;\n}\nstatic inline uint32_t shl32(uint32_t x, uint32_t y)\n{\n    return x << y;\n}\nstatic inline uint64_t shl64(uint64_t x, uint64_t y)\n{\n    return x << y;\n}\nstatic inline uint8_t lshr8(uint8_t x, uint8_t y)\n{\n    return x >> y;\n}\nstatic inline uint16_t lshr16(uint16_t x, uint16_t y)\n{\n    return x >> y;\n}\nstatic inline uint32_t lshr32(uint32_t x, uint32_t y)\n{\n    return x >> y;\n}\nstatic inline uint64_t lshr64(uint64_t x, uint64_t y)\n{\n    return x >> y;\n}\nstatic inline int8_t ashr8(int8_t x, int8_t y)\n{\n    return x >> y;\n}\nstatic inline int16_t ashr16(int16_t x, int16_t y)\n{\n    return x >> y;\n}\nstatic inline int32_t ashr32(int32_t x, int32_t y)\n{\n    return x >> y;\n}\nstatic inline int64_t ashr64(int64_t x, int64_t y)\n{\n    return x >> y;\n}\nstatic inline uint8_t and8(uint8_t x, uint8_t y)\n{\n    return x & y;\n}\nstatic inline uint16_t and16(uint16_t x, uint16_t y)\n{\n    return x & y;\n}\nstatic inline uint32_t and32(uint32_t x, uint32_t y)\n{\n    return x & y;\n}\nstatic inline uint64_t and64(uint64_t x, uint64_t y)\n{\n    return x & y;\n}\nstatic inline uint8_t or8(uint8_t x, uint8_t y)\n{\n    return x | y;\n}\nstatic inline uint16_t or16(uint16_t x, uint16_t y)\n{\n    return x | y;\n}\nstatic inline uint32_t or32(uint32_t x, uint32_t y)\n{\n    return x | y;\n}\nstatic inline uint64_t or64(uint64_t x, uint64_t y)\n{\n    return x | y;\n}\nstatic inline uint8_t xor8(uint8_t x, uint8_t y)\n{\n    return x ^ y;\n}\nstatic inline uint16_t xor16(uint16_t x, uint16_t y)\n{\n    return x ^ y;\n}\nstatic inline uint32_t xor32(uint32_t x, uint32_t y)\n{\n    return x ^ y;\n}\nstatic inline uint64_t xor64(uint64_t x, uint64_t y)\n{\n    return x ^ y;\n}\nstatic inline char ult8(uint8_t x, uint8_t y)\n{\n    return x < y;\n}\nstatic inline char ult16(uint16_t x, uint16_t y)\n{\n    return x < y;\n}\nstatic inline char ult32(uint32_t x, uint32_t y)\n{\n    return x < y;\n}\nstatic inline char ult64(uint64_t x, uint64_t y)\n{\n    return x < y;\n}\nstatic inline char ule8(uint8_t x, uint8_t y)\n{\n    return x <= y;\n}\nstatic inline char ule16(uint16_t x, uint16_t y)\n{\n    return x <= y;\n}\nstatic inline char ule32(uint32_t x, uint32_t y)\n{\n    return x <= y;\n}\nstatic inline char ule64(uint64_t x, uint64_t y)\n{\n    return x <= y;\n}\nstatic inline char slt8(int8_t x, int8_t y)\n{\n    return x < y;\n}\nstatic inline char slt16(int16_t x, int16_t y)\n{\n    return x < y;\n}\nstatic inline char slt32(int32_t x, int32_t y)\n{\n    return x < y;\n}\nstatic inline char slt64(int64_t x, int64_t y)\n{\n    return x < y;\n}\nstatic inline char sle8(int8_t x, int8_t y)\n{\n    return x <= y;\n}\nstatic inline char sle16(int16_t x, int16_t y)\n{\n    return x <= y;\n}\nstatic inline char sle32(int32_t x, int32_t y)\n{\n    return x <= y;\n}\nstatic inline char sle64(int64_t x, int64_t y)\n{\n    return x <= y;\n}\nstatic inline int8_t pow8(int8_t x, int8_t y)\n{\n    int8_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int16_t pow16(int16_t x, int16_t y)\n{\n    int16_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int32_t pow32(int32_t x, int32_t y)\n{\n    int32_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int64_t pow64(int64_t x, int64_t y)\n{\n    int64_t res = 1, rem = y;\n    \n    while (rem != 0) {\n        if (rem & 1)\n            res *= x;\n        rem >>= 1;\n        x *= x;\n    }\n    return res;\n}\nstatic inline int8_t sext_i8_i8(int8_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i8_i16(int8_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i8_i32(int8_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i8_i64(int8_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i16_i8(int16_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i16_i16(int16_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i16_i32(int16_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i16_i64(int16_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i32_i8(int32_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i32_i16(int32_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i32_i32(int32_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i32_i64(int32_t x)\n{\n    return x;\n}\nstatic inline int8_t sext_i64_i8(int64_t x)\n{\n    return x;\n}\nstatic inline int16_t sext_i64_i16(int64_t x)\n{\n    return x;\n}\nstatic inline int32_t sext_i64_i32(int64_t x)\n{\n    return x;\n}\nstatic inline int64_t sext_i64_i64(int64_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i8_i8(uint8_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i8_i16(uint8_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i8_i32(uint8_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i8_i64(uint8_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i16_i8(uint16_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i16_i16(uint16_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i16_i32(uint16_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i16_i64(uint16_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i32_i8(uint32_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i32_i16(uint32_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i32_i32(uint32_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i32_i64(uint32_t x)\n{\n    return x;\n}\nstatic inline uint8_t zext_i64_i8(uint64_t x)\n{\n    return x;\n}\nstatic inline uint16_t zext_i64_i16(uint64_t x)\n{\n    return x;\n}\nstatic inline uint32_t zext_i64_i32(uint64_t x)\n{\n    return x;\n}\nstatic inline uint64_t zext_i64_i64(uint64_t x)\n{\n    return x;\n}\nstatic inline float fdiv32(float x, float y)\n{\n    return x / y;\n}\nstatic inline float fadd32(float x, float y)\n{\n    return x + y;\n}\nstatic inline float fsub32(float x, float y)\n{\n    return x - y;\n}\nstatic inline float fmul32(float x, float y)\n{\n    return x * y;\n}\nstatic inline float fpow32(float x, float y)\n{\n    return pow(x, y);\n}\nstatic inline char cmplt32(float x, float y)\n{\n    return x < y;\n}\nstatic inline char cmple32(float x, float y)\n{\n    return x <= y;\n}\nstatic inline float sitofp_i8_f32(int8_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i16_f32(int16_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i32_f32(int32_t x)\n{\n    return x;\n}\nstatic inline float sitofp_i64_f32(int64_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i8_f32(uint8_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i16_f32(uint16_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i32_f32(uint32_t x)\n{\n    return x;\n}\nstatic inline float uitofp_i64_f32(uint64_t x)\n{\n    return x;\n}\nstatic inline int8_t fptosi_f32_i8(float x)\n{\n    return x;\n}\nstatic inline int16_t fptosi_f32_i16(float x)\n{\n    return x;\n}\nstatic inline int32_t fptosi_f32_i32(float x)\n{\n    return x;\n}\nstatic inline int64_t fptosi_f32_i64(float x)\n{\n    return x;\n}\nstatic inline uint8_t fptoui_f32_i8(float x)\n{\n    return x;\n}\nstatic inline uint16_t fptoui_f32_i16(float x)\n{\n    return x;\n}\nstatic inline uint32_t fptoui_f32_i32(float x)\n{\n    return x;\n}\nstatic inline uint64_t fptoui_f32_i64(float x)\n{\n    return x;\n}\n";
static const char fut_opencl_program[] = FUT_KERNEL(
__kernel void map_kernel_52(__global unsigned char *a_mem_136, int32_t size_37,
                            __global unsigned char *mem_138)
{
    const uint kernel_thread_index_52 = get_global_id(0);
    
    if (kernel_thread_index_52 >= size_37)
        return;
    
    int32_t i_53;
    int32_t binop_param_noncurried_54;
    
    // compute thread index
    {
        i_53 = kernel_thread_index_52;
    }
    // read kernel parameters
    {
        binop_param_noncurried_54 = *(__global int32_t *) &a_mem_136[i_53 * 4];
    }
    
    int32_t res_55 = binop_param_noncurried_54 + 10;
    
    // write kernel result
    {
        *(__global int32_t *) &mem_138[i_53 * 4] = res_55;
    }
}
__kernel void fut_kernel_map_transpose_i32(__global int32_t *odata,
                                           uint odata_offset, __global
                                           int32_t *idata, uint idata_offset,
                                           uint width, uint height,
                                           uint total_size, __local
                                           int32_t *block)
{
    uint x_index;
    uint y_index;
    uint our_array_offset;
    
    // Adjust the input and output arrays with the basic offset.
    odata += odata_offset / sizeof(int32_t);
    idata += idata_offset / sizeof(int32_t);
    // Adjust the input and output arrays for the third dimension.
    our_array_offset = get_global_id(2) * width * height;
    odata += our_array_offset;
    idata += our_array_offset;
    // read the matrix tile into shared memory
    x_index = get_global_id(0);
    y_index = get_global_id(1);
    
    uint index_in = y_index * width + x_index;
    
    if ((x_index < width && y_index < height) && index_in < total_size)
        block[get_local_id(1) * (FUT_BLOCK_DIM + 1) + get_local_id(0)] =
            idata[index_in];
    barrier(CLK_LOCAL_MEM_FENCE);
    // Write the transposed matrix tile to global memory.
    x_index = get_group_id(1) * FUT_BLOCK_DIM + get_local_id(0);
    y_index = get_group_id(0) * FUT_BLOCK_DIM + get_local_id(1);
    
    uint index_out = y_index * height + x_index;
    
    if ((x_index < height && y_index < width) && index_out < total_size)
        odata[index_out] = block[get_local_id(0) * (FUT_BLOCK_DIM + 1) +
                                 get_local_id(1)];
}
__kernel void scan_kernel_62(__local volatile
                             int32_t *restrict binop_param_x_mem_local_aligned_0,
                             int32_t per_thread_elements_61,
                             int32_t group_size_57, int32_t size_37, __global
                             unsigned char *mem_145, int32_t num_threads_58,
                             __global unsigned char *mem_147, __global
                             unsigned char *mem_150)
{
    __local volatile char *restrict binop_param_x_mem_local_182 =
                          binop_param_x_mem_local_aligned_0;
    int32_t local_id_172;
    int32_t group_id_173;
    int32_t wave_size_174;
    int32_t thread_chunk_size_176;
    int32_t skip_waves_175;
    int32_t my_index_62;
    int32_t other_index_63;
    int32_t binop_param_x_42;
    int32_t binop_param_y_43;
    int32_t my_index_177;
    int32_t other_index_178;
    int32_t binop_param_x_179;
    int32_t binop_param_y_180;
    int32_t my_index_64;
    int32_t other_index_65;
    int32_t binop_param_x_66;
    int32_t binop_param_y_67;
    
    local_id_172 = get_local_id(0);
    group_id_173 = get_group_id(0);
    skip_waves_175 = get_global_id(0);
    wave_size_174 = LOCKSTEP_WIDTH;
    my_index_64 = skip_waves_175 * per_thread_elements_61;
    
    int32_t starting_point_185 = skip_waves_175 * per_thread_elements_61;
    int32_t remaining_elements_186 = size_37 - starting_point_185;
    
    if (sle32(remaining_elements_186, 0) || sle32(size_37,
                                                  starting_point_185)) {
        thread_chunk_size_176 = 0;
    } else {
        if (slt32(size_37, (skip_waves_175 + 1) * per_thread_elements_61)) {
            thread_chunk_size_176 = size_37 - skip_waves_175 *
                per_thread_elements_61;
        } else {
            thread_chunk_size_176 = per_thread_elements_61;
        }
    }
    binop_param_x_66 = 0;
    // sequentially scan a chunk
    {
        for (int elements_scanned_184 = 0; elements_scanned_184 <
             thread_chunk_size_176; elements_scanned_184++) {
            binop_param_y_67 = *(__global
                                 int32_t *) &mem_145[(elements_scanned_184 *
                                                      num_threads_58 +
                                                      skip_waves_175) * 4];
            
            int32_t res_68 = binop_param_x_66 + binop_param_y_67;
            
            binop_param_x_66 = res_68;
            *(__global int32_t *) &mem_147[(elements_scanned_184 *
                                            num_threads_58 + skip_waves_175) *
                                           4] = binop_param_x_66;
            my_index_64 += 1;
        }
    }
    *(__local volatile int32_t *) &binop_param_x_mem_local_182[local_id_172 *
                                                               sizeof(int32_t)] =
        binop_param_x_66;
    binop_param_y_43 = *(__local volatile
                         int32_t *) &binop_param_x_mem_local_182[local_id_172 *
                                                                 sizeof(int32_t)];
    // in-wave scan (no barriers needed)
    {
        int32_t skip_threads_187 = 1;
        
        while (slt32(skip_threads_187, wave_size_174)) {
            if (sle32(skip_threads_187, local_id_172 - squot32(local_id_172,
                                                               wave_size_174) *
                      wave_size_174)) {
                // read operands
                {
                    binop_param_x_42 = *(__local volatile
                                         int32_t *) &binop_param_x_mem_local_182[(local_id_172 -
                                                                                  skip_threads_187) *
                                                                                 sizeof(int32_t)];
                }
                // perform operation
                {
                    int32_t res_44 = binop_param_x_42 + binop_param_y_43;
                    
                    binop_param_y_43 = res_44;
                }
                // write result
                {
                    *(__local volatile
                      int32_t *) &binop_param_x_mem_local_182[local_id_172 *
                                                              sizeof(int32_t)] =
                        binop_param_y_43;
                }
            }
            skip_threads_187 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of wave 'i' writes its result to offset 'i'
    {
        if ((local_id_172 - squot32(local_id_172, wave_size_174) *
             wave_size_174) == wave_size_174 - 1) {
            *(__local volatile
              int32_t *) &binop_param_x_mem_local_182[squot32(local_id_172,
                                                              wave_size_174) *
                                                      sizeof(int32_t)] =
                binop_param_y_43;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first wave, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_id_172, wave_size_174) == 0) {
            binop_param_y_180 = *(__local volatile
                                  int32_t *) &binop_param_x_mem_local_182[local_id_172 *
                                                                          sizeof(int32_t)];
            // in-wave scan (no barriers needed)
            {
                int32_t skip_threads_188 = 1;
                
                while (slt32(skip_threads_188, wave_size_174)) {
                    if (sle32(skip_threads_188, local_id_172 -
                              squot32(local_id_172, wave_size_174) *
                              wave_size_174)) {
                        // read operands
                        {
                            binop_param_x_179 = *(__local volatile
                                                  int32_t *) &binop_param_x_mem_local_182[(local_id_172 -
                                                                                           skip_threads_188) *
                                                                                          sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            int32_t res_181 = binop_param_x_179 +
                                    binop_param_y_180;
                            
                            binop_param_y_180 = res_181;
                        }
                        // write result
                        {
                            *(__local volatile
                              int32_t *) &binop_param_x_mem_local_182[local_id_172 *
                                                                      sizeof(int32_t)] =
                                binop_param_y_180;
                        }
                    }
                    skip_threads_188 *= 2;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every wave except the first
    {
        if (!(squot32(local_id_172, wave_size_174) == 0)) {
            // read operands
            {
                binop_param_x_42 = *(__local volatile
                                     int32_t *) &binop_param_x_mem_local_182[(squot32(local_id_172,
                                                                                      wave_size_174) -
                                                                              1) *
                                                                             sizeof(int32_t)];
            }
            // perform operation
            {
                int32_t res_44 = binop_param_x_42 + binop_param_y_43;
                
                binop_param_y_43 = res_44;
            }
        }
    }
    *(__global int32_t *) &mem_150[(group_id_173 * group_size_57 +
                                    local_id_172) * 4] = binop_param_y_43;
}
__kernel void map_kernel_71(int32_t num_groups_56, int32_t group_size_57,
                            int32_t last_in_group_index_73, __global
                            unsigned char *mem_150, __global
                            unsigned char *mem_155)
{
    const uint lasts_map_index_71 = get_global_id(0);
    
    if (lasts_map_index_71 >= num_groups_56)
        return;
    
    int32_t group_id_72;
    
    // compute thread index
    {
        group_id_72 = lasts_map_index_71;
    }
    // read kernel parameters
    { }
    
    char cond_76 = slt32(0, group_id_72);
    int32_t preceding_group_74 = group_id_72 - 1;
    int32_t group_lasts_78;
    
    if (cond_76) {
        int32_t x_75 = *(__global int32_t *) &mem_150[(preceding_group_74 *
                                                       group_size_57 +
                                                       last_in_group_index_73) *
                                                      4];
        
        group_lasts_78 = x_75;
    } else {
        group_lasts_78 = 0;
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_155[group_id_72 * 4] = group_lasts_78;
    }
}
__kernel void scan_kernel_80(__local volatile
                             int32_t *restrict binop_param_x_mem_local_aligned_0,
                             int32_t num_groups_56, __global
                             unsigned char *mem_155, __global
                             unsigned char *mem_157, __global
                             unsigned char *mem_160)
{
    __local volatile char *restrict binop_param_x_mem_local_201 =
                          binop_param_x_mem_local_aligned_0;
    int32_t local_id_191;
    int32_t group_id_192;
    int32_t wave_size_193;
    int32_t thread_chunk_size_195;
    int32_t skip_waves_194;
    int32_t my_index_80;
    int32_t other_index_81;
    int32_t binop_param_x_82;
    int32_t binop_param_y_83;
    int32_t my_index_196;
    int32_t other_index_197;
    int32_t binop_param_x_198;
    int32_t binop_param_y_199;
    int32_t my_index_85;
    int32_t other_index_86;
    int32_t binop_param_x_87;
    int32_t binop_param_y_88;
    
    local_id_191 = get_local_id(0);
    group_id_192 = get_group_id(0);
    skip_waves_194 = get_global_id(0);
    wave_size_193 = LOCKSTEP_WIDTH;
    my_index_85 = skip_waves_194;
    
    int32_t starting_point_204 = skip_waves_194;
    int32_t remaining_elements_205 = num_groups_56 - starting_point_204;
    
    if (sle32(remaining_elements_205, 0) || sle32(num_groups_56,
                                                  starting_point_204)) {
        thread_chunk_size_195 = 0;
    } else {
        if (slt32(num_groups_56, skip_waves_194 + 1)) {
            thread_chunk_size_195 = num_groups_56 - skip_waves_194;
        } else {
            thread_chunk_size_195 = 1;
        }
    }
    binop_param_x_87 = 0;
    // sequentially scan a chunk
    {
        for (int elements_scanned_203 = 0; elements_scanned_203 <
             thread_chunk_size_195; elements_scanned_203++) {
            binop_param_y_88 = *(__global int32_t *) &mem_155[(skip_waves_194 +
                                                               elements_scanned_203) *
                                                              4];
            
            int32_t res_89 = binop_param_x_87 + binop_param_y_88;
            
            binop_param_x_87 = res_89;
            *(__global int32_t *) &mem_157[(skip_waves_194 +
                                            elements_scanned_203) * 4] =
                binop_param_x_87;
            my_index_85 += 1;
        }
    }
    *(__local volatile int32_t *) &binop_param_x_mem_local_201[local_id_191 *
                                                               sizeof(int32_t)] =
        binop_param_x_87;
    binop_param_y_83 = *(__local volatile
                         int32_t *) &binop_param_x_mem_local_201[local_id_191 *
                                                                 sizeof(int32_t)];
    // in-wave scan (no barriers needed)
    {
        int32_t skip_threads_206 = 1;
        
        while (slt32(skip_threads_206, wave_size_193)) {
            if (sle32(skip_threads_206, local_id_191 - squot32(local_id_191,
                                                               wave_size_193) *
                      wave_size_193)) {
                // read operands
                {
                    binop_param_x_82 = *(__local volatile
                                         int32_t *) &binop_param_x_mem_local_201[(local_id_191 -
                                                                                  skip_threads_206) *
                                                                                 sizeof(int32_t)];
                }
                // perform operation
                {
                    int32_t res_84 = binop_param_x_82 + binop_param_y_83;
                    
                    binop_param_y_83 = res_84;
                }
                // write result
                {
                    *(__local volatile
                      int32_t *) &binop_param_x_mem_local_201[local_id_191 *
                                                              sizeof(int32_t)] =
                        binop_param_y_83;
                }
            }
            skip_threads_206 *= 2;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // last thread of wave 'i' writes its result to offset 'i'
    {
        if ((local_id_191 - squot32(local_id_191, wave_size_193) *
             wave_size_193) == wave_size_193 - 1) {
            *(__local volatile
              int32_t *) &binop_param_x_mem_local_201[squot32(local_id_191,
                                                              wave_size_193) *
                                                      sizeof(int32_t)] =
                binop_param_y_83;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // scan the first wave, after which offset 'i' contains carry-in for warp 'i+1'
    {
        if (squot32(local_id_191, wave_size_193) == 0) {
            binop_param_y_199 = *(__local volatile
                                  int32_t *) &binop_param_x_mem_local_201[local_id_191 *
                                                                          sizeof(int32_t)];
            // in-wave scan (no barriers needed)
            {
                int32_t skip_threads_207 = 1;
                
                while (slt32(skip_threads_207, wave_size_193)) {
                    if (sle32(skip_threads_207, local_id_191 -
                              squot32(local_id_191, wave_size_193) *
                              wave_size_193)) {
                        // read operands
                        {
                            binop_param_x_198 = *(__local volatile
                                                  int32_t *) &binop_param_x_mem_local_201[(local_id_191 -
                                                                                           skip_threads_207) *
                                                                                          sizeof(int32_t)];
                        }
                        // perform operation
                        {
                            int32_t res_200 = binop_param_x_198 +
                                    binop_param_y_199;
                            
                            binop_param_y_199 = res_200;
                        }
                        // write result
                        {
                            *(__local volatile
                              int32_t *) &binop_param_x_mem_local_201[local_id_191 *
                                                                      sizeof(int32_t)] =
                                binop_param_y_199;
                        }
                    }
                    skip_threads_207 *= 2;
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    // carry-in for every wave except the first
    {
        if (!(squot32(local_id_191, wave_size_193) == 0)) {
            // read operands
            {
                binop_param_x_82 = *(__local volatile
                                     int32_t *) &binop_param_x_mem_local_201[(squot32(local_id_191,
                                                                                      wave_size_193) -
                                                                              1) *
                                                                             sizeof(int32_t)];
            }
            // perform operation
            {
                int32_t res_84 = binop_param_x_82 + binop_param_y_83;
                
                binop_param_y_83 = res_84;
            }
        }
    }
    *(__global int32_t *) &mem_160[(group_id_192 * num_groups_56 +
                                    local_id_191) * 4] = binop_param_y_83;
}
__kernel void map_kernel_96(__global unsigned char *mem_160,
                            int32_t num_groups_56, int32_t group_size_57,
                            __global unsigned char *mem_150, __global
                            unsigned char *mem_163)
{
    const uint chunk_carry_out_index_96 = get_global_id(0);
    
    if (chunk_carry_out_index_96 >= num_groups_56 * group_size_57)
        return;
    
    int32_t group_id_97;
    int32_t elem_id_98;
    int32_t binop_param_x_93;
    int32_t binop_param_y_94;
    
    // compute thread index
    {
        group_id_97 = squot32(chunk_carry_out_index_96, group_size_57);
        elem_id_98 = chunk_carry_out_index_96 -
            squot32(chunk_carry_out_index_96, group_size_57) * group_size_57;
    }
    // read kernel parameters
    {
        binop_param_x_93 = *(__global int32_t *) &mem_160[group_id_97 * 4];
        binop_param_y_94 = *(__global int32_t *) &mem_150[(group_id_97 *
                                                           group_size_57 +
                                                           elem_id_98) * 4];
    }
    
    int32_t res_95 = binop_param_x_93 + binop_param_y_94;
    
    // write kernel result
    {
        *(__global int32_t *) &mem_163[(group_id_97 * group_size_57 +
                                        elem_id_98) * 4] = res_95;
    }
}
__kernel void map_kernel_104(__global unsigned char *mem_153,
                             int32_t per_thread_elements_61,
                             int32_t group_size_57, int32_t size_37, __global
                             unsigned char *mem_163, __global
                             unsigned char *mem_165)
{
    const uint result_map_index_104 = get_global_id(0);
    
    if (result_map_index_104 >= size_37)
        return;
    
    int32_t j_105;
    int32_t binop_param_y_102;
    
    // compute thread index
    {
        j_105 = result_map_index_104;
    }
    // read kernel parameters
    {
        binop_param_y_102 = *(__global int32_t *) &mem_153[(squot32(j_105,
                                                                    per_thread_elements_61) *
                                                            per_thread_elements_61 +
                                                            (j_105 -
                                                             squot32(j_105,
                                                                     per_thread_elements_61) *
                                                             per_thread_elements_61)) *
                                                           4];
    }
    
    int32_t thread_id_106 = squot32(j_105, per_thread_elements_61);
    char cond_107 = 0 == thread_id_106;
    int32_t carry_in_index_108 = thread_id_106 - 1;
    int32_t new_index_110 = squot32(carry_in_index_108, group_size_57);
    int32_t y_112 = new_index_110 * group_size_57;
    int32_t x_113 = carry_in_index_108 - y_112;
    int32_t final_result_109;
    
    if (cond_107) {
        final_result_109 = binop_param_y_102;
    } else {
        int32_t binop_param_x_101 = *(__global
                                      int32_t *) &mem_163[(new_index_110 *
                                                           group_size_57 +
                                                           x_113) * 4];
        int32_t res_103 = binop_param_x_101 + binop_param_y_102;
        
        final_result_109 = res_103;
    }
    // write kernel result
    {
        *(__global int32_t *) &mem_165[j_105 * 4] = final_result_109;
    }
}
);
static cl_kernel map_kernel_52;
static int map_kernel_52total_runtime = 0;
static int map_kernel_52runs = 0;
static cl_kernel fut_kernel_map_transpose_i32;
static int fut_kernel_map_transpose_i32total_runtime = 0;
static int fut_kernel_map_transpose_i32runs = 0;
static cl_kernel scan_kernel_62;
static int scan_kernel_62total_runtime = 0;
static int scan_kernel_62runs = 0;
static cl_kernel map_kernel_71;
static int map_kernel_71total_runtime = 0;
static int map_kernel_71runs = 0;
static cl_kernel scan_kernel_80;
static int scan_kernel_80total_runtime = 0;
static int scan_kernel_80runs = 0;
static cl_kernel map_kernel_96;
static int map_kernel_96total_runtime = 0;
static int map_kernel_96runs = 0;
static cl_kernel map_kernel_104;
static int map_kernel_104total_runtime = 0;
static int map_kernel_104runs = 0;
void setup_opencl_and_load_kernels()

{
    cl_int error;
    cl_program prog = setup_opencl(fut_opencl_prelude, fut_opencl_program);
    
    {
        map_kernel_52 = clCreateKernel(prog, "map_kernel_52", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_52");
    }
    {
        fut_kernel_map_transpose_i32 = clCreateKernel(prog,
                                                      "fut_kernel_map_transpose_i32",
                                                      &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n",
                    "fut_kernel_map_transpose_i32");
    }
    {
        scan_kernel_62 = clCreateKernel(prog, "scan_kernel_62", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "scan_kernel_62");
    }
    {
        map_kernel_71 = clCreateKernel(prog, "map_kernel_71", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_71");
    }
    {
        scan_kernel_80 = clCreateKernel(prog, "scan_kernel_80", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "scan_kernel_80");
    }
    {
        map_kernel_96 = clCreateKernel(prog, "map_kernel_96", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_96");
    }
    {
        map_kernel_104 = clCreateKernel(prog, "map_kernel_104", &error);
        assert(error == 0);
        if (cl_debug)
            fprintf(stderr, "Created kernel %s.\n", "map_kernel_104");
    }
}
void post_opencl_setup(struct opencl_device_option *option)
{
    if (strcmp(option->platform_name, "NVIDIA CUDA") == 0 &&
        option->device_type == CL_DEVICE_TYPE_GPU) {
        cl_lockstep_width = 32;
        if (cl_debug)
            fprintf(stderr, "Setting lockstep width to: %d\n",
                    cl_lockstep_width);
    }
    if (strcmp(option->platform_name, "AMD Accelerated Parallel Processing") ==
        0 && option->device_type == CL_DEVICE_TYPE_GPU) {
        cl_lockstep_width = 64;
        if (cl_debug)
            fprintf(stderr, "Setting lockstep width to: %d\n",
                    cl_lockstep_width);
    }
}
struct memblock_device {
    int *references;
    cl_mem mem;
} ;
static void memblock_unref_device(struct memblock_device *block)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (*block->references == 0) {
            OPENCL_SUCCEED(clReleaseMemObject(block->mem));
            free(block->references);
            block->references = NULL;
        }
    }
}
static void memblock_alloc_device(struct memblock_device *block, int32_t size)
{
    memblock_unref_device(block);
    
    cl_int clCreateBuffer_succeeded_263;
    
    block->mem = clCreateBuffer(fut_cl_context, CL_MEM_READ_WRITE, size >
                                0 ? size : 1, NULL,
                                &clCreateBuffer_succeeded_263);
    OPENCL_SUCCEED(clCreateBuffer_succeeded_263);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
}
static void memblock_set_device(struct memblock_device *lhs,
                                struct memblock_device *rhs)
{
    memblock_unref_device(lhs);
    (*rhs->references)++;
    *lhs = *rhs;
}
struct memblock_local {
    int *references;
    unsigned char mem;
} ;
static void memblock_unref_local(struct memblock_local *block)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (*block->references == 0) {
            free(block->references);
            block->references = NULL;
        }
    }
}
static void memblock_alloc_local(struct memblock_local *block, int32_t size)
{
    memblock_unref_local(block);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
}
static void memblock_set_local(struct memblock_local *lhs,
                               struct memblock_local *rhs)
{
    memblock_unref_local(lhs);
    (*rhs->references)++;
    *lhs = *rhs;
}
struct memblock {
    int *references;
    char *mem;
} ;
static void memblock_unref(struct memblock *block)
{
    if (block->references != NULL) {
        *block->references -= 1;
        if (*block->references == 0) {
            free(block->mem);
            free(block->references);
            block->references = NULL;
        }
    }
}
static void memblock_alloc(struct memblock *block, int32_t size)
{
    memblock_unref(block);
    block->mem = (char *) malloc(size);
    block->references = (int *) malloc(sizeof(int));
    *block->references = 1;
}
static void memblock_set(struct memblock *lhs, struct memblock *rhs)
{
    memblock_unref(lhs);
    (*rhs->references)++;
    *lhs = *rhs;
}
struct tuple_int32_t_device_mem_int32_t {
    int32_t elem_0;
    struct memblock_device elem_1;
    int32_t elem_2;
} ;
static struct tuple_int32_t_device_mem_int32_t
futhark_main(int32_t a_mem_size_135, struct memblock_device a_mem_136, int32_t size_37);
static inline float futhark_log32(float x)
{
    return log(x);
}
static inline float futhark_sqrt32(float x)
{
    return sqrt(x);
}
static inline float futhark_exp32(float x)
{
    return exp(x);
}
static inline float futhark_cos32(float x)
{
    return cos(x);
}
static inline float futhark_sin32(float x)
{
    return sin(x);
}
static inline float futhark_atan2_32(float x, float y)
{
    return atan2(x, y);
}
static inline char futhark_isnan32(float x)
{
    return isnan(x);
}
static inline char futhark_isinf32(float x)
{
    return isinf(x);
}
static inline double futhark_log64(double x)
{
    return log(x);
}
static inline double futhark_sqrt64(double x)
{
    return sqrt(x);
}
static inline double futhark_exp64(double x)
{
    return exp(x);
}
static inline double futhark_cos64(double x)
{
    return cos(x);
}
static inline double futhark_sin64(double x)
{
    return sin(x);
}
static inline double futhark_atan2_64(double x, double y)
{
    return atan2(x, y);
}
static inline char futhark_isnan64(double x)
{
    return isnan(x);
}
static inline char futhark_isinf64(double x)
{
    return isinf(x);
}
static inline int8_t add8(int8_t x, int8_t y)
{
    return x + y;
}
static inline int16_t add16(int16_t x, int16_t y)
{
    return x + y;
}
static inline int32_t add32(int32_t x, int32_t y)
{
    return x + y;
}
static inline int64_t add64(int64_t x, int64_t y)
{
    return x + y;
}
static inline int8_t sub8(int8_t x, int8_t y)
{
    return x - y;
}
static inline int16_t sub16(int16_t x, int16_t y)
{
    return x - y;
}
static inline int32_t sub32(int32_t x, int32_t y)
{
    return x - y;
}
static inline int64_t sub64(int64_t x, int64_t y)
{
    return x - y;
}
static inline int8_t mul8(int8_t x, int8_t y)
{
    return x * y;
}
static inline int16_t mul16(int16_t x, int16_t y)
{
    return x * y;
}
static inline int32_t mul32(int32_t x, int32_t y)
{
    return x * y;
}
static inline int64_t mul64(int64_t x, int64_t y)
{
    return x * y;
}
static inline uint8_t udiv8(uint8_t x, uint8_t y)
{
    return x / y;
}
static inline uint16_t udiv16(uint16_t x, uint16_t y)
{
    return x / y;
}
static inline uint32_t udiv32(uint32_t x, uint32_t y)
{
    return x / y;
}
static inline uint64_t udiv64(uint64_t x, uint64_t y)
{
    return x / y;
}
static inline uint8_t umod8(uint8_t x, uint8_t y)
{
    return x % y;
}
static inline uint16_t umod16(uint16_t x, uint16_t y)
{
    return x % y;
}
static inline uint32_t umod32(uint32_t x, uint32_t y)
{
    return x % y;
}
static inline uint64_t umod64(uint64_t x, uint64_t y)
{
    return x % y;
}
static inline int8_t sdiv8(int8_t x, int8_t y)
{
    int8_t q = x / y;
    int8_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int16_t sdiv16(int16_t x, int16_t y)
{
    int16_t q = x / y;
    int16_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int32_t sdiv32(int32_t x, int32_t y)
{
    int32_t q = x / y;
    int32_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int64_t sdiv64(int64_t x, int64_t y)
{
    int64_t q = x / y;
    int64_t r = x % y;
    
    return q - ((r != 0 && r < 0 != y < 0) ? 1 : 0);
}
static inline int8_t smod8(int8_t x, int8_t y)
{
    int8_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int16_t smod16(int16_t x, int16_t y)
{
    int16_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int32_t smod32(int32_t x, int32_t y)
{
    int32_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int64_t smod64(int64_t x, int64_t y)
{
    int64_t r = x % y;
    
    return r + (r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) ? 0 : y);
}
static inline int8_t squot8(int8_t x, int8_t y)
{
    return x / y;
}
static inline int16_t squot16(int16_t x, int16_t y)
{
    return x / y;
}
static inline int32_t squot32(int32_t x, int32_t y)
{
    return x / y;
}
static inline int64_t squot64(int64_t x, int64_t y)
{
    return x / y;
}
static inline int8_t srem8(int8_t x, int8_t y)
{
    return x % y;
}
static inline int16_t srem16(int16_t x, int16_t y)
{
    return x % y;
}
static inline int32_t srem32(int32_t x, int32_t y)
{
    return x % y;
}
static inline int64_t srem64(int64_t x, int64_t y)
{
    return x % y;
}
static inline uint8_t shl8(uint8_t x, uint8_t y)
{
    return x << y;
}
static inline uint16_t shl16(uint16_t x, uint16_t y)
{
    return x << y;
}
static inline uint32_t shl32(uint32_t x, uint32_t y)
{
    return x << y;
}
static inline uint64_t shl64(uint64_t x, uint64_t y)
{
    return x << y;
}
static inline uint8_t lshr8(uint8_t x, uint8_t y)
{
    return x >> y;
}
static inline uint16_t lshr16(uint16_t x, uint16_t y)
{
    return x >> y;
}
static inline uint32_t lshr32(uint32_t x, uint32_t y)
{
    return x >> y;
}
static inline uint64_t lshr64(uint64_t x, uint64_t y)
{
    return x >> y;
}
static inline int8_t ashr8(int8_t x, int8_t y)
{
    return x >> y;
}
static inline int16_t ashr16(int16_t x, int16_t y)
{
    return x >> y;
}
static inline int32_t ashr32(int32_t x, int32_t y)
{
    return x >> y;
}
static inline int64_t ashr64(int64_t x, int64_t y)
{
    return x >> y;
}
static inline uint8_t and8(uint8_t x, uint8_t y)
{
    return x & y;
}
static inline uint16_t and16(uint16_t x, uint16_t y)
{
    return x & y;
}
static inline uint32_t and32(uint32_t x, uint32_t y)
{
    return x & y;
}
static inline uint64_t and64(uint64_t x, uint64_t y)
{
    return x & y;
}
static inline uint8_t or8(uint8_t x, uint8_t y)
{
    return x | y;
}
static inline uint16_t or16(uint16_t x, uint16_t y)
{
    return x | y;
}
static inline uint32_t or32(uint32_t x, uint32_t y)
{
    return x | y;
}
static inline uint64_t or64(uint64_t x, uint64_t y)
{
    return x | y;
}
static inline uint8_t xor8(uint8_t x, uint8_t y)
{
    return x ^ y;
}
static inline uint16_t xor16(uint16_t x, uint16_t y)
{
    return x ^ y;
}
static inline uint32_t xor32(uint32_t x, uint32_t y)
{
    return x ^ y;
}
static inline uint64_t xor64(uint64_t x, uint64_t y)
{
    return x ^ y;
}
static inline char ult8(uint8_t x, uint8_t y)
{
    return x < y;
}
static inline char ult16(uint16_t x, uint16_t y)
{
    return x < y;
}
static inline char ult32(uint32_t x, uint32_t y)
{
    return x < y;
}
static inline char ult64(uint64_t x, uint64_t y)
{
    return x < y;
}
static inline char ule8(uint8_t x, uint8_t y)
{
    return x <= y;
}
static inline char ule16(uint16_t x, uint16_t y)
{
    return x <= y;
}
static inline char ule32(uint32_t x, uint32_t y)
{
    return x <= y;
}
static inline char ule64(uint64_t x, uint64_t y)
{
    return x <= y;
}
static inline char slt8(int8_t x, int8_t y)
{
    return x < y;
}
static inline char slt16(int16_t x, int16_t y)
{
    return x < y;
}
static inline char slt32(int32_t x, int32_t y)
{
    return x < y;
}
static inline char slt64(int64_t x, int64_t y)
{
    return x < y;
}
static inline char sle8(int8_t x, int8_t y)
{
    return x <= y;
}
static inline char sle16(int16_t x, int16_t y)
{
    return x <= y;
}
static inline char sle32(int32_t x, int32_t y)
{
    return x <= y;
}
static inline char sle64(int64_t x, int64_t y)
{
    return x <= y;
}
static inline int8_t pow8(int8_t x, int8_t y)
{
    int8_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int16_t pow16(int16_t x, int16_t y)
{
    int16_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int32_t pow32(int32_t x, int32_t y)
{
    int32_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int64_t pow64(int64_t x, int64_t y)
{
    int64_t res = 1, rem = y;
    
    while (rem != 0) {
        if (rem & 1)
            res *= x;
        rem >>= 1;
        x *= x;
    }
    return res;
}
static inline int8_t sext_i8_i8(int8_t x)
{
    return x;
}
static inline int16_t sext_i8_i16(int8_t x)
{
    return x;
}
static inline int32_t sext_i8_i32(int8_t x)
{
    return x;
}
static inline int64_t sext_i8_i64(int8_t x)
{
    return x;
}
static inline int8_t sext_i16_i8(int16_t x)
{
    return x;
}
static inline int16_t sext_i16_i16(int16_t x)
{
    return x;
}
static inline int32_t sext_i16_i32(int16_t x)
{
    return x;
}
static inline int64_t sext_i16_i64(int16_t x)
{
    return x;
}
static inline int8_t sext_i32_i8(int32_t x)
{
    return x;
}
static inline int16_t sext_i32_i16(int32_t x)
{
    return x;
}
static inline int32_t sext_i32_i32(int32_t x)
{
    return x;
}
static inline int64_t sext_i32_i64(int32_t x)
{
    return x;
}
static inline int8_t sext_i64_i8(int64_t x)
{
    return x;
}
static inline int16_t sext_i64_i16(int64_t x)
{
    return x;
}
static inline int32_t sext_i64_i32(int64_t x)
{
    return x;
}
static inline int64_t sext_i64_i64(int64_t x)
{
    return x;
}
static inline uint8_t zext_i8_i8(uint8_t x)
{
    return x;
}
static inline uint16_t zext_i8_i16(uint8_t x)
{
    return x;
}
static inline uint32_t zext_i8_i32(uint8_t x)
{
    return x;
}
static inline uint64_t zext_i8_i64(uint8_t x)
{
    return x;
}
static inline uint8_t zext_i16_i8(uint16_t x)
{
    return x;
}
static inline uint16_t zext_i16_i16(uint16_t x)
{
    return x;
}
static inline uint32_t zext_i16_i32(uint16_t x)
{
    return x;
}
static inline uint64_t zext_i16_i64(uint16_t x)
{
    return x;
}
static inline uint8_t zext_i32_i8(uint32_t x)
{
    return x;
}
static inline uint16_t zext_i32_i16(uint32_t x)
{
    return x;
}
static inline uint32_t zext_i32_i32(uint32_t x)
{
    return x;
}
static inline uint64_t zext_i32_i64(uint32_t x)
{
    return x;
}
static inline uint8_t zext_i64_i8(uint64_t x)
{
    return x;
}
static inline uint16_t zext_i64_i16(uint64_t x)
{
    return x;
}
static inline uint32_t zext_i64_i32(uint64_t x)
{
    return x;
}
static inline uint64_t zext_i64_i64(uint64_t x)
{
    return x;
}
static inline float fdiv32(float x, float y)
{
    return x / y;
}
static inline float fadd32(float x, float y)
{
    return x + y;
}
static inline float fsub32(float x, float y)
{
    return x - y;
}
static inline float fmul32(float x, float y)
{
    return x * y;
}
static inline float fpow32(float x, float y)
{
    return pow(x, y);
}
static inline char cmplt32(float x, float y)
{
    return x < y;
}
static inline char cmple32(float x, float y)
{
    return x <= y;
}
static inline float sitofp_i8_f32(int8_t x)
{
    return x;
}
static inline float sitofp_i16_f32(int16_t x)
{
    return x;
}
static inline float sitofp_i32_f32(int32_t x)
{
    return x;
}
static inline float sitofp_i64_f32(int64_t x)
{
    return x;
}
static inline float uitofp_i8_f32(uint8_t x)
{
    return x;
}
static inline float uitofp_i16_f32(uint16_t x)
{
    return x;
}
static inline float uitofp_i32_f32(uint32_t x)
{
    return x;
}
static inline float uitofp_i64_f32(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f32_i8(float x)
{
    return x;
}
static inline int16_t fptosi_f32_i16(float x)
{
    return x;
}
static inline int32_t fptosi_f32_i32(float x)
{
    return x;
}
static inline int64_t fptosi_f32_i64(float x)
{
    return x;
}
static inline uint8_t fptoui_f32_i8(float x)
{
    return x;
}
static inline uint16_t fptoui_f32_i16(float x)
{
    return x;
}
static inline uint32_t fptoui_f32_i32(float x)
{
    return x;
}
static inline uint64_t fptoui_f32_i64(float x)
{
    return x;
}
static inline double fdiv64(double x, double y)
{
    return x / y;
}
static inline double fadd64(double x, double y)
{
    return x + y;
}
static inline double fsub64(double x, double y)
{
    return x - y;
}
static inline double fmul64(double x, double y)
{
    return x * y;
}
static inline double fpow64(double x, double y)
{
    return pow(x, y);
}
static inline char cmplt64(double x, double y)
{
    return x < y;
}
static inline char cmple64(double x, double y)
{
    return x <= y;
}
static inline double sitofp_i8_f64(int8_t x)
{
    return x;
}
static inline double sitofp_i16_f64(int16_t x)
{
    return x;
}
static inline double sitofp_i32_f64(int32_t x)
{
    return x;
}
static inline double sitofp_i64_f64(int64_t x)
{
    return x;
}
static inline double uitofp_i8_f64(uint8_t x)
{
    return x;
}
static inline double uitofp_i16_f64(uint16_t x)
{
    return x;
}
static inline double uitofp_i32_f64(uint32_t x)
{
    return x;
}
static inline double uitofp_i64_f64(uint64_t x)
{
    return x;
}
static inline int8_t fptosi_f64_i8(double x)
{
    return x;
}
static inline int16_t fptosi_f64_i16(double x)
{
    return x;
}
static inline int32_t fptosi_f64_i32(double x)
{
    return x;
}
static inline int64_t fptosi_f64_i64(double x)
{
    return x;
}
static inline uint8_t fptoui_f64_i8(double x)
{
    return x;
}
static inline uint16_t fptoui_f64_i16(double x)
{
    return x;
}
static inline uint32_t fptoui_f64_i32(double x)
{
    return x;
}
static inline uint64_t fptoui_f64_i64(double x)
{
    return x;
}
static inline float fpconv_f32_f32(float x)
{
    return x;
}
static inline double fpconv_f32_f64(float x)
{
    return x;
}
static inline float fpconv_f64_f32(double x)
{
    return x;
}
static inline double fpconv_f64_f64(double x)
{
    return x;
}
static int detail_timing = 0;
static
struct tuple_int32_t_device_mem_int32_t futhark_main(int32_t a_mem_size_135,
                                                     struct memblock_device a_mem_136,
                                                     int32_t size_37)
{
    int32_t out_memsize_167;
    struct memblock_device out_mem_166;
    
    out_mem_166.references = NULL;
    
    int32_t out_arrsize_168;
    int32_t bytes_137 = 4 * size_37;
    struct memblock_device mem_138;
    
    mem_138.references = NULL;
    memblock_alloc_device(&mem_138, bytes_137);
    
    int32_t group_size_169;
    int32_t num_groups_170;
    
    group_size_169 = cl_group_size;
    num_groups_170 = squot32(size_37 + group_size_169 - 1, group_size_169);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_52, 0, sizeof(a_mem_136.mem),
                                  &a_mem_136.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_52, 1, sizeof(size_37), &size_37));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_52, 2, sizeof(mem_138.mem),
                                  &mem_138.mem));
    if (1 * (num_groups_170 * group_size_169) != 0) {
        const size_t global_work_size_213[1] = {num_groups_170 *
                     group_size_169};
        const size_t local_work_size_217[1] = {group_size_169};
        int64_t time_start_214, time_end_215;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_52");
            fprintf(stderr, "%zu", global_work_size_213[0]);
            fprintf(stderr, "].\n");
            time_start_214 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_52, 1,
                                              NULL, global_work_size_213,
                                              local_work_size_217, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_215 = get_wall_time();
            
            long time_diff_216 = time_end_215 - time_start_214;
            
            if (detail_timing) {
                map_kernel_52total_runtime += time_diff_216;
                map_kernel_52runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_52",
                        (int) time_diff_216);
            }
        }
    }
    
    int32_t num_groups_56;
    
    num_groups_56 = cl_num_groups;
    
    int32_t group_size_57;
    
    group_size_57 = cl_group_size;
    
    int32_t num_threads_58 = num_groups_56 * group_size_57;
    int32_t y_59 = num_threads_58 - 1;
    int32_t x_60 = size_37 + y_59;
    int32_t per_thread_elements_61 = squot32(x_60, num_threads_58);
    int32_t y_115 = smod32(size_37, num_threads_58);
    int32_t x_116 = num_threads_58 - y_115;
    int32_t y_117 = smod32(x_116, num_threads_58);
    int32_t padded_size_118 = size_37 + y_117;
    int32_t padding_119 = padded_size_118 - size_37;
    int32_t x_121 = padded_size_118 + y_59;
    int32_t offset_multiple_122 = squot32(x_121, num_threads_58);
    int32_t bytes_139 = 4 * padding_119;
    struct memblock_device mem_140;
    
    mem_140.references = NULL;
    memblock_alloc_device(&mem_140, bytes_139);
    
    int32_t bytes_141 = 4 * padded_size_118;
    struct memblock_device mem_142;
    
    mem_142.references = NULL;
    memblock_alloc_device(&mem_142, bytes_141);
    
    int32_t tmp_offs_171 = 0;
    
    if (size_37 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_138.mem,
                                           mem_142.mem, 0, tmp_offs_171 * 4,
                                           size_37 * sizeof(int32_t), 0, NULL,
                                           NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    tmp_offs_171 += size_37;
    if (padding_119 * sizeof(int32_t) > 0) {
        OPENCL_SUCCEED(clEnqueueCopyBuffer(fut_cl_queue, mem_140.mem,
                                           mem_142.mem, 0, tmp_offs_171 * 4,
                                           padding_119 * sizeof(int32_t), 0,
                                           NULL, NULL));
        if (cl_debug)
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
    }
    tmp_offs_171 += padding_119;
    
    int32_t x_144 = 4 * per_thread_elements_61;
    int32_t bytes_143 = x_144 * num_threads_58;
    struct memblock_device mem_145;
    
    mem_145.references = NULL;
    memblock_alloc_device(&mem_145, bytes_143);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                  sizeof(mem_145.mem), &mem_145.mem));
    
    int32_t kernel_arg_218 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                  sizeof(kernel_arg_218), &kernel_arg_218));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                  sizeof(mem_142.mem), &mem_142.mem));
    
    int32_t kernel_arg_219 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                  sizeof(kernel_arg_219), &kernel_arg_219));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                  sizeof(per_thread_elements_61),
                                  &per_thread_elements_61));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                  sizeof(num_threads_58), &num_threads_58));
    
    int32_t kernel_arg_220 = per_thread_elements_61 * num_threads_58;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                  sizeof(kernel_arg_220), &kernel_arg_220));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7, (16 + 1) *
                                  16 * sizeof(int32_t), NULL));
    if (1 * (per_thread_elements_61 + srem32(16 - srem32(per_thread_elements_61,
                                                         16), 16)) *
        (num_threads_58 + srem32(16 - srem32(num_threads_58, 16), 16)) * 1 !=
        0) {
        const size_t global_work_size_221[3] = {per_thread_elements_61 +
                                                srem32(16 -
                                                       srem32(per_thread_elements_61,
                                                              16), 16),
                                                num_threads_58 + srem32(16 -
                                                                        srem32(num_threads_58,
                                                                               16),
                                                                        16), 1};
        const size_t local_work_size_225[3] = {16, 16, 1};
        int64_t time_start_222, time_end_223;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i32");
            fprintf(stderr, "%zu", global_work_size_221[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_221[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_221[2]);
            fprintf(stderr, "].\n");
            time_start_222 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i32, 3,
                                              NULL, global_work_size_221,
                                              local_work_size_225, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_223 = get_wall_time();
            
            long time_diff_224 = time_end_223 - time_start_222;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i32total_runtime += time_diff_224;
                fut_kernel_map_transpose_i32runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i32", (int) time_diff_224);
            }
        }
    }
    
    struct memblock_device mem_147;
    
    mem_147.references = NULL;
    memblock_alloc_device(&mem_147, bytes_141);
    
    int32_t x_149 = 4 * num_groups_56;
    int32_t bytes_148 = x_149 * group_size_57;
    struct memblock_device mem_150;
    
    mem_150.references = NULL;
    memblock_alloc_device(&mem_150, bytes_148);
    
    int32_t total_size_183 = sizeof(int32_t) * group_size_57;
    
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_62, 0, total_size_183, NULL));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_62, 1,
                                  sizeof(per_thread_elements_61),
                                  &per_thread_elements_61));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_62, 2, sizeof(group_size_57),
                                  &group_size_57));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_62, 3, sizeof(size_37),
                                  &size_37));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_62, 4, sizeof(mem_145.mem),
                                  &mem_145.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_62, 5, sizeof(num_threads_58),
                                  &num_threads_58));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_62, 6, sizeof(mem_147.mem),
                                  &mem_147.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_62, 7, sizeof(mem_150.mem),
                                  &mem_150.mem));
    if (1 * (num_groups_56 * group_size_57) != 0) {
        const size_t global_work_size_226[1] = {num_groups_56 * group_size_57};
        const size_t local_work_size_230[1] = {group_size_57};
        int64_t time_start_227, time_end_228;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "scan_kernel_62");
            fprintf(stderr, "%zu", global_work_size_226[0]);
            fprintf(stderr, "].\n");
            time_start_227 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, scan_kernel_62, 1,
                                              NULL, global_work_size_226,
                                              local_work_size_230, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_228 = get_wall_time();
            
            long time_diff_229 = time_end_228 - time_start_227;
            
            if (detail_timing) {
                scan_kernel_62total_runtime += time_diff_229;
                scan_kernel_62runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "scan_kernel_62",
                        (int) time_diff_229);
            }
        }
    }
    
    int32_t x_152 = 4 * num_threads_58;
    int32_t bytes_151 = x_152 * per_thread_elements_61;
    struct memblock_device mem_153;
    
    mem_153.references = NULL;
    memblock_alloc_device(&mem_153, bytes_151);
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 0,
                                  sizeof(mem_153.mem), &mem_153.mem));
    
    int32_t kernel_arg_231 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 1,
                                  sizeof(kernel_arg_231), &kernel_arg_231));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 2,
                                  sizeof(mem_147.mem), &mem_147.mem));
    
    int32_t kernel_arg_232 = 0;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 3,
                                  sizeof(kernel_arg_232), &kernel_arg_232));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 4,
                                  sizeof(num_threads_58), &num_threads_58));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 5,
                                  sizeof(per_thread_elements_61),
                                  &per_thread_elements_61));
    
    int32_t kernel_arg_233 = num_threads_58 * per_thread_elements_61;
    
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 6,
                                  sizeof(kernel_arg_233), &kernel_arg_233));
    OPENCL_SUCCEED(clSetKernelArg(fut_kernel_map_transpose_i32, 7, (16 + 1) *
                                  16 * sizeof(int32_t), NULL));
    if (1 * (num_threads_58 + srem32(16 - srem32(num_threads_58, 16), 16)) *
        (per_thread_elements_61 + srem32(16 - srem32(per_thread_elements_61,
                                                     16), 16)) * 1 != 0) {
        const size_t global_work_size_234[3] = {num_threads_58 + srem32(16 -
                                                                        srem32(num_threads_58,
                                                                               16),
                                                                        16),
                                                per_thread_elements_61 +
                                                srem32(16 -
                                                       srem32(per_thread_elements_61,
                                                              16), 16), 1};
        const size_t local_work_size_238[3] = {16, 16, 1};
        int64_t time_start_235, time_end_236;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "fut_kernel_map_transpose_i32");
            fprintf(stderr, "%zu", global_work_size_234[0]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_234[1]);
            fprintf(stderr, ", ");
            fprintf(stderr, "%zu", global_work_size_234[2]);
            fprintf(stderr, "].\n");
            time_start_235 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue,
                                              fut_kernel_map_transpose_i32, 3,
                                              NULL, global_work_size_234,
                                              local_work_size_238, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_236 = get_wall_time();
            
            long time_diff_237 = time_end_236 - time_start_235;
            
            if (detail_timing) {
                fut_kernel_map_transpose_i32total_runtime += time_diff_237;
                fut_kernel_map_transpose_i32runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n",
                        "fut_kernel_map_transpose_i32", (int) time_diff_237);
            }
        }
    }
    
    int32_t last_in_group_index_73 = group_size_57 - 1;
    struct memblock_device mem_155;
    
    mem_155.references = NULL;
    memblock_alloc_device(&mem_155, x_149);
    
    int32_t group_size_189;
    int32_t num_groups_190;
    
    group_size_189 = cl_group_size;
    num_groups_190 = squot32(num_groups_56 + group_size_189 - 1,
                             group_size_189);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_71, 0, sizeof(num_groups_56),
                                  &num_groups_56));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_71, 1, sizeof(group_size_57),
                                  &group_size_57));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_71, 2,
                                  sizeof(last_in_group_index_73),
                                  &last_in_group_index_73));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_71, 3, sizeof(mem_150.mem),
                                  &mem_150.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_71, 4, sizeof(mem_155.mem),
                                  &mem_155.mem));
    if (1 * (num_groups_190 * group_size_189) != 0) {
        const size_t global_work_size_239[1] = {num_groups_190 *
                     group_size_189};
        const size_t local_work_size_243[1] = {group_size_189};
        int64_t time_start_240, time_end_241;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_71");
            fprintf(stderr, "%zu", global_work_size_239[0]);
            fprintf(stderr, "].\n");
            time_start_240 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_71, 1,
                                              NULL, global_work_size_239,
                                              local_work_size_243, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_241 = get_wall_time();
            
            long time_diff_242 = time_end_241 - time_start_240;
            
            if (detail_timing) {
                map_kernel_71total_runtime += time_diff_242;
                map_kernel_71runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_71",
                        (int) time_diff_242);
            }
        }
    }
    
    struct memblock_device mem_157;
    
    mem_157.references = NULL;
    memblock_alloc_device(&mem_157, x_149);
    
    struct memblock_device mem_160;
    
    mem_160.references = NULL;
    memblock_alloc_device(&mem_160, x_149);
    
    int32_t total_size_202 = sizeof(int32_t) * num_groups_56;
    
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_80, 0, total_size_202, NULL));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_80, 1, sizeof(num_groups_56),
                                  &num_groups_56));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_80, 2, sizeof(mem_155.mem),
                                  &mem_155.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_80, 3, sizeof(mem_157.mem),
                                  &mem_157.mem));
    OPENCL_SUCCEED(clSetKernelArg(scan_kernel_80, 4, sizeof(mem_160.mem),
                                  &mem_160.mem));
    if (1 * num_groups_56 != 0) {
        const size_t global_work_size_244[1] = {num_groups_56};
        const size_t local_work_size_248[1] = {num_groups_56};
        int64_t time_start_245, time_end_246;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "scan_kernel_80");
            fprintf(stderr, "%zu", global_work_size_244[0]);
            fprintf(stderr, "].\n");
            time_start_245 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, scan_kernel_80, 1,
                                              NULL, global_work_size_244,
                                              local_work_size_248, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_246 = get_wall_time();
            
            long time_diff_247 = time_end_246 - time_start_245;
            
            if (detail_timing) {
                scan_kernel_80total_runtime += time_diff_247;
                scan_kernel_80runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "scan_kernel_80",
                        (int) time_diff_247);
            }
        }
    }
    
    struct memblock_device mem_163;
    
    mem_163.references = NULL;
    memblock_alloc_device(&mem_163, bytes_148);
    
    int32_t group_size_208;
    int32_t num_groups_209;
    
    group_size_208 = cl_group_size;
    num_groups_209 = squot32(num_groups_56 * group_size_57 + group_size_208 - 1,
                             group_size_208);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_96, 0, sizeof(mem_160.mem),
                                  &mem_160.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_96, 1, sizeof(num_groups_56),
                                  &num_groups_56));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_96, 2, sizeof(group_size_57),
                                  &group_size_57));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_96, 3, sizeof(mem_150.mem),
                                  &mem_150.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_96, 4, sizeof(mem_163.mem),
                                  &mem_163.mem));
    if (1 * (num_groups_209 * group_size_208) != 0) {
        const size_t global_work_size_249[1] = {num_groups_209 *
                     group_size_208};
        const size_t local_work_size_253[1] = {group_size_208};
        int64_t time_start_250, time_end_251;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_96");
            fprintf(stderr, "%zu", global_work_size_249[0]);
            fprintf(stderr, "].\n");
            time_start_250 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_96, 1,
                                              NULL, global_work_size_249,
                                              local_work_size_253, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_251 = get_wall_time();
            
            long time_diff_252 = time_end_251 - time_start_250;
            
            if (detail_timing) {
                map_kernel_96total_runtime += time_diff_252;
                map_kernel_96runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_96",
                        (int) time_diff_252);
            }
        }
    }
    
    struct memblock_device mem_165;
    
    mem_165.references = NULL;
    memblock_alloc_device(&mem_165, bytes_137);
    
    int32_t group_size_210;
    int32_t num_groups_211;
    
    group_size_210 = cl_group_size;
    num_groups_211 = squot32(size_37 + group_size_210 - 1, group_size_210);
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_104, 0, sizeof(mem_153.mem),
                                  &mem_153.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_104, 1,
                                  sizeof(per_thread_elements_61),
                                  &per_thread_elements_61));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_104, 2, sizeof(group_size_57),
                                  &group_size_57));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_104, 3, sizeof(size_37),
                                  &size_37));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_104, 4, sizeof(mem_163.mem),
                                  &mem_163.mem));
    OPENCL_SUCCEED(clSetKernelArg(map_kernel_104, 5, sizeof(mem_165.mem),
                                  &mem_165.mem));
    if (1 * (num_groups_211 * group_size_210) != 0) {
        const size_t global_work_size_254[1] = {num_groups_211 *
                     group_size_210};
        const size_t local_work_size_258[1] = {group_size_210};
        int64_t time_start_255, time_end_256;
        
        if (cl_debug) {
            fprintf(stderr, "Launching %s with global work size [",
                    "map_kernel_104");
            fprintf(stderr, "%zu", global_work_size_254[0]);
            fprintf(stderr, "].\n");
            time_start_255 = get_wall_time();
        }
        OPENCL_SUCCEED(clEnqueueNDRangeKernel(fut_cl_queue, map_kernel_104, 1,
                                              NULL, global_work_size_254,
                                              local_work_size_258, 0, NULL,
                                              NULL));
        if (cl_debug) {
            OPENCL_SUCCEED(clFinish(fut_cl_queue));
            time_end_256 = get_wall_time();
            
            long time_diff_257 = time_end_256 - time_start_255;
            
            if (detail_timing) {
                map_kernel_104total_runtime += time_diff_257;
                map_kernel_104runs++;
                fprintf(stderr, "kernel %s runtime: %ldus\n", "map_kernel_104",
                        (int) time_diff_257);
            }
        }
    }
    memblock_set_device(&out_mem_166, &mem_165);
    out_arrsize_168 = size_37;
    out_memsize_167 = bytes_137;
    
    struct tuple_int32_t_device_mem_int32_t retval_212;
    
    retval_212.elem_0 = out_memsize_167;
    retval_212.elem_1.references = NULL;
    memblock_set_device(&retval_212.elem_1, &out_mem_166);
    retval_212.elem_2 = out_arrsize_168;
    memblock_unref_device(&out_mem_166);
    memblock_unref_device(&mem_138);
    memblock_unref_device(&mem_140);
    memblock_unref_device(&mem_142);
    memblock_unref_device(&mem_145);
    memblock_unref_device(&mem_147);
    memblock_unref_device(&mem_150);
    memblock_unref_device(&mem_153);
    memblock_unref_device(&mem_155);
    memblock_unref_device(&mem_157);
    memblock_unref_device(&mem_160);
    memblock_unref_device(&mem_163);
    memblock_unref_device(&mem_165);
    return retval_212;
}
struct array_reader {
  char* elems;
  int64_t n_elems_space;
  int64_t elem_size;
  int64_t n_elems_used;
  int64_t *shape;
  int (*elem_reader)(void*);
};

static int peekc() {
  int c = getchar();
  ungetc(c,stdin);
  return c;
}

static int next_is_not_constituent() {
  int c = peekc();
  return c == EOF || !isalnum(c);
}

static void skipspaces() {
  int c = getchar();
  if (isspace(c)) {
    skipspaces();
  } else if (c == '-' && peekc() == '-') {
    // Skip to end of line.
    for (; c != '\n' && c != EOF; c = getchar());
    // Next line may have more spaces.
    skipspaces();
  } else if (c != EOF) {
    ungetc(c, stdin);
  }
}

static int read_elem(struct array_reader *reader) {
  int ret;
  if (reader->n_elems_used == reader->n_elems_space) {
    reader->n_elems_space *= 2;
    reader->elems = (char*) realloc(reader->elems,
                                    reader->n_elems_space * reader->elem_size);
  }

  ret = reader->elem_reader(reader->elems + reader->n_elems_used * reader->elem_size);

  if (ret == 0) {
    reader->n_elems_used++;
  }

  return ret;
}

static int read_array_elems(struct array_reader *reader, int dims) {
  int c;
  int ret;
  int first = 1;
  char *knows_dimsize = (char*) calloc(dims,sizeof(char));
  int cur_dim = dims-1;
  int64_t *elems_read_in_dim = (int64_t*) calloc(dims,sizeof(int64_t));
  while (1) {
    skipspaces();

    c = getchar();
    if (c == ']') {
      if (knows_dimsize[cur_dim]) {
        if (reader->shape[cur_dim] != elems_read_in_dim[cur_dim]) {
          ret = 1;
          break;
        }
      } else {
        knows_dimsize[cur_dim] = 1;
        reader->shape[cur_dim] = elems_read_in_dim[cur_dim];
      }
      if (cur_dim == 0) {
        ret = 0;
        break;
      } else {
        cur_dim--;
        elems_read_in_dim[cur_dim]++;
      }
    } else if (c == ',') {
      skipspaces();
      c = getchar();
      if (c == '[') {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        first = 1;
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else if (cur_dim == dims - 1) {
        ungetc(c, stdin);
        ret = read_elem(reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
      } else {
        ret = 1;
        break;
      }
    } else if (c == EOF) {
      ret = 1;
      break;
    } else if (first) {
      if (c == '[') {
        if (cur_dim == dims - 1) {
          ret = 1;
          break;
        }
        cur_dim++;
        elems_read_in_dim[cur_dim] = 0;
      } else {
        ungetc(c, stdin);
        ret = read_elem(reader);
        if (ret != 0) {
          break;
        }
        elems_read_in_dim[cur_dim]++;
        first = 0;
      }
    } else {
      ret = 1;
      break;
    }
  }

  free(knows_dimsize);
  free(elems_read_in_dim);
  return ret;
}

static int read_array(int64_t elem_size, int (*elem_reader)(void*),
               void **data, int64_t *shape, int64_t dims) {
  int ret;
  struct array_reader reader;
  int64_t read_dims = 0;
  while (1) {
    int c;
    skipspaces();
    c = getchar();
    if (c=='[') {
      read_dims++;
    } else {
      if (c != EOF) {
        ungetc(c, stdin);
      }
      break;
    }
  }

  if (read_dims != dims) {
    return 1;
  }

  reader.shape = shape;
  reader.n_elems_used = 0;
  reader.elem_size = elem_size;
  reader.n_elems_space = 16;
  reader.elems = (char*) realloc(*data, elem_size*reader.n_elems_space);
  reader.elem_reader = elem_reader;

  ret = read_array_elems(&reader, dims);

  *data = reader.elems;

  return ret;
}

static int read_int8(void* dest) {
  skipspaces();
  if (scanf("%hhi", (int8_t*)dest) == 1) {
    scanf("i8");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_int16(void* dest) {
  skipspaces();
  if (scanf("%hi", (int16_t*)dest) == 1) {
    scanf("i16");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_int32(void* dest) {
  skipspaces();
  if (scanf("%i", (int32_t*)dest) == 1) {
    scanf("i32");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_int64(void* dest) {
  skipspaces();
  if (scanf("%Li", (int64_t*)dest) == 1) {
    scanf("i64");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_char(void* dest) {
  skipspaces();
  if (scanf("%c", (char*)dest) == 1) {
    return 0;
  } else {
    return 1;
  }
}

static int read_double(void* dest) {
  skipspaces();
  if (scanf("%lf", (double*)dest) == 1) {
    scanf("f64");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_float(void* dest) {
  skipspaces();
  if (scanf("%f", (float*)dest) == 1) {
    scanf("f32");
    return next_is_not_constituent() ? 0 : 1;
  } else {
    return 1;
  }
}

static int read_bool(void* dest) {
  /* This is a monstrous hack.  Maybe we should get a proper lexer in here. */
  char b[4];
  skipspaces();
  if (scanf("%4c", b) == 1) {
    if (strncmp(b, "True", 4) == 0) {
      *(int*)dest = 1;
      return 0;
    } else if (strncmp(b, "Fals", 4) == 0 && getchar() == 'e') {
      *(int*)dest = 0;
      return 0;
    } else {
      return 1;
    }
  } else {
    return 1;
  }
}

static FILE *runtime_file;
static int perform_warmup = 0;
static int num_runs = 1;
int parse_options(int argc, char *const argv[])
{
    int ch;
    static struct option long_options[] = {{"write-runtime-to",
                                            required_argument, NULL, 1},
                                           {"runs", required_argument, NULL, 2},
                                           {"platform", required_argument, NULL,
                                            3}, {"device", required_argument,
                                                 NULL, 4}, {"synchronous",
                                                            no_argument, NULL,
                                                            5}, {"group-size",
                                                                 required_argument,
                                                                 NULL, 6},
                                           {"num-groups", required_argument,
                                            NULL, 7}, {0, 0, 0, 0}};
    
    while ((ch = getopt_long(argc, argv, ":t:r:p:d:s", long_options, NULL)) !=
           -1) {
        if (ch == 1 || ch == 't') {
            runtime_file = fopen(optarg, "w");
            if (runtime_file == NULL)
                panic(1, "Cannot open %s: %s", optarg, strerror(errno));
        }
        if (ch == 2 || ch == 'r') {
            num_runs = atoi(optarg);
            perform_warmup = 1;
            if (num_runs <= 0)
                panic(1, "Need a positive number of runs, not %s", optarg);
        }
        if (ch == 3 || ch == 'p')
            cl_preferred_platform = optarg;
        if (ch == 4 || ch == 'd')
            cl_preferred_device = optarg;
        if (ch == 5 || ch == 's')
            cl_debug = 1;
        if (ch == 6)
            cl_group_size = atoi(optarg);
        if (ch == 7)
            cl_num_groups = atoi(optarg);
        if (ch == ':')
            panic(-1, "Missing argument for option %s", argv[optind - 1]);
        if (ch == '?')
            panic(-1, "Unknown option %s", argv[optind - 1]);
    }
    return optind;
}
int main(int argc, char **argv)
{
    int64_t t_start, t_end;
    int time_runs;
    
    fut_progname = argv[0];
    
    int parsed_options = parse_options(argc, argv);
    
    argc -= parsed_options;
    argv += parsed_options;
    setup_opencl_and_load_kernels();
    
    int32_t a_mem_size_135;
    struct memblock a_mem_136;
    
    a_mem_136.references = NULL;
    memblock_alloc(&a_mem_136, 0);
    
    int32_t size_37;
    struct tuple_int32_t_device_mem_int32_t main_ret_259;
    
    {
        int64_t shape[1];
        
        if (read_array(sizeof(int32_t), read_int32, (void **) &a_mem_136.mem,
                       shape, 1) != 0)
            panic(1, "Syntax error when reading %s.\n", "[i32]");
        size_37 = shape[0];
        a_mem_size_135 = sizeof(int32_t) * shape[0];
    }
    
    struct memblock_device a_mem_device_260;
    
    a_mem_device_260.references = NULL;
    memblock_alloc_device(&a_mem_device_260, a_mem_size_135);
    if (a_mem_size_135 > 0)
        OPENCL_SUCCEED(clEnqueueWriteBuffer(fut_cl_queue, a_mem_device_260.mem,
                                            CL_TRUE, 0, a_mem_size_135,
                                            a_mem_136.mem + 0, 0, NULL, NULL));
    
    int32_t out_memsize_167;
    struct memblock out_mem_166;
    
    out_mem_166.references = NULL;
    
    int32_t out_arrsize_168;
    
    if (perform_warmup) {
        time_runs = 0;
        t_start = get_wall_time();
        main_ret_259 = futhark_main(a_mem_size_135, a_mem_device_260, size_37);
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        memblock_unref_device(&main_ret_259.elem_1);
    }
    time_runs = 1;
    /* Proper run. */
    for (int run = 0; run < num_runs; run++) {
        if (run == num_runs - 1)
            detail_timing = 1;
        t_start = get_wall_time();
        main_ret_259 = futhark_main(a_mem_size_135, a_mem_device_260, size_37);
        OPENCL_SUCCEED(clFinish(fut_cl_queue));
        t_end = get_wall_time();
        
        long elapsed_usec = t_end - t_start;
        
        if (time_runs && runtime_file != NULL)
            fprintf(runtime_file, "%ld\n", elapsed_usec);
        if (run < num_runs - 1) {
            memblock_unref_device(&main_ret_259.elem_1);
        }
    }
    memblock_unref(&a_mem_136);
    out_memsize_167 = main_ret_259.elem_0;
    memblock_alloc(&out_mem_166, out_memsize_167);
    if (out_memsize_167 > 0)
        OPENCL_SUCCEED(clEnqueueReadBuffer(fut_cl_queue,
                                           main_ret_259.elem_1.mem, CL_TRUE, 0,
                                           out_memsize_167, out_mem_166.mem + 0,
                                           0, NULL, NULL));
    out_arrsize_168 = main_ret_259.elem_2;
    if (out_arrsize_168 == 0)
        printf("empty(%s)", "i32");
    else {
        int print_i_261;
        
        putchar('[');
        for (print_i_261 = 0; print_i_261 < out_arrsize_168; print_i_261++) {
            int32_t *print_elem_262 = (int32_t *) out_mem_166.mem +
                    print_i_261 * 1;
            
            printf("%di32", *print_elem_262);
            if (print_i_261 != out_arrsize_168 - 1)
                printf(", ");
        }
        putchar(']');
    }
    printf("\n");
    
    int total_runtime = 0;
    int total_runs = 0;
    
    if (cl_debug) {
        fprintf(stderr,
                "Kernel map_kernel_52                executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_52runs, (long) map_kernel_52total_runtime /
                (map_kernel_52runs != 0 ? map_kernel_52runs : 1),
                (long) map_kernel_52total_runtime);
        total_runtime += map_kernel_52total_runtime;
        total_runs += map_kernel_52runs;
        fprintf(stderr,
                "Kernel fut_kernel_map_transpose_i32 executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                fut_kernel_map_transpose_i32runs,
                (long) fut_kernel_map_transpose_i32total_runtime /
                (fut_kernel_map_transpose_i32runs !=
                 0 ? fut_kernel_map_transpose_i32runs : 1),
                (long) fut_kernel_map_transpose_i32total_runtime);
        total_runtime += fut_kernel_map_transpose_i32total_runtime;
        total_runs += fut_kernel_map_transpose_i32runs;
        fprintf(stderr,
                "Kernel scan_kernel_62               executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                scan_kernel_62runs, (long) scan_kernel_62total_runtime /
                (scan_kernel_62runs != 0 ? scan_kernel_62runs : 1),
                (long) scan_kernel_62total_runtime);
        total_runtime += scan_kernel_62total_runtime;
        total_runs += scan_kernel_62runs;
        fprintf(stderr,
                "Kernel map_kernel_71                executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_71runs, (long) map_kernel_71total_runtime /
                (map_kernel_71runs != 0 ? map_kernel_71runs : 1),
                (long) map_kernel_71total_runtime);
        total_runtime += map_kernel_71total_runtime;
        total_runs += map_kernel_71runs;
        fprintf(stderr,
                "Kernel scan_kernel_80               executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                scan_kernel_80runs, (long) scan_kernel_80total_runtime /
                (scan_kernel_80runs != 0 ? scan_kernel_80runs : 1),
                (long) scan_kernel_80total_runtime);
        total_runtime += scan_kernel_80total_runtime;
        total_runs += scan_kernel_80runs;
        fprintf(stderr,
                "Kernel map_kernel_96                executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_96runs, (long) map_kernel_96total_runtime /
                (map_kernel_96runs != 0 ? map_kernel_96runs : 1),
                (long) map_kernel_96total_runtime);
        total_runtime += map_kernel_96total_runtime;
        total_runs += map_kernel_96runs;
        fprintf(stderr,
                "Kernel map_kernel_104               executed %6d times, with average runtime: %6ldus\tand total runtime: %6ldus\n",
                map_kernel_104runs, (long) map_kernel_104total_runtime /
                (map_kernel_104runs != 0 ? map_kernel_104runs : 1),
                (long) map_kernel_104total_runtime);
        total_runtime += map_kernel_104total_runtime;
        total_runs += map_kernel_104runs;
    }
    if (cl_debug)
        fprintf(stderr, "Ran %d kernels with cumulative runtime: %6ldus\n",
                total_runs, total_runtime);
    memblock_unref_device(&main_ret_259.elem_1);
    if (runtime_file != NULL)
        fclose(runtime_file);
    return 0;
}
