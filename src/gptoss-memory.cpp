#include "gptoss-memory.h"

gptoss_memory_status gptoss_memory_status_combine(gptoss_memory_status s0, gptoss_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case GPTOSS_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case GPTOSS_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case GPTOSS_MEMORY_STATUS_FAILED_PREPARE:
        case GPTOSS_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case GPTOSS_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case GPTOSS_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case GPTOSS_MEMORY_STATUS_FAILED_PREPARE:
        case GPTOSS_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? GPTOSS_MEMORY_STATUS_SUCCESS : GPTOSS_MEMORY_STATUS_NO_UPDATE;
}

bool gptoss_memory_status_is_fail(gptoss_memory_status status) {
    switch (status) {
        case GPTOSS_MEMORY_STATUS_SUCCESS:
        case GPTOSS_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case GPTOSS_MEMORY_STATUS_FAILED_PREPARE:
        case GPTOSS_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}
