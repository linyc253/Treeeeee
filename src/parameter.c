#include "ini.h"
#include "parameter.h"
#include <string.h>
#include <stdlib.h>
#define MAX_PARAMS 100

// Global variables
param_t params[MAX_PARAMS];
int param_count = 0;
int DIM;

// Parse the parameter file with ini.c (New BSD license)
void Read_Input_Parameter(const char* filename){
    if (ini_parse(filename, handler, NULL) < 0) {
        printf("Failed to read %s\n", filename);
        exit(EXIT_FAILURE);
    }
    return;
}

int handler(void* user, const char* section, const char* name, const char* value) {
    (void)user; // unused

    if (param_count < MAX_PARAMS) {
        snprintf(params[param_count].key, sizeof(params[param_count].key),
                 "%s.%s", section, name);
        strncpy(params[param_count].value, value, sizeof(params[param_count].value));
        param_count++;
    }

    return 1;
}

// Same as get_int, but used to read in string
const char* get_string(const char* key, const char* init) {
    for (int i = 0; i < param_count; i++) {
        if (strcmp(params[i].key, key) == 0)
            return params[i].value;
    }
    return init;
}

// Same as get_int, but used to read in floating-point parameter
double get_double(const char* key, double init) {
    const char* val = get_string(key, NULL);
    return val ? atof(val) : init;
}

// Read integer with parameter 'key', if not exist, then return 'init'
// For example, if the file looks like
//    [basic setting]
//    DIM = 2
// then you should use DIM = get_int("basic setting.DIM", 3) to parse the parameter
int get_int(const char* key, int init) {
    const char* val = get_string(key, NULL);
    return val ? atoi(val) : init;
}