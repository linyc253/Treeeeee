#ifndef PARAMETER_H_INCLUDED
#define PARAMETER_H_INCLUDED

typedef struct {
    char key[64];
    char value[64];
} param_t;

// Global variables!!
extern param_t params[];
extern int param_count;
extern int DIM;

int handler(void* user, const char* section, const char* name, const char* value);
const char* get_string(const char* key, const char* init);
double get_double(const char* key, double init);
int get_int(const char* key, int init);
void Read_Input_Parameter();

#endif