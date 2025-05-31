// Written by CharGPT
#ifndef NODE_POOL_H
#define NODE_POOL_H

#include <stdlib.h>
#include <stdio.h>
#include "tree.h"

// NodePool with chunked growth to keep pointers stable
typedef struct {
    Node **chunks;         // array of chunk pointers
    int num_chunks;        // how many chunks currently allocated
    int capacity_chunks;   // capacity of the chunks array
    int chunk_size;        // number of nodes per chunk
    int cur_chunk;         // index of current chunk
    int offset;            // next free index within current chunk
} NodePool;

// Create a pool with specified chunk size
NodePool* create_node_pool(int chunk_size);

// Allocate a node from the pool
Node* alloc_node(NodePool *pool);

// Reset pool to reuse in next region
void reset_node_pool(NodePool *pool);

// Free the pool when done
void free_node_pool(NodePool *pool);

#endif // NODE_POOL_H
