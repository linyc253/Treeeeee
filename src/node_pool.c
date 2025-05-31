// Written by CharGPT
#include "node_pool.h"

#define INITIAL_CHUNKS_CAP 10

NodePool* create_node_pool(int chunk_size) {
    NodePool *pool = (NodePool*)malloc(sizeof(NodePool));
    if (!pool) {
        fprintf(stderr, "ERROR: Could not allocate NodePool\n");
        exit(EXIT_FAILURE);
    }
    pool->chunk_size = chunk_size;
    pool->capacity_chunks = INITIAL_CHUNKS_CAP;
    pool->num_chunks = 0;
    pool->cur_chunk = 0;
    pool->offset = 0;
    pool->chunks = (Node**)malloc(sizeof(Node*) * pool->capacity_chunks);
    if (!pool->chunks) {
        fprintf(stderr, "ERROR: Could not allocate chunks array\n");
        free(pool);
        exit(EXIT_FAILURE);
    }
    // allocate first chunk
    pool->chunks[pool->num_chunks++] = (Node*)malloc(sizeof(Node) * chunk_size);
    if (!pool->chunks[0]) {
        fprintf(stderr, "ERROR: Could not allocate initial chunk\n");
        free(pool->chunks);
        free(pool);
        exit(EXIT_FAILURE);
    }
    return pool;
}

Node* alloc_node(NodePool *pool) {
    if (pool->offset >= pool->chunk_size) {
        // need new chunk
        if (pool->num_chunks >= pool->capacity_chunks) {
            pool->capacity_chunks *= 2;
            Node **new_chunks = (Node**)realloc(pool->chunks, sizeof(Node*) * pool->capacity_chunks);
            if (!new_chunks) {
                fprintf(stderr, "ERROR: Could not realloc chunks array\n");
                exit(EXIT_FAILURE);
            }
            pool->chunks = new_chunks;
        }
        // allocate next chunk
        pool->chunks[pool->num_chunks] = (Node*)malloc(sizeof(Node) * pool->chunk_size);
        if (!pool->chunks[pool->num_chunks]) {
            fprintf(stderr, "ERROR: Could not allocate new chunk\n");
            exit(EXIT_FAILURE);
        }
        pool->cur_chunk = pool->num_chunks;
        pool->num_chunks++;
        pool->offset = 0;
    }
    return &pool->chunks[pool->cur_chunk][pool->offset++];
}

void reset_node_pool(NodePool *pool) {
    pool->cur_chunk = 0;
    pool->offset = 0;
}

void free_node_pool(NodePool *pool) {
    if (pool) {
        for (int i = 0; i < pool->num_chunks; i++) {
            free(pool->chunks[i]);
        }
        free(pool->chunks);
        free(pool);
    }
}
