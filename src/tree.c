// This tree algorithm is implemented by following the psuedo code from: 
//   https://web.archive.org/web/20160510023001/http://www.eecs.berkeley.edu/~demmel/cs267/lecture26/lecture26.html
// I preserve the psuedo code as comments for better understanding about what I was doing:)
#include <stdlib.h>
#include <stdio.h>
#include "parameter.h"
#include "particle.h"
#include "tree.h"
#include "math.h"
#include "string.h"
#include "../lib/dsyevh3.h"

#ifdef CUDA
#include "force_gpu.h"
#endif

#ifdef OMP
#include <omp.h>
#endif

#ifdef DEBUG
#include <sys/time.h>
#endif

Tree Initialize_Tree(Particle* P, int npart){
    Tree T;
    T.root = (Node*) malloc(sizeof(Node));
    T.root->npart = 0;
    T.root->parent = NULL;
    T.root->D = 0.0;
    for (int i = 0; i < DIM; i++){
        T.box_max[i] = P[0].x[i];
        T.box_min[i] = P[0].x[i];
        for(int j = 1; j < npart; j++){
            T.box_max[i] = fmax(T.box_max[i], P[j].x[i]);
            T.box_min[i] = fmin(T.box_min[i], P[j].x[i]);
        }
        T.root->D = fmax(T.root->D, T.box_max[i] - T.box_min[i]);
        T.root->c[i] = (T.box_min[i] + T.box_max[i]) / 2.0; // temporarily set to center
    }

    return T;
}

// find belonging child for particle
int Which_Child(Node* node, Particle p){
    int i = 0;
    for(int j = 0; j < DIM; j++){
        if(p.x[j] > node->c[j]) i += (1<<j);
    }
    return i;
}

Node* New_Node(Node* parent, int index){
    Node* newNode = (Node*) malloc(sizeof(Node));
    newNode->parent = parent;
    newNode->children = NULL;
    newNode->npart = 0;
    newNode->i = -1;
    newNode->D = parent->D / 2.0;
    parent->children[index] = newNode;
    for(int j = 0; j < DIM; j++){
        if((index / (1<<j)) % 2 == 0) newNode->c[j] = parent->c[j] - newNode->D / 2.0;
        else newNode->c[j] = parent->c[j] + newNode->D / 2.0;
    }
    return newNode;
}

void Initialize_Children(Node* node){
    node->children = (Node**) malloc((1 << DIM) * sizeof(Node*)); // bit operation: 1<<n = 2^n
    for(int i = 0; i < 1<<DIM; i++) node->children[i] = NULL;
}

// procedure QuadInsert(i,n)   
//      ... Try to insert particle i at node n in quadtree
//      ... By construction, each leaf will contain either 
//      ... 1 or 0 particles
//      if the subtree rooted at n contains more than 1 particle
//         determine which child c of node n particle i lies in
//           QuadInsert(i,c)
//      else if the subtree rooted at n contains one particle 
//         ... n is a leaf
//         add n's four children to the Quadtree
//         move the particle already in n into the child 
//            in which it lies
//         let c be child in which particle i lies
//         QuadInsert(i,c)
//      else if the subtree rooted at n is empty        
//         ... n is a leaf 
//         store particle i in node n
//      endif

// In principle: node == parent->children[index]
void Tree_Insert(Node* node, Node* parent, int index, Particle* P, int i){
    if(node == NULL){
        node = New_Node(parent, index);
        node->i = i;
    }
    else if(node->npart > 1){
        index = Which_Child(node, P[i]);
        Tree_Insert(node->children[index], node, index, P, i);
    }
    else if(node->npart == 1){
        Initialize_Children(node);
        index = Which_Child(node, P[i]);
        Tree_Insert(node->children[index], node, index, P, i);
        index = Which_Child(node, P[node->i]);
        Tree_Insert(node->children[index], node, index, P, node->i);
        node->i = -1;
    }
    else{ // for initial insert when tree is empty
        node->i = i;
    }
    node->npart++;
}

void Clear_Empty(Node* node){
    // if(node->npart == 0) return
    for(int i = 0; i < 1<<DIM; i++){
        if(node->children[i]->npart == 0) {
            free(node->children[i]);
            node->children[i] = NULL;
        }
        else if(node->children[i]->npart > 1) {
            Clear_Empty(node->children[i]);
        }
    }
}


Node* Tree_Merge(Node* node1, Node* node2, Particle* P){
    if(node1 == NULL || node1->npart == 0){ // node1 is empty
        return node2;
    }
    if(node2 == NULL || node2->npart == 0){ // node2 is empty
        return node1;
    }
    if(node1->npart == 1){
        Initialize_Children(node1);
        int index = Which_Child(node1, P[node1->i]);
        Node* child = New_Node(node1, index);
        child->npart = node1->npart;
        child->i = node1->i;
        child->cost = node1->cost; // Ideally we should +1 here, but then we'll need to update their ancestor, so let's not do it
        child->m = node1->m;
        for(int i = 0; i < DIM; i++) child->x[i] = node1->x[i];
    }
    if(node2->npart == 1){
        Initialize_Children(node2);
        int index = Which_Child(node2, P[node2->i]);
        Node* child = New_Node(node2, index);
        child->npart = node2->npart;
        child->i = node2->i;
        child->cost = node2->cost; // Ideally we should +1 here, but then we'll need to update their ancestor, so let's not do it
        child->m = node2->m;
        for(int i = 0; i < DIM; i++) child->x[i] = node2->x[i];
    }

    for(int i = 0; i < 1<<DIM; i++){
        node1->children[i] = Tree_Merge(node1->children[i], node2->children[i], P);
    }
    node1->npart += node2->npart;
    node1->i = -1;
    node1->cost += node2->cost;
    double m = node1->m + node2->m;
    for(int i = 0; i < DIM; i++) node1->x[i] = (node1->x[i] * node1->m + node2->x[i] + node2->m) / m;
    node1->m = m;

    free(node2); // memory leak ?
    return node1;
}

// procedure QuadtreeBuild
//      Quadtree = {empty}
//        For i = 1 to n          ... loop over all particles
//          QuadInsert(i, root)   ... insert particle i in quadtree
//        end for
//        ... at this point, the quadtree may have some empty 
//        ... leaves, whose siblings are not empty
//        Traverse the tree (via, say, breadth first search), 
//          eliminating empty leaves
Tree Tree_Build(Particle* P, int npart, int tid){
    Tree T = Initialize_Tree(P, npart);
    for (int i = 0; i < npart; i++){
         if(P[i].zone == tid) Tree_Insert(T.root, NULL, 0, P, i);
    }
    return T;
}



// function ( mass, cm ) = Compute_Mass(n)    
//        ... Compute the mass and center of mass (cm) of 
//        ... all the particles in the subtree rooted at n
//        if n contains 1 particle
//             ... the mass and cm of n are identical to 
//             ... the particle's mass and position
//             store ( mass, cm ) at n
//             return ( mass, cm )
//        else
//             for all four children c(i) of n (i=1,2,3,4)
//                 ( mass(i), cm(i) ) = Compute_Mass(c(i))
//             end for
//             mass = mass(1) + mass(2) + mass(3) + mass(4) 
//                  ... the mass of a node is the sum of 
//                  ... the masses of the children
//             cm = (  mass(1)*cm(1) + mass(2)*cm(2) 
//                   + mass(3)*cm(3) + mass(4)*cm(4)) / mass
//                  ... the cm of a node is a weighted sum of 
//                  ... the cm's of the children
//             store ( mass, cm ) at n
//             return ( mass, cm )
//        end
int Compute_m_and_x(Node* node, Particle* P, int depth){
    if(node == NULL || node->npart == 0) return -1;
    
    if(node->npart == 1){
        node->m = P[node->i].m;
        node->cost = (long long)depth;
        for(int i = 0; i < DIM; i++) node->x[i] = P[node->i].x[i];
    }
    else{
        node->m = 0.0;
        node->cost = 0;
        for(int i = 0; i < DIM; i++) node->x[i] = 0.0;

        for(int i = 0; i < 1<<DIM; i++){
            if(Compute_m_and_x(node->children[i], P, depth + 1) != -1){
                node->m += node->children[i]->m;
                node->cost += node->children[i]->cost;
                for(int j = 0; j < DIM; j++) node->x[j] += node->children[i]->m * node->children[i]->x[j];
            }
        }
        for(int i = 0; i < DIM; i++) node->x[i] /= node->m;
    }
    return 0;
}

long long Set_Costzone(Node* node, Particle* P, long long cost, long long cost_tot, int OMP_NUM_THREADS, int tid){
    if(node == NULL) return cost;

    long long zone_left = cost / ((cost_tot + (long long)OMP_NUM_THREADS - 1) / (long long)OMP_NUM_THREADS);
    long long zone_right = (cost + node->cost - 1) / ((cost_tot + (long long)OMP_NUM_THREADS - 1) / (long long)OMP_NUM_THREADS);

    if(tid < zone_left || tid > zone_right) return cost + node->cost;

    if(node->npart == 1){
        if(tid == zone_right) P[node->i].zone = tid;
        return cost + node->cost;
    }
        
    for(int i = 0; i < 1<<DIM; i++){
        cost = Set_Costzone(node->children[i], P, cost, cost_tot, OMP_NUM_THREADS, tid);
    }
    return cost;
}

// compute quadrupole tensor and pseudoparticle positions
int compute_quadrupole(Node* node, Particle* particles){
    if(node == NULL) {
        return -1;
    }
    // for single particle
    if(node->npart == 1){
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                node->p2_x[i][j] = node->x[j];
            }
        }
    }
    // sum over particles
    else{
        // initialise
        double quad_tensor[3][3];
        for(int i = 0; i < 3; i++) {
            for(int j = 0; j < 3; j++) {
                quad_tensor[i][j] = 0;
                node->p2_x[i][j] = 0;
            }
        }
        // sum over all particles inside
        for(int p_index = 0; p_index < 1<<DIM; p_index++) {
            Node* child = node->children[p_index];
            if(compute_quadrupole(child, particles) != -1){
                for (int pp = 0; pp < 3; pp++) {
                    double r[3] = { child->p2_x[pp][0] - node->x[0], child->p2_x[pp][1] - node->x[1], child->p2_x[pp][2] - node->x[2] };
                    double singlet = child->m / 3 / 2 * (pow(r[0], 2) + pow(r[1], 2) + pow(r[2], 2));
                    for (int i = 0; i < 3; i++) {
                        for (int j = 0; j < 3; j++) {
                            quad_tensor[i][j] += 3 * child->m / 3 / 2 * r[i] * r[j];
                        }
                        quad_tensor[i][i] -= singlet;
                    }
                }
            }
        }
        // compute eigenvalue of quadrupole tensor
        double eigval[3];
        double eigvec[3][3];
        dsyevh3(quad_tensor, eigvec, eigval);

        // sort in eigenvalue
        for (int i = 0; i < 3; i++){
            if (i == 2){
                i = 0;
                if (eigval[i] < eigval[i + 1]){
                    double temp_val = eigval[i];
                    eigval[i] = eigval[i + 1];
                    eigval[i + 1] = temp_val;

                    double temp_vec[3] = {eigvec[0][i], eigvec[1][i], eigvec[2][i]};
                    eigvec[0][i] = eigvec[0][i + 1];
                    eigvec[1][i] = eigvec[1][i + 1];
                    eigvec[2][i] = eigvec[2][i + 1];
                    eigvec[0][i + 1] = temp_vec[0];
                    eigvec[1][i + 1] = temp_vec[1];
                    eigvec[2][i + 1] = temp_vec[2];
                }
                break;
            }
            if (eigval[i] < eigval[i + 1]){
                double temp_val = eigval[i];
                eigval[i] = eigval[i + 1];
                eigval[i + 1] = temp_val;

                double temp_vec[3] = {eigvec[0][i], eigvec[1][i], eigvec[2][i]};
                eigvec[0][i] = eigvec[0][i + 1];
                eigvec[1][i] = eigvec[1][i + 1];
                eigvec[2][i] = eigvec[2][i + 1];
                eigvec[0][i + 1] = temp_vec[0];
                eigvec[1][i + 1] = temp_vec[1];
                eigvec[2][i + 1] = temp_vec[2];
            }
        }
        
        // alpha, beta, and check for imaginary number
        double alpha = 2 * eigval[0] + eigval[1];
        if (fabs(alpha) < 1e-10){
            alpha = 0;
        } 
        else{
            alpha = sqrt(alpha / node->m);
        }
        double beta = eigval[0] + 2 * eigval[1];
        if (fabs(beta) < 1e-10){
            beta = 0;
        } 
        else{
            beta = sqrt(beta / (3 * node->m));
        }

        double temp_p2_x[3][3];

        temp_p2_x[0][0] = 0;
        temp_p2_x[0][1] = 2 * beta;
        temp_p2_x[0][2] = 0;
        temp_p2_x[1][0] = alpha;
        temp_p2_x[1][1] = -beta;
        temp_p2_x[1][2] = 0;
        temp_p2_x[2][0] = -alpha;
        temp_p2_x[2][1] = -beta;
        temp_p2_x[2][2] = 0;

        // transform back
        for (int pp = 0; pp < 3; pp++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    node->p2_x[pp][i] += eigvec[i][j] * temp_p2_x[pp][j];
                }
                node->p2_x[pp][i] += node->x[i];
            }
        }
    }
    return 0;
}

// Calculate gravitational force by (G = 1)
//  a.f += -(a.m * b.m / (|r|^2 + epsilon^2)^{3/2}) r
// where the vector r = a.x - b.x
void compute_force(Coord4* group_xyzm, Coord4* cell_xyzm, Coord3* force_xyz, int number_in_group, int filled_cell, double epsilon){
    for (int i = 0; i < number_in_group; i++) {
        for (int j = 0; j < 3; j++) {
            force_xyz[i].x[j] = 0;
        }
    }
    // particle to node 
    for (int p = 0; p < number_in_group; p++) {
        for (int c = 0; c < filled_cell; c++) {
            double r[DIM];
            double r_norm = 0;
            for (int i = 0; i < DIM; i++) {
                r[i] = group_xyzm[p].x[i] - cell_xyzm[c].x[i];
                r_norm += pow(r[i], 2);
            }
            for (int i = 0; i < DIM; i++) {
                force_xyz[p].x[i] += -group_xyzm[p].m * cell_xyzm[c].m * r[i] / pow(r_norm + pow(epsilon, 2), 1.5);
            }
        }
    }
    // particle to particle inside group
    for (int p = 0; p < number_in_group; p++) {
        for (int c = 0; c < number_in_group; c++) {
            if (p == c) {
                continue;
            }
            double r[DIM];
            double r_norm = 0;
            for (int i = 0; i < DIM; i++) {
                r[i] = group_xyzm[p].x[i] - group_xyzm[c].x[i];
                r_norm += pow(r[i], 2);
            }
            for (int i = 0; i < DIM; i++) {
                force_xyz[p].x[i] += -group_xyzm[p].m * group_xyzm[c].m * r[i] / pow(r_norm + pow(epsilon, 2), 1.5);
            }
        }
    }
}

// fill particle coordinates and mass to grouping array
void fill_xyzm(Node* node, Particle* particles, Coord4* group_xyzm, int* particle_indices, int* filled) {
    if (node == NULL){
        return;
    }
    if (node->npart == 1) {
        particle_indices[*filled] = node->i;
        for (int i = 0; i < 3; i++) {
            group_xyzm[*filled].x[i] = particles[node->i].x[i];
        }
        group_xyzm[*filled].m = particles[node->i].m;
        *filled = *filled + 1;
    }
    else {
        for (int i = 0; i < 1 << DIM; i++) {
            fill_xyzm(node->children[i], particles, group_xyzm, particle_indices, filled);
        }
    }
}

// create grouping
void assign_group(Node* node, Particle* particles, int n_crit, int* n_groups, Node** group_nodes) {
    if (node == NULL) return;

    if (node->npart <= n_crit) {
        group_nodes[*n_groups] = node;
        *n_groups = *n_groups + 1;
    }
    else {
        for (int i = 0; i < 1 << DIM; i++) {
            assign_group(node->children[i], particles, n_crit, n_groups, group_nodes);
        }
    }
}

// traverse and fill interaction list
void traverse_node(Node* node, Node* group_node, Coord4* cell_xyzm, int* filled, int poles, double theta) {
    if (node == NULL || node == group_node) {
        return;
    }
    if (node->npart == 1) {
        for (int j = 0; j < 3; j++) {
            cell_xyzm[*filled].x[j] = node->x[j];
        }
        cell_xyzm[*filled].m = node->m;
        *filled = *filled + 1;
        return;
    }

    double r = 0;
    for (int j = 0; j < 3; j++) {
        r += pow(group_node->c[j] - node->x[j], 2);
    }
    r = sqrt(r) - sqrt((double)DIM) * (group_node->D) / 2;
    if (node->D / r < theta && r > 0) {
        if (poles == 1) {
            for (int j = 0; j < 3; j++) {
                cell_xyzm[*filled].x[j] = node->x[j];
            }
            cell_xyzm[*filled].m = node->m;
            *filled = *filled + 1;
        }
        else if (poles == 2) {
            for (int pp = 0; pp < 3; pp++) {
                for (int j = 0; j < 3; j++) {
                    cell_xyzm[*filled].x[j] = node->p2_x[pp][j];
                }
                cell_xyzm[*filled].m = node->m / 3;
                *filled = *filled + 1;
            }
        }
    }
    else {
        for (int j = 0; j < 1 << DIM; j++) {
            traverse_node(node->children[j], group_node, cell_xyzm, filled, poles, theta);
        }
    }
}

// construct interaction list
void compute_interaction(Node* root, Particle* particles, Coord4* groups_xyzm, Node* group_node, int* particle_indices,
                         int n_particles, int number_in_group, double theta, int poles, double epsilon){

    if (poles == 1) {
        Coord4* cell_xyzm = (Coord4*) malloc((n_particles - number_in_group) * sizeof(Coord4));
        int filled = 0;
        traverse_node(root, group_node, cell_xyzm, &filled, poles, theta);

        Coord3 force_xyz[number_in_group];
        #ifdef CUDA
        int threadsPerBlock = get_int("GPU.threadsPerBlock", 32);
        Particle_Cell_Force_gpu(groups_xyzm, number_in_group, cell_xyzm, filled, force_xyz, epsilon, threadsPerBlock);
        #else
        compute_force(groups_xyzm, cell_xyzm, force_xyz, number_in_group, filled, epsilon);
        #endif
        for (int i = 0; i < number_in_group; i++) {
            for (int j = 0; j < DIM; j++) {
                particles[particle_indices[i]].f[j] += force_xyz[i].x[j];
            }
        }
        free(cell_xyzm);
    }
    else if (poles == 2) {
        Coord4* cell_xyzm = (Coord4*) malloc(3 * (n_particles - number_in_group) * sizeof(Coord4));
        int filled = 0;
        traverse_node(root, group_node, cell_xyzm, &filled, poles, theta);

        Coord3 force_xyz[number_in_group];
        #ifdef CUDA
        int threadsPerBlock = get_int("GPU.threadsPerBlock", 32);
        Particle_Cell_Force_gpu(groups_xyzm, number_in_group, cell_xyzm, filled, force_xyz, epsilon, threadsPerBlock);
        #else
        compute_force(groups_xyzm, cell_xyzm, force_xyz, number_in_group, filled, epsilon);
        #endif
        for (int i = 0; i < number_in_group; i++) {
            for (int j = 0; j < DIM; j++) {
                particles[particle_indices[i]].f[j] += force_xyz[i].x[j];
                filled++;
            }
        }
        free(cell_xyzm);
    }
    else {
        printf("The parameter POLES looks very funny, please don't try to break the program\n");
    }
    return;
}

// Set all the force to zero
void Zero_Force(Particle* P, int npart){
    for(int i = 0; i < npart; i++){
        for(int j = 0; j < DIM; j++) { 
            P[i].f[j] = 0.0; 
        }
    }
    return;
}

void Free_Tree(Node* node){
    if(node == NULL) return;
    if(node->children != NULL){
        for(int i = 0; i < 1<<DIM; i++){
            Free_Tree(node->children[i]);
        }
        free(node->children);
    }
    free(node);
}

// Main routine to calculate the tree force
void total_force_tree(Particle* P, int npart){
    // ---------------1. Build the Tree---------------
    #ifdef DEBUG
    struct timeval t0, t1;
    gettimeofday(&t0, 0);
    #endif
    
    #ifdef OMP
    int OMP_NUM_THREADS = get_int("Openmp.THREADS", 1);
    omp_set_num_threads(OMP_NUM_THREADS);
    Tree T_local[OMP_NUM_THREADS];
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        T_local[tid] = Tree_Build(P, npart, tid);
        Compute_m_and_x(T_local[tid].root, P, 0);
    }

    Tree T;
    T = T_local[0];
    for (int i = 1; i < OMP_NUM_THREADS; i++){
        T.root = Tree_Merge(T.root, T_local[i].root, P);
    }
    
    if(T.root->npart != npart){
        printf("BUG: number of particle in tree (%d) mismatch with npart (%d)\n", T.root->npart, npart);
        exit(EXIT_FAILURE);
    }

    if(OMP_NUM_THREADS > 1){
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            long long cost = Set_Costzone(T.root, P, 0, T.root->cost, OMP_NUM_THREADS, tid);
            if(T.root->cost != cost){
                printf("BUG: number of cost in tree (%lld) mismatch with cost (%lld)\n", T.root->cost, cost);
                exit(EXIT_FAILURE);
            }
        }
        
    }
    
    #else
    Tree T = Tree_Build(P, npart, 0);
    if(T.root->npart != npart){
        printf("BUG: number of particle in tree (%d) mismatch with npart (%d)\n", T.root->npart, npart);
        exit(EXIT_FAILURE);
    }
    Compute_m_and_x(T.root, P, 0);
    #endif
    
    #ifdef DEBUG
    gettimeofday(&t1, 0);
    printf("timeElapsed for Tree_Build(): %lu ms\n", (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000); 
    #endif
    
    // ---------------2. Compute the quadrupole (not parallelized yet) ---------------
    #ifdef DEBUG
    gettimeofday(&t0, 0);
    #endif

    int poles = get_int("Tree.POLES", 1);
    if (poles == 2) {
        compute_quadrupole(T.root, P);
    }
    #ifdef DEBUG
    gettimeofday(&t1, 0);
    printf("timeElapsed for compute_quadrupole(): %lu ms\n", (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000); 
    #endif

    // ---------------3. Traverse the tree and calculate force---------------
    #ifdef DEBUG
    gettimeofday(&t0, 0);
    #endif

    // create grouping 
    int n_crit = get_double("Tree.NCRIT", 1);
    int n_groups = 0;
    Node** group_nodes = (Node**) malloc(npart * sizeof(Node*));
    assign_group(T.root, P, n_crit, &n_groups, group_nodes);

    Zero_Force(P, npart);

    double theta = get_double("Tree.THETA", 0.01);
    double epsilon = get_double("BasicSetting.EPSILON", 1e-10);

    // calculate force with groups_xyzm[i]
    #ifdef OMP
    int OMP_CHUNK = get_int("Openmp.CHUNK", 1);
    omp_set_num_threads(OMP_NUM_THREADS);
    #pragma omp parallel for schedule(dynamic, OMP_CHUNK)
    #endif
    for (int g = 0; g < n_groups; g++) {
        int number_in_group = group_nodes[g]->npart;
        Coord4* groups_xyzm = (Coord4*) malloc(number_in_group * sizeof(Coord4));
        int* particle_indices = (int*) malloc(number_in_group * sizeof(int));
        int filled = 0;
        
        fill_xyzm(group_nodes[g], P, groups_xyzm, particle_indices, &filled);
        compute_interaction(T.root, P, groups_xyzm, group_nodes[g], particle_indices, npart, number_in_group, theta, poles, epsilon);
        free(groups_xyzm);
        free(particle_indices);
    }

    #ifdef DEBUG
    gettimeofday(&t1, 0);
    printf("timeElapsed for Tree_Force(): %lu ms\n", (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000); 
    #endif
    #ifdef DEBUG
    gettimeofday(&t0, 0);
    #endif
    Free_Tree(T.root);
    #ifdef DEBUG
    gettimeofday(&t1, 0);
    printf("timeElapsed for Free_Tree(): %lu ms\n", (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000); 
    #endif
    free(group_nodes);
}