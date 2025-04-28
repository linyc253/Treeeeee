// This tree algorithm is implemented by following the psuedo code from: 
//   https://web.archive.org/web/20160510023001/http://www.eecs.berkeley.edu/~demmel/cs267/lecture26/lecture26.html
// I preserve the psuedo code as comments for better understanding about what I was doing:)
#include <stdlib.h>
#include <stdio.h>
#include "parameter.h"
#include "particle.h"
#include "tree.h"
#include "math.h"
#include "dsyevh3.h"

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
        T.root->x[i] = (T.box_min[i] + T.box_max[i]) / 2.0; // temporarily set to center
    }

    return T;
}

// find belonging child for particle
int Which_Child(Node* node, Particle p){
    int i = 0;
    for(int j = 0; j < DIM; j++){
        if(p.x[j] > node->x[j]) i += (1<<j);
    }
    return i;
}

void Initialize_Children(Node* node){
    node->children = (Node**) malloc((1<<DIM) * sizeof(Node*)); // bit operation: 1<<n = 2^n
    for(int i = 0; i < 1<<DIM; i++){
        Node* newNode = (Node*) malloc(sizeof(Node));
        newNode->parent = node;
        newNode->children = NULL;
        newNode->npart = 0;
        newNode->i = -1;
        newNode->D = node->D / 2.0;
        node->children[i] = newNode;
        for(int j = 0; j < DIM; j++){
            if((i / (1<<j)) % 2 == 0) newNode->x[j] = node->x[j] - newNode->D / 2.0;
            else newNode->x[j] = node->x[j] + newNode->D / 2.0;
        }
    }
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
void Tree_Insert(Node* node, Particle* P, int i){
    if(node->npart > 1){
        Tree_Insert(node->children[Which_Child(node, P[i])], P, i);
    }
    else if(node->npart == 1){
        Initialize_Children(node);
        Tree_Insert(node->children[Which_Child(node, P[i])], P, i);
        Tree_Insert(node->children[Which_Child(node, P[node->i])], P, node->i);
        node->i = -1;
    }
    else{ // node->npart == 0
        node->i = i;
    }
    node->npart++;
}

void Clear_Empty(Node* node){
    // if(node->npart == 0) return
    for(int i = 0; i < 1<<DIM; i++){
        if(node->children[i]->npart == 0){
            free(node->children[i]);
            node->children[i] = NULL;
        }
        else if(node->children[i]->npart > 1){
            Clear_Empty(node->children[i]);
        }
    }
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
Tree Tree_Build(Particle* P, int npart){
    Tree T = Initialize_Tree(P, npart);
    for (int i = 0; i < npart; i++){
        Tree_Insert(T.root, P, i);
    }
    Clear_Empty(T.root);
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
int Compute_m_and_x(Node* node, Particle* P){
    if(node == NULL) return -1;
    
    if(node->npart == 1){
        node->m = P[node->i].m;
        for(int i = 0; i < DIM; i++) node->x[i] = P[node->i].x[i];
    }
    else{
        node->m = 0.0;
        for(int i = 0; i < DIM; i++) node->x[i] = 0.0;

        for(int i = 0; i < 1<<DIM; i++){
            if(Compute_m_and_x(node->children[i], P) != -1){
                node->m += node->children[i]->m;
                for(int j = 0; j < DIM; j++) node->x[j] += node->children[i]->m * node->children[i]->x[j];
            }
        }
        for(int i = 0; i < DIM; i++) node->x[i] /= node->m;
    }
    return 0;
}

// compute quadrupole tensor and pseudoparticle positions
#ifdef __cplusplus
extern "C" {
#endif
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
                // #ifdef DEBUG
                // printf("child quadrupole = \n(%f, %f, %f)\n(%f, %f, %f)\n(%f, %f, %f)\n", child->p2_x[0][0], child->p2_x[0][1], child->p2_x[0][2],
                //     child->p2_x[1][0], child->p2_x[1][1], child->p2_x[1][2],
                //     child->p2_x[2][0], child->p2_x[2][1], child->p2_x[2][2]);
                // #endif                
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
        // #ifdef DEBUG
        // printf("quadrupole tensor = \n(%f, %f, %f)\n(%f, %f, %f)\n(%f, %f, %f)\n", quad_tensor[0][0], quad_tensor[0][1], quad_tensor[0][2],
        //     quad_tensor[1][0], quad_tensor[1][1], quad_tensor[1][2],
        //     quad_tensor[2][0], quad_tensor[2][1], quad_tensor[2][2]);
        // #endif
        // compute eigenvalue of quadrupole tensor
        double eigval[3];
        double eigvec[3][3];
        dsyevh3(quad_tensor, eigvec, eigval);
        #ifdef DEBUG
        printf("eigenvector unsorted = \n(%f, %f, %f)\n(%f, %f, %f)\n(%f, %f, %f)\n", eigvec[0][0], eigvec[0][1], eigvec[0][2],
                                                                                      eigvec[1][0], eigvec[1][1], eigvec[1][2],
                                                                                      eigvec[2][0], eigvec[2][1], eigvec[2][2]);
        #endif

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
        
        // #ifdef DEBUG
        // printf("eigenvalue = (%f, %f, %f)\n", eigval[0], eigval[1], eigval[2]);
        // #endif
        // #ifdef DEBUG
        // printf("eigenvector = \n(%f, %f, %f)\n(%f, %f, %f)\n(%f, %f, %f)\n", eigvec[0][0], eigvec[0][1], eigvec[0][2],
        //                                                                               eigvec[1][0], eigvec[1][1], eigvec[1][2],
        //                                                                               eigvec[2][0], eigvec[2][1], eigvec[2][2]);
        // #endif

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

        // #ifdef DEBUG
        // printf("alpha = %f, beta = %f\n", alpha, beta);
        // #endif

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
        // #ifdef DEBUG
        // printf("pseudo-particle positions = \n(%f, %f, %f)\n(%f, %f, %f)\n(%f, %f, %f)\n", node->p2_x[0][0], node->p2_x[0][1], node->p2_x[0][2],
        //     node->p2_x[1][0], node->p2_x[1][1], node->p2_x[1][2],
        //     node->p2_x[2][0], node->p2_x[2][1], node->p2_x[2][2]);
        // #endif
    }
    return 0;
}
#ifdef __cplusplus
}
#endif

// Calculate gravitational force by (G = 1)
//  a.f += -(a.m * b.m / (|r|^2 + epsilon^2)^{3/2}) r
// where the vector r = a.x - b.x
void add_cell_particle_force(Particle* a, Node* b, double epsilon){
    double m_a = a->m;
    double m_b = b->m;
    double dx[DIM];
    double r_sq = 0.0; // r^2
    for (int i = 0; i < DIM; i++) {
        dx[i] = a->x[i] - b->x[i];
        r_sq += dx[i]*dx[i];
    }

    // calculate gravitational force (assume G = 1.0) 
    double F_mag = -(m_a * m_b) / pow(r_sq + epsilon*epsilon, 1.5);

    for (int i = 0; i < DIM; i++) {
        a->f[i] += F_mag * dx[i];
    }
}

// calculate gravatational force using pseudo-particles
void add_cell_particle_force_quad(Particle* particle, Node* node, double epsilon){
    for (int pp = 0; pp < 3; pp++) {
        double r[DIM];
        double r_norm = 0;
        for (int i = 0; i < DIM; i++) {
            r[i] = particle->x[i] - node->p2_x[pp][i];
            r_norm += pow(r[i], 2);
        }
        for (int i = 0; i < DIM; i++) {
            particle->f[i] += -particle->m * node->m / 3 * r[i] / pow(r_norm + pow(epsilon, 2), 1.5);
        }
    }
    // #ifdef DEBUG
    // printf("force = (%f, %f, %f)\n", particle->f[0], particle->f[1], particle->f[2]);
    // for (int i = 0; i < DIM; i++) {
    //     particle->f[i] = 0;
    // }
    // #endif
}

// Calculate gravitational force by (G = 1)
//  force = -(a.m * b.m / (|r|^2 + epsilon^2)^{3/2}) r
// where the vector r = a.x - b.x
double cell_particle_distance(Particle* a, Node* b){
    double r_sq = 0.0; // r^2
    for (int i = 0; i < DIM; i++) {
        double dx = a->x[i] - b->x[i];
        r_sq += dx * dx;
    }
    return pow(r_sq, 0.5);
}

// function f = TreeForce(i,n)
//           ... Compute gravitational force on particle i 
//           ... due to all particles in the box at n
//           f = 0
//           if n contains one particle
//               f = force computed using formula (*) above
//           else 
//               r = distance from particle i to 
//                      center of mass of particles in n
//               D = size of box n
//               if D/r < theta
//                   compute f using formula (*) above
//               else
//                   for all children c of n
//                       f = f + TreeForce(i,c)
//                   end for
//               end if
//           end if
void Tree_Force(Node* node, Particle* P, int i, double THETA, double epsilon){
    if(node == NULL || node->i == i) return;
    
    if(node->npart == 1){
        #ifdef DEBUG
        add_cell_particle_force_quad(&P[i], node, epsilon);
        #endif
        add_cell_particle_force(&P[i], node, epsilon);
    }
    else{
        double r = cell_particle_distance(&P[i], node);
        if(node->D / r < THETA){
            if (get_int("Tree.POLES", 1) == 1) {
                add_cell_particle_force(&P[i], node, epsilon);
            }
            else if (get_int("Tree.POLES", 1) == 2) {
                add_cell_particle_force_quad(&P[i], node, epsilon);
            }
        }
        else{            for(int j = 0; j < 1<<DIM; j++){
                Tree_Force(node->children[j], P, i, THETA, epsilon);
            }
        }
    }
}

// Set all the force to zero
void Zero_Force(Particle* P, int npart){
    for(int i = 0; i < npart; i++){
        for(int j = 0; j < DIM; j++) P[i].f[j] = 0.0;
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
    // 1. Build the Tree
#ifdef DEBUG
    struct timeval t0, t1;
    gettimeofday(&t0, 0);
#endif
    Tree T = Tree_Build(P, npart);
#ifdef DEBUG
    gettimeofday(&t1, 0);
    printf("timeElapsed for Tree_Build(): %lu ms\n", (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000); 
#endif
    // 2. Compute the mass & center-of-mass
#ifdef DEBUG
    gettimeofday(&t0, 0);
#endif
    Compute_m_and_x(T.root, P);
    if (get_int("Tree.POLES", 1) == 2) {
        compute_quadrupole(T.root, P);
    }
#ifdef DEBUG
    gettimeofday(&t1, 0);
    printf("timeElapsed for Compute_m_and_x(): %lu ms\n", (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000); 
#endif

    // 3. Traverse the tree and calculate force
#ifdef DEBUG
    gettimeofday(&t0, 0);
#endif
    Zero_Force(P, npart);
    double epsilon = get_double("BasicSetting.epsilon", 1e-10);
    double THETA = get_double("Tree.THETA", 0.01);
    for(int i = 0; i < npart; i++) {
        Tree_Force(T.root, P, i, THETA, epsilon);
    }
#ifdef DEBUG
    gettimeofday(&t1, 0);
    printf("timeElapsed for Tree_Force(): %lu ms\n", (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000); 
#endif
    Free_Tree(T.root);
}
