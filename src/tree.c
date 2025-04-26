#include <stdlib.h>
#include <stdio.h>
#include "force.h"
#include "parameter.h"
#include "particle.h"
#include "tree.h"
#include "math.h"


Tree Initialize_Tree(Particle* P, int npart){
    Tree T;
    T.root = (Node*) malloc(sizeof(Node));
    T.root->npart = 0;
    T.root->parent = NULL;
    
    for (int i = 0; i < DIM; i++){
        T.root->box_max[i] = P[0].x[i];
        T.root->box_min[i] = P[0].x[i];
        for(int j = 1; j < npart; j++){
            T.root->box_max[i] = fmax(T.root->box_max[i], P[j].x[i]);
            T.root->box_min[i] = fmin(T.root->box_min[i], P[j].x[i]);
        }
    }

    return T;
}


int Which_Child(Node* node, Particle p){
    int i = 0;
    for(int j = 0; j < DIM; j++){
        if(p.x[j] > (node->box_min[j] + node->box_max[j]) / 2.0) i += (1<<j);
    }
    return i;
}

void Initialize_Children(Node* node){
    node->children = (Node**) malloc((1<<DIM) * sizeof(Node*)); // bit operation: 1<<n = 2^n
    for(int i = 0; i < 1<<DIM; i++){
        Node* newNode = (Node*) malloc(sizeof(Node));
        newNode->parent = node;
        newNode->npart = 0;
        newNode->i = -1;
        node->children[i] = newNode;
        for(int j = 0; j < DIM; j++){
            if((i / (1<<j)) % 2 == 0){
                newNode->box_min[j] = node->box_min[j];
                newNode->box_max[j] = (node->box_min[j] + node->box_max[j]) / 2.0;
            }
            else{
                newNode->box_max[j] = node->box_max[j];
                newNode->box_min[j] = (node->box_min[j] + node->box_max[j]) / 2.0;
            }
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
    
    return T;
}

void total_force(Particle* p, int npart)


