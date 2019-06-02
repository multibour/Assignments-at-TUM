#include "familytree.h"

int _traverse(tree*);


int traverse(tree *node, int numThreads){
    int iq;

    #pragma omp parallel num_threads(numThreads) shared(iq, node)
    {
        #pragma omp single
        iq = _traverse(node);
    }

    return iq;
}


int _traverse(tree* node){
    if (node == NULL)
        return 0;
    
    int father_iq, mother_iq;
    
    #pragma omp task shared(father_iq)
    father_iq = _traverse(node->father);

    #pragma omp task shared(mother_iq)
    mother_iq = _traverse(node->mother);

    #pragma omp taskwait

    node->IQ = compute_IQ(node->data, father_iq, mother_iq);

    #pragma omp critical
    genius[node->id] = node->IQ;

    return node->IQ;
}
