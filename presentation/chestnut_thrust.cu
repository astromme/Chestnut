





int main() {
    // allocate three device_vectors with 10 elements
    thrust::device_vector<int> X(10);
    thrust::device_vector<int> Y(10);

    // fill X with twos
    thrust::fill(X.begin(), X.end(), 2);

    // fill Y with ones
    thrust::fill(Y.begin(), Y.end(), 1);

    // compute X = X + 1
    thrust::transform(X.begin(), X.end(), 
        Y.begin(), X.begin(), 
        thrust::plus<int>());

    thrust::host_vector<int> host = X;

    return 0;    
}




                                

