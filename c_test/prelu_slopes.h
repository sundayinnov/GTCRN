#ifndef PRELU_SLOPES_H
#define PRELU_SLOPES_H
static const float prelu_slopes[15] = {
    0.07284414f,  // [0] node_prelu
    0.19165090f,  // [1] node_prelu_1
    0.18543091f,  // [2] node_prelu_2
    0.37440112f,  // [3] node_prelu_3
    0.63517940f,  // [4] node_prelu_4
    -0.00254873f,  // [5] node_prelu_5
    0.41709870f,  // [6] node_prelu_6
    0.31753048f,  // [7] node_prelu_7
    0.36614743f,  // [8] node_prelu_8
    -0.26041290f,  // [9] node_prelu_9
    0.00004147f,  // [10] node_prelu_10
    0.43424520f,  // [11] node_prelu_11
    0.04158230f,  // [12] node_prelu_12
    0.42213863f,  // [13] node_prelu_13
    0.00147689f,  // [14] node_prelu_14
};
#endif
