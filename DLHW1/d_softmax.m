function y = d_softmax(x)
y=softmax(x)*(1-softmax(x));