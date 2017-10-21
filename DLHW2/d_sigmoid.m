function y = d_sigmoid(x)
y=sigmoid(x)*(1.0-sigmoid(x));
%y=1-(sigmoid(x))^2;
