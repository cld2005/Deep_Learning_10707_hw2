function y = testing_func(x,x2,momentum,learning_rate,batch_size)
    y=momentum*x+learning_rate*(x2/batch_size);
end