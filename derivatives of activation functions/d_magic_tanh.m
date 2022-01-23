function d = d_magic_tanh(x)
    d = 1.7159 / 1.5 * sech(2 * x  / 3).^2;
end

