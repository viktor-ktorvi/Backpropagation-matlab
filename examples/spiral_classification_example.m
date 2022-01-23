clc;
close all;
clear variables;

set(groot, 'defaulttextinterpreter', 'latex');  
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');  
set(groot, 'defaultLegendInterpreter', 'latex');
%% Generating data
rng(26); % init random seed

N = 500;
t = rand(1, N);
a = 1;
b = 10;

sigma_rho = a / 10;
sigma_phi = b / 100;

rho = a * sqrt(t) + sigma_rho * randn(1, length(t));
phi = b * sqrt(t) + sigma_phi * randn(1, length(t));

x1 = rho .* cos(phi);
y1 = rho .* sin(phi);

x2 = rho .* cos(phi + pi);
y2 = rho .* sin(phi + pi);
%% Plot
figure
hold on;
plot(x1, y1, 'b*')
plot(x2, y2, 'r*')
title('Data')
xlabel('$x_1$')
ylabel('$x_2$')
legend('class 1', 'class 2')

%% Preparing the data and shuffling

data = [x1, x2; y1, y2];
labels = [ones(1, N), zeros(1, N); zeros(1, N), ones(1, N)];

permutation = randperm(size(labels, 2));

data = data(:, permutation);
labels = labels(:, permutation);

%% Splitting
split_index = round(0.7 * size(data, 2));

data_train = data(:, 1:split_index);
labels_train = labels(:, 1:split_index);

data_test = data(:, split_index + 1:end);
labels_test = labels(:, split_index + 1:end);
%% Normalizing

train_mean = mean(data_train, 2);
train_std = std(data_train');

data_train_norm = (data_train - train_mean) ./ train_std';
data_test_norm = (data_test - train_mean) ./ train_std';        

figure
hold on
axis equal
scatter(data_train_norm(1, labels_train(1, :) == 1), data_train_norm(2, labels_train(1, :) == 1), 'bo')
scatter(data_train_norm(1, labels_train(1, :) == 0), data_train_norm(2, labels_train(1, :) == 0), 'ro')

title('Normalized training data')
xlabel('$x_1$')
ylabel('$x_2$')
legend('class A', 'class B')

X = data_train_norm;
%% Network parameters

learning_rate = 0.001;
lambda = 0.00001;
epochs = 1000;
batch_size = 64;

% gradient clipping
clip_flg = 1;
clip_norm = 1;
clip_val = 1;

% weight initialization
init_options.name = "xavier";
init_options.distribution = "gauss";

%% Network

% Still having problems with NaNs and exploding gradient

% layer def
layer_sizes = [size(X, 1); 10; 20; 10; size(labels, 1)];

% activation functions and their derivatives by layer
activations = {@magic_tanh; @magic_tanh; @magic_tanh; @sigmoid};
d_activations = {@d_magic_tanh; @d_magic_tanh; @d_magic_tanh; @d_sigmoid};

% activations = {@relu; @relu; @relu; @sigmoid};
% d_activations = {@d_relu; @d_relu; @d_relu; @d_sigmoid};

% error derivative
dEdy = @d_binary_cross_entrpoy;

model = MultilayerPerceptron(layer_sizes, activations, d_activations, init_options, dEdy, lambda, clip_flg, clip_norm, clip_val);
% model.to_gpu();
%% Training
loss_array = zeros(epochs, 1);
tic
for i = 1:epochs
    loss = model.train(X, labels_train, batch_size, learning_rate, @binary_cross_entropy, 0) / N;
    
    if mod(i, 50) == 0
        fprintf("Epoch = %d Error = %2.5f\n", i, loss)
    end
    loss_array(i) = loss;
end
toc

figure
plot(loss_array)
title('Training loss')
xlabel('Epoch [num]')
ylabel('Loss')

%% Results
range = 1:size(labels_test, 2); % which samples to test on
[accuracy, prediction, ground_truth] = classification_accuracy(model, data_test_norm, labels_test, range);
fprintf("\nTest accuracy = %2.2f %%\n", 100 * accuracy)

figure
axis equal
hold on
scatter(data_test(1, prediction == 1), data_test(2, prediction == 1), 'bx')
scatter(data_test(1, prediction == 2), data_test(2, prediction == 2), 'rx')

title('Test results')
xlabel('$x_1$')
ylabel('$x_2$')
legend('predicted 1', 'predicted 2')