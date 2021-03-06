mod neuronal_network;
mod matrix;

use matrix::*;

// sepal length, width, petal length, width
// Iris virginica = 1 0 0
// Iris versicolor = 0 1 0
// Iris setosa = 0 0 1
const SAMPLE_DATA: [Precision; 150 * 7] = [
	5.1, 3.5, 1.4, 0.2, 0.0, 0.0, 1.0,
	4.9, 3.0, 1.4, 0.2, 0.0, 0.0, 1.0,
	4.7, 3.2, 1.3, 0.2, 0.0, 0.0, 1.0,
	4.6, 3.1, 1.5, 0.2, 0.0, 0.0, 1.0,
	5.0, 3.6, 1.4, 0.2, 0.0, 0.0, 1.0,
	5.4, 3.9, 1.7, 0.4, 0.0, 0.0, 1.0,
	4.6, 3.4, 1.4, 0.3, 0.0, 0.0, 1.0,
	5.0, 3.4, 1.5, 0.2, 0.0, 0.0, 1.0,
	4.4, 2.9, 1.4, 0.2, 0.0, 0.0, 1.0,
	4.9, 3.1, 1.5, 0.1, 0.0, 0.0, 1.0,
	5.4, 3.7, 1.5, 0.2, 0.0, 0.0, 1.0,
	4.8, 3.4, 1.6, 0.2, 0.0, 0.0, 1.0,
	4.8, 3.0, 1.4, 0.1, 0.0, 0.0, 1.0,
	4.3, 3.0, 1.1, 0.1, 0.0, 0.0, 1.0,
	5.8, 4.0, 1.2, 0.2, 0.0, 0.0, 1.0,
	5.7, 4.4, 1.5, 0.4, 0.0, 0.0, 1.0,
	5.4, 3.9, 1.3, 0.4, 0.0, 0.0, 1.0,
	5.1, 3.5, 1.4, 0.3, 0.0, 0.0, 1.0,
	5.7, 3.8, 1.7, 0.3, 0.0, 0.0, 1.0,
	5.1, 3.8, 1.5, 0.3, 0.0, 0.0, 1.0,
	5.4, 3.4, 1.7, 0.2, 0.0, 0.0, 1.0,
	5.1, 3.7, 1.5, 0.4, 0.0, 0.0, 1.0,
	4.6, 3.6, 1.0, 0.2, 0.0, 0.0, 1.0,
	5.1, 3.3, 1.7, 0.5, 0.0, 0.0, 1.0,
	4.8, 3.4, 1.9, 0.2, 0.0, 0.0, 1.0,
	5.0, 3.0, 1.6, 0.2, 0.0, 0.0, 1.0,
	5.0, 3.4, 1.6, 0.4, 0.0, 0.0, 1.0,
	5.2, 3.5, 1.5, 0.2, 0.0, 0.0, 1.0,
	5.2, 3.4, 1.4, 0.2, 0.0, 0.0, 1.0,
	4.7, 3.2, 1.6, 0.2, 0.0, 0.0, 1.0,
	4.8, 3.1, 1.6, 0.2, 0.0, 0.0, 1.0,
	5.4, 3.4, 1.5, 0.4, 0.0, 0.0, 1.0,
	5.2, 4.1, 1.5, 0.1, 0.0, 0.0, 1.0,
	5.5, 4.2, 1.4, 0.2, 0.0, 0.0, 1.0,
	4.9, 3.1, 1.5, 0.1, 0.0, 0.0, 1.0,
	5.0, 3.2, 1.2, 0.2, 0.0, 0.0, 1.0,
	5.5, 3.5, 1.3, 0.2, 0.0, 0.0, 1.0,
	4.9, 3.1, 1.5, 0.1, 0.0, 0.0, 1.0,
	4.4, 3.0, 1.3, 0.2, 0.0, 0.0, 1.0,
	5.1, 3.4, 1.5, 0.2, 0.0, 0.0, 1.0,
	5.0, 3.5, 1.3, 0.3, 0.0, 0.0, 1.0,
	4.5, 2.3, 1.3, 0.3, 0.0, 0.0, 1.0,
	4.4, 3.2, 1.3, 0.2, 0.0, 0.0, 1.0,
	5.0, 3.5, 1.6, 0.6, 0.0, 0.0, 1.0,
	5.1, 3.8, 1.9, 0.4, 0.0, 0.0, 1.0,
	4.8, 3.0, 1.4, 0.3, 0.0, 0.0, 1.0,
	5.1, 3.8, 1.6, 0.2, 0.0, 0.0, 1.0,
	4.6, 3.2, 1.4, 0.2, 0.0, 0.0, 1.0,
	5.3, 3.7, 1.5, 0.2, 0.0, 0.0, 1.0,
	5.0, 3.3, 1.4, 0.2, 0.0, 0.0, 1.0,
	7.0, 3.2, 4.7, 1.4, 0.0, 1.0, 0.0,
	6.4, 3.2, 4.5, 1.5, 0.0, 1.0, 0.0,
	6.9, 3.1, 4.9, 1.5, 0.0, 1.0, 0.0,
	5.5, 2.3, 4.0, 1.3, 0.0, 1.0, 0.0,
	6.5, 2.8, 4.6, 1.5, 0.0, 1.0, 0.0,
	5.7, 2.8, 4.5, 1.3, 0.0, 1.0, 0.0,
	6.3, 3.3, 4.7, 1.6, 0.0, 1.0, 0.0,
	4.9, 2.4, 3.3, 1.0, 0.0, 1.0, 0.0,
	6.6, 2.9, 4.6, 1.3, 0.0, 1.0, 0.0,
	5.2, 2.7, 3.9, 1.4, 0.0, 1.0, 0.0,
	5.0, 2.0, 3.5, 1.0, 0.0, 1.0, 0.0,
	5.9, 3.0, 4.2, 1.5, 0.0, 1.0, 0.0,
	6.0, 2.2, 4.0, 1.0, 0.0, 1.0, 0.0,
	6.1, 2.9, 4.7, 1.4, 0.0, 1.0, 0.0,
	5.6, 2.9, 3.6, 1.3, 0.0, 1.0, 0.0,
	6.7, 3.1, 4.4, 1.4, 0.0, 1.0, 0.0,
	5.6, 3.0, 4.5, 1.5, 0.0, 1.0, 0.0,
	5.8, 2.7, 4.1, 1.0, 0.0, 1.0, 0.0,
	6.2, 2.2, 4.5, 1.5, 0.0, 1.0, 0.0,
	5.6, 2.5, 3.9, 1.1, 0.0, 1.0, 0.0,
	5.9, 3.2, 4.8, 1.8, 0.0, 1.0, 0.0,
	6.1, 2.8, 4.0, 1.3, 0.0, 1.0, 0.0,
	6.3, 2.5, 4.9, 1.5, 0.0, 1.0, 0.0,
	6.1, 2.8, 4.7, 1.2, 0.0, 1.0, 0.0,
	6.4, 2.9, 4.3, 1.3, 0.0, 1.0, 0.0,
	6.6, 3.0, 4.4, 1.4, 0.0, 1.0, 0.0,
	6.8, 2.8, 4.8, 1.4, 0.0, 1.0, 0.0,
	6.7, 3.0, 5.0, 1.7, 0.0, 1.0, 0.0,
	6.0, 2.9, 4.5, 1.5, 0.0, 1.0, 0.0,
	5.7, 2.6, 3.5, 1.0, 0.0, 1.0, 0.0,
	5.5, 2.4, 3.8, 1.1, 0.0, 1.0, 0.0,
	5.5, 2.4, 3.7, 1.0, 0.0, 1.0, 0.0,
	5.8, 2.7, 3.9, 1.2, 0.0, 1.0, 0.0,
	6.0, 2.7, 5.1, 1.6, 0.0, 1.0, 0.0,
	5.4, 3.0, 4.5, 1.5, 0.0, 1.0, 0.0,
	6.0, 3.4, 4.5, 1.6, 0.0, 1.0, 0.0,
	6.7, 3.1, 4.7, 1.5, 0.0, 1.0, 0.0,
	6.3, 2.3, 4.4, 1.3, 0.0, 1.0, 0.0,
	5.6, 3.0, 4.1, 1.3, 0.0, 1.0, 0.0,
	5.5, 2.5, 4.0, 1.3, 0.0, 1.0, 0.0,
	5.5, 2.6, 4.4, 1.2, 0.0, 1.0, 0.0,
	6.1, 3.0, 4.6, 1.4, 0.0, 1.0, 0.0,
	5.8, 2.6, 4.0, 1.2, 0.0, 1.0, 0.0,
	5.0, 2.3, 3.3, 1.0, 0.0, 1.0, 0.0,
	5.6, 2.7, 4.2, 1.3, 0.0, 1.0, 0.0,
	5.7, 3.0, 4.2, 1.2, 0.0, 1.0, 0.0,
	5.7, 2.9, 4.2, 1.3, 0.0, 1.0, 0.0,
	6.2, 2.9, 4.3, 1.3, 0.0, 1.0, 0.0,
	5.1, 2.5, 3.0, 1.1, 0.0, 1.0, 0.0,
	5.7, 2.8, 4.1, 1.3, 0.0, 1.0, 0.0,
	6.3, 3.3, 6.0, 2.5, 1.0, 0.0, 0.0,
	5.8, 2.7, 5.1, 1.9, 1.0, 0.0, 0.0,
	7.1, 3.0, 5.9, 2.1, 1.0, 0.0, 0.0,
	6.3, 2.9, 5.6, 1.8, 1.0, 0.0, 0.0,
	6.5, 3.0, 5.8, 2.2, 1.0, 0.0, 0.0,
	7.6, 3.0, 6.6, 2.1, 1.0, 0.0, 0.0,
	4.9, 2.5, 4.5, 1.7, 1.0, 0.0, 0.0,
	7.3, 2.9, 6.3, 1.8, 1.0, 0.0, 0.0,
	6.7, 2.5, 5.8, 1.8, 1.0, 0.0, 0.0,
	7.2, 3.6, 6.1, 2.5, 1.0, 0.0, 0.0,
	6.5, 3.2, 5.1, 2.0, 1.0, 0.0, 0.0,
	6.4, 2.7, 5.3, 1.9, 1.0, 0.0, 0.0,
	6.8, 3.0, 5.5, 2.1, 1.0, 0.0, 0.0,
	5.7, 2.5, 5.0, 2.0, 1.0, 0.0, 0.0,
	5.8, 2.8, 5.1, 2.4, 1.0, 0.0, 0.0,
	6.4, 3.2, 5.3, 2.3, 1.0, 0.0, 0.0,
	6.5, 3.0, 5.5, 1.8, 1.0, 0.0, 0.0,
	7.7, 3.8, 6.7, 2.2, 1.0, 0.0, 0.0,
	7.7, 2.6, 6.9, 2.3, 1.0, 0.0, 0.0,
	6.0, 2.2, 5.0, 1.5, 1.0, 0.0, 0.0,
	6.9, 3.2, 5.7, 2.3, 1.0, 0.0, 0.0,
	5.6, 2.8, 4.9, 2.0, 1.0, 0.0, 0.0,
	7.7, 2.8, 6.7, 2.0, 1.0, 0.0, 0.0,
	6.3, 2.7, 4.9, 1.8, 1.0, 0.0, 0.0,
	6.7, 3.3, 5.7, 2.1, 1.0, 0.0, 0.0,
	7.2, 3.2, 6.0, 1.8, 1.0, 0.0, 0.0,
	6.2, 2.8, 4.8, 1.8, 1.0, 0.0, 0.0,
	6.1, 3.0, 4.9, 1.8, 1.0, 0.0, 0.0,
	6.4, 2.8, 5.6, 2.1, 1.0, 0.0, 0.0,
	7.2, 3.0, 5.8, 1.6, 1.0, 0.0, 0.0,
	7.4, 2.8, 6.1, 1.9, 1.0, 0.0, 0.0,
	7.9, 3.8, 6.4, 2.0, 1.0, 0.0, 0.0,
	6.4, 2.8, 5.6, 2.2, 1.0, 0.0, 0.0,
	6.3, 2.8, 5.1, 1.5, 1.0, 0.0, 0.0,
	6.1, 2.6, 5.6, 1.4, 1.0, 0.0, 0.0,
	7.7, 3.0, 6.1, 2.3, 1.0, 0.0, 0.0,
	6.3, 3.4, 5.6, 2.4, 1.0, 0.0, 0.0,
	6.4, 3.1, 5.5, 1.8, 1.0, 0.0, 0.0,
	6.0, 3.0, 4.8, 1.8, 1.0, 0.0, 0.0,
	6.9, 3.1, 5.4, 2.1, 1.0, 0.0, 0.0,
	6.7, 3.1, 5.6, 2.4, 1.0, 0.0, 0.0,
	6.9, 3.1, 5.1, 2.3, 1.0, 0.0, 0.0,
	5.8, 2.7, 5.1, 1.9, 1.0, 0.0, 0.0,
	6.8, 3.2, 5.9, 2.3, 1.0, 0.0, 0.0,
	6.7, 3.3, 5.7, 2.5, 1.0, 0.0, 0.0,
	6.7, 3.0, 5.2, 2.3, 1.0, 0.0, 0.0,
	6.3, 2.5, 5.0, 1.9, 1.0, 0.0, 0.0,
	6.5, 3.0, 5.2, 2.0, 1.0, 0.0, 0.0,
	6.2, 3.4, 5.4, 2.3, 1.0, 0.0, 0.0,
	5.9, 3.0, 5.1, 1.8, 1.0, 0.0, 0.0,
];


fn main() {
	let max_epochs: usize = 2000;
	let learn_rate: Precision = 0.05;
  let momentum: Precision = 0.01;
	let weight_decay: Precision = 0.0001;
	let min_mse: Precision = 0.020;
	let num_input: usize = 4;
	let num_hidden: usize = 7;
	let num_output: usize = 3;

	let mut nn = neuronal_network::NeuronalNetwork::new(num_input, num_hidden, num_output);
	nn.initialize_weights();

	let mut train_data = Matrix::from(&SAMPLE_DATA[0 .. 130 * 7], 7, 130);
	//neuronal_network::NeuronalNetwork::normalize(&mut train_data);

	nn.train(train_data.get_data(), max_epochs, learn_rate, momentum, weight_decay, min_mse);

	let mut test_data = Matrix::from(&SAMPLE_DATA[130 * 7 .. 130 * 7 + 20 * 7], 7, 20);
	//neuronal_network::NeuronalNetwork::normalize(&mut test_data);
	let accuracy = nn.accuracy(test_data.get_data());

	println!("Accuracy is {}", accuracy);
}
