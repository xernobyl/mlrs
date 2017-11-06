/*
Based on http://quaetrix.com/Build2014.html

From the original:
"This is an enhanced neural network. It is fully-connected
and feed-forward. The training algorithm is back-propagation
with momentum and weight decay. The input data is normalized
so training is quite fast."
*/

extern crate rand;
use self::rand::Rng;
use matrix::*;
use std::iter::FromIterator;

#[allow(unused_mut)]
#[allow(dead_code)]

pub struct NeuronalNetwork {
	rng: Box<rand::Rng>,

	num_input: usize,
	num_hidden: usize,
	num_output: usize,
	num_weights: usize,

	inputs: Box<[Precision]>,

	ih_weights: Matrix, // input-hidden
	//ih_weights_stride: usize,
	h_biases: Box<[Precision]>,
	h_outputs: Box<[Precision]>,

	ho_weights: Matrix, // hidden-output
	//ho_weights_stride: usize,
	o_biases: Box<[Precision]>,
	outputs: Box<[Precision]>,

	// back-prop specific arrays (these could be local to method UpdateWeights)
	o_grads: Box<[Precision]>, // output gradients for back-propagation
	h_grads: Box<[Precision]>, // hidden gradients for back-propagation

	// back-prop momentum specific arrays (could be local to method Train)
	ih_prev_weights_delta: Matrix,  // for momentum with back-propagation
	//ih_prev_weights_delta_stride: usize,
	h_prev_biases_delta: Box<[Precision]>,
	ho_prev_weights_delta: Matrix,
	//ho_prev_weights_delta_stride: usize,
	o_prev_biases_delta: Box<[Precision]>,
}

impl NeuronalNetwork {
	fn boxed_slice(init: Precision, size: usize) -> Box<[Precision]> {
		vec![init; size].into_boxed_slice()
	}

	pub fn new(num_input: usize, num_hidden: usize, num_output: usize) -> Self {
		Self {
			num_weights: (num_input * num_hidden) + (num_hidden * num_output) + num_hidden + num_output,
			
			rng: Box::new(rand::weak_rng()),

			num_input: num_input,
			num_hidden: num_hidden,
			num_output: num_output,

			inputs: Self::boxed_slice(0.0, num_input),

			ih_weights: Matrix::new(num_hidden, num_input),
			h_biases: Self::boxed_slice(0.0, num_hidden),
			h_outputs: Self::boxed_slice(0.0, num_hidden),

			ho_weights: Matrix::new(num_output, num_hidden),
			o_biases: Self::boxed_slice(0.0, num_output),

			outputs: Self::boxed_slice(0.0, num_output),

			// back-prop related arrays below
			h_grads: Self::boxed_slice(0.0, num_hidden),
			o_grads: Self::boxed_slice(0.0, num_output),

			ih_prev_weights_delta: Matrix::new(num_hidden, num_input),
			h_prev_biases_delta: Self::boxed_slice(0.0, num_hidden),
			ho_prev_weights_delta: Matrix::new(num_output, num_hidden),
			o_prev_biases_delta: Self::boxed_slice(0.0, num_output),
		}
	}

	// copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
	fn set_weights(&mut self, weights: Box<[Precision]>) {
		if weights.len() != self.num_weights {
			panic!("Bad weights array length (got {}, expected {}).", weights.len(), self.num_weights);
		}

		let mut k = 0;	// points into weights param
		
		for i in 0..self.ih_weights.get_rows() {
			for j in 0..self.ih_weights.get_columns() {
				self.ih_weights[i][j] = weights[k];
				k += 1;
			}
		}

		for i in 0..self.num_hidden {
			self.h_biases[i] = weights[k];
			k += 1;
		}

		for i in 0..self.ho_weights.get_rows() {
			for j in 0..self.ho_weights.get_columns() {
				self.ho_weights[i][j] = weights[k];
				k += 1;
			}
		}

		for i in 0..self.num_output {
			self.o_biases[i] = weights[k];
			k += 1;
		}
	}

	// initialize weights and biases to small random values
	pub fn initialize_weights(&mut self) {
		let mut initial_weights = Self::boxed_slice(0.0, self.num_weights);
		let lo = -0.01;
		let hi = 0.01;

		for i in 0..initial_weights.len() {
			initial_weights[i] = self.rng.gen_range(lo, hi);
		}

		self.set_weights(initial_weights);
	}

	// returns the current set of wweights, presumably after training
	fn get_weights(&self) -> Box<[Precision]> {
		let mut result = Self::boxed_slice(0.0, self.num_weights);
		let mut k = 0;

		for i in 0..self.ih_weights.get_rows() {
			for j in 0..self.ih_weights.get_columns() {
				result[k] = self.ih_weights[i][j];
				k += 1;
			}
		}

		for i in 0..self.h_biases.len() {
			result[k] = self.h_biases[i];
			k += 1;
		}

		for i in 0..self.ho_weights.get_rows() {
			for j in 0..self.ho_weights.get_columns() {
				result[k] = self.ho_weights[i][j];
				k += 1;
			}
		}

		for i in 0..self.o_biases.len() {
			result[k] = self.o_biases[i];
			k += 1;
		}

		result
	}

	fn compute_outputs(&mut self, x_values: &[Precision]) -> Box<[Precision]> {
		let mut h_sums = Self::boxed_slice(0.0, self.num_hidden);	// hidden nodes sums scratch array
		let mut o_sums = Self::boxed_slice(0.0, self.num_output);	// output nodes sums

		for i in 0..self.num_input {	// copy x-values to inputs
			self.inputs[i] = x_values[i];
		}

		for j in 0..self.ih_weights.get_columns() {			// compute i-h sum of weights * inputs
			for i in 0..self.ih_weights.get_rows() {
				h_sums[j] += self.inputs[i] * self.ih_weights[i][j];	// note +=
			}
		}

		for i in 0..self.num_hidden {			// add biases to input-to-hidden sums
			h_sums[i] += self.h_biases[i];
		}

		for i in 0..self.num_hidden {	// apply activation
			self.h_outputs[i] = Self::tanh(h_sums[i])	// hard-coded
		}

		for j in 0..self.ho_weights.get_columns() {
			for i in 0..self.ho_weights.get_rows() {	// compute h-o sum of weights * h_outputs
				o_sums[j] += self.h_outputs[i] * self.ho_weights[i][j];	// note +=
			}
		}

		for i in 0..self.num_output {	// add biases to input-to-hidden sums
			o_sums[i] += self.o_biases[i];
		}

		let soft_out = Self::soft_max(o_sums);	// softmax activation does all outputs at once for efficiency
		self.outputs.copy_from_slice(&soft_out);	// could define a GetOutputs method instead
		self.outputs.clone()
	}

	fn tanh(x: Precision) -> Precision {
		// approximation is correct to 30 decimals
		if x < -20.0 {
			-1.0
		}
		else if x > 20.0 {
			1.0
		}
		else {
			Precision::tanh(x)
		}
	}

	fn soft_max(o_sums: Box<[Precision]>) -> Box<[Precision]> {
		// determine max output sum
		// does all output nodes at once so scale doesn't have to be re-computed each time
		let mut max = o_sums[0];

		// TODO: replace with map and filter
		for i in 0..o_sums.len() {
			if o_sums[i] > max {
				max = o_sums[i];
			}
		}

		// determine scaling factor -- sum of exp(each val - max)
		let mut scale = 0.0;
		for i in 0..o_sums.len() {
			scale += Precision::exp(o_sums[i] - max);
		}

		let mut result = Self::boxed_slice(0.0, o_sums.len());
		for i in 0..o_sums.len() {
			result[i] = Precision::exp(o_sums[i] - max) / scale;
		}

		result // now scaled so that xi sum to 1.0
	}

	fn update_weights(&mut self, t_values: &Box<[Precision]>, learn_rate: Precision, momentum: Precision, weight_decay: Precision) {
		// update the weights and biases using back-propagation, with target values, eta (learning rate),
		// alpha (momentum).
		// assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays
		// and matrices have values (other than 0.0)
		if t_values.len() != self.num_output {
			panic!("target values not same Length as output in UpdateWeights");
		}

		// 1. compute output gradients
		for i in 0..self.o_grads.len() {
			// derivative of softmax = (1 - y) * y (same as log-sigmoid)
			let derivative = (1.0 - self.outputs[i]) * self.outputs[i];
			// 'mean squared error version' includes (1-y)(y) derivative
			self.o_grads[i] = derivative * (t_values[i] - self.outputs[i]);
		}

		// 2. compute hidden gradients
		for i in 0..self.ho_weights.get_rows() {
			// derivative of tanh = (1 - y) * (1 + y)
			let derivative = (1.0 - self.h_outputs[i]) * (1.0 + self.h_outputs[i]);
			let mut sum = 0.0;
			for j in 0..self.ho_weights.get_columns() {
				let x = self.o_grads[j] * self.ho_weights[i][j];
				sum += x;
			}
			self.h_grads[i] = derivative * sum;
		}

		// 3a. update hidden weights (gradients must be computed right-to-left but weights
		// can be updated in any order)
		for i in 0..self.ih_weights.get_rows() {
			for j in 0..self.ih_weights.get_columns() {
				let delta = learn_rate * self.h_grads[j] * self.inputs[i]; // compute the new delta
				self.ih_weights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
				// now add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
				self.ih_weights[i][j] += momentum * self.ih_prev_weights_delta[i][j];
				self.ih_weights[i][j] -= weight_decay * self.ih_weights[i][j]; // weight decay
				self.ih_prev_weights_delta[i][j] = delta; // don't forget to save the delta for momentum
			}
		}

		// 3b. update hidden biases
		for i in 0..self.h_biases.len() {
			let delta = learn_rate * self.h_grads[i] * 1.0; // t1.0 is constant input for bias; could leave out
			self.h_biases[i] += delta;
			self.h_biases[i] += momentum * self.h_prev_biases_delta[i]; // momentum
			self.h_biases[i] -= weight_decay * self.h_biases[i]; // weight decay
			self.h_prev_biases_delta[i] = delta; // don't forget to save the delta
		}

		// 4. update hidden-output weights
		for i in 0..self.ho_weights.get_rows() {
			for j in 0..self.ho_weights.get_columns() {
				// see above: h_outputs are inputs to the nn outputs
				let delta = learn_rate * self.o_grads[j] * self.h_outputs[i];
				self.ho_weights[i][j] += delta;
				self.ho_weights[i][j] += momentum * self.ho_prev_weights_delta[i][j]; // momentum
				self.ho_weights[i][j] -= weight_decay * self.ho_weights[i][j]; // weight decay
				self.ho_prev_weights_delta[i][j] = delta; // save
			}
		}


		// 4b. update output biases
		for i in 0..self.o_biases.len() {
			let delta = learn_rate * self.o_grads[i] * 1.0;
			self.o_biases[i] += delta;
			self.o_biases[i] += momentum * self.o_prev_biases_delta[i]; // momentum
			self.o_biases[i] -= weight_decay * self.o_biases[i]; // weight decay
			self.o_prev_biases_delta[i] = delta; // save
		}
	} // update_weights

	pub fn train(&mut self, train_data: &[Precision], max_epochs: usize, learn_rate: Precision, momentum: Precision, weight_decay: Precision, min_mse: Precision) {
		// train a back-prop style NN classifier using learning rate and momentum
		// weight decay reduces the magnitude of a weight value over time unless that value
		// is constantly increased

		let train_data_stride = self.num_input + self.num_output;

		let mut epoch = 0;
		let mut x_values = Self::boxed_slice(0.0, self.num_input); // input
		let mut t_values = Self::boxed_slice(0.0, self.num_output); // target values

		let mut sequence = Vec::from_iter(0..train_data.len() / train_data_stride);
		for i in 0..sequence.len() {
			sequence[i] = i;
		}

		while epoch < max_epochs {
			println!("Epoch {}", epoch);
			let mse = self.mean_squared_error(&train_data);
			println!("Mean squared error {}", mse);

			if mse < min_mse {
				break;
			}

			self.rng.shuffle(&mut sequence);	// visit each training data in random order
			
			for i in 0..train_data.len() / train_data_stride {
				let idx = sequence[i];
				x_values.copy_from_slice(&train_data [(idx * train_data_stride)..(idx * train_data_stride + self.num_input)]);
				t_values.copy_from_slice(&train_data [(idx * train_data_stride + self.num_input)..(idx * train_data_stride + self.num_input + self.num_output)]);
				self.compute_outputs(&x_values); // copy x_values in, compute outputs (store them internally)
				self.update_weights(&t_values, learn_rate, momentum, weight_decay); // find better weights
			} // each training tuple
			epoch += 1;
		}
	}

	fn mean_squared_error(&mut self, train_data: &[Precision]) -> Precision { // used as a training stopping condition
		let train_data_stride = self.num_input + self.num_output;

		// average squared error per training tuple
		let mut sum_squared_error = 0.0;
		let mut x_values = Self::boxed_slice(0.0, self.num_input); // first num_input values in train_data
		let mut t_values = Self::boxed_slice(0.0, self.num_output); // last num_output values

		// walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
		for i in 0..train_data.len() / train_data_stride {
			x_values.copy_from_slice(&train_data [(i * train_data_stride)..(i * train_data_stride + self.num_input)]);
			t_values.copy_from_slice(&train_data [(i * train_data_stride + self.num_input)..(i * train_data_stride + self.num_input + self.num_output)]);	// get target values
			let y_values = self.compute_outputs(&x_values); // compute output using current weights

			for j in 0..self.num_output {
				let err = t_values[j] - y_values[j];
				sum_squared_error += err * err;
			}
		}

		sum_squared_error / train_data.len() as Precision
	}

	pub fn accuracy(&mut self, test_data: &[Precision]) -> Precision {
		let test_data_stride = self.num_input + self.num_output;

		// percentage correct using winner-takes all
		let mut num_correct = 0;
		let mut num_wrong = 0;

		let mut x_values = Self::boxed_slice(0.0, self.num_input);	// inputs
		let mut t_values = Self::boxed_slice(0.0, self.num_output);	// targets

		for i in 0..test_data.len() / test_data_stride {
			x_values.copy_from_slice(&test_data[i * test_data_stride..i * test_data_stride + self.num_input]);
			t_values.copy_from_slice(&test_data[i * test_data_stride + self.num_input..i * test_data_stride + self.num_input + self.num_output]);

			let y_values = self.compute_outputs(&x_values);
			let max_index = Self::max_index(y_values); // which cell in y_values has largest value?

			if t_values[max_index] == 1.0 { // ugly. consider AreEqual(double x, double y)
				num_correct += 1;
			} else {
				num_wrong += 1;
			}
		}

		Precision::from(num_correct) / Precision::from(num_correct + num_wrong) // ugly 2 - check for divide by zero
	}

	fn max_index(vector: Box<[Precision]>) -> usize {
		let mut big_index: usize = 0;
		let mut biggest_val = vector[0];
		for i in 1..vector.len() {
			if vector[i] > biggest_val {
				biggest_val = vector[i];
				big_index = i;
			}
		}
		big_index
	}

	pub fn normalize(data: &mut Matrix) {
		for col in 0..data.get_columns() {
			let mut sum = 0.0;
			for row in 0..data.get_rows() {
				sum += data[row][col]
			}
			let mean = sum / data.get_rows() as f64;
			sum = 0.0;
			for row in 0..data.get_rows() {
				sum += (data[row][col] - mean) * (data[row][col] - mean);
			}
			let sd = f64::sqrt(sum / (data.get_rows() - 1) as f64);
			for row in 0..data.get_rows() {
				data[row][col] = (data[row][col] - mean) / sd;
			}
		}
	}
}