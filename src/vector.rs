pub struct Vector {
	data: Vec<Precision>,
}

impl Vector {
	fn new(n: usize) -> Self {
		Self {
			data: vec![0.0; n]
		}
	}
}
