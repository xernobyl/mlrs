use std::ops::{Index, IndexMut};

pub type Precision = f64;

pub struct Matrix {
	data: Vec<Precision>,
	columns: usize,
	rows: usize,
}

impl Matrix {
	pub fn new(columns: usize, rows: usize) -> Self {
		Self {
			data: vec![0.0; columns * rows],
			columns: columns,
			rows: rows,
		}
	}

	pub fn from(data: &[Precision], columns: usize, rows: usize) -> Self {
		Self {
			data: Vec::from(data),
			columns: columns,
			rows: rows,
		}
	}

	pub fn get_columns(&self) -> usize {
		self.columns
	}

	pub fn get_rows(&self) -> usize {
		self.rows
	}	

	pub fn get_data(&self) -> &[Precision] {
		self.data.as_slice()
	}
}

impl Index<usize> for Matrix {
	type Output = [Precision];

	fn index(&self, i: usize) -> &[Precision] {
		&self.data[i * self.columns .. i * self.columns + self.columns]
	}
}

impl IndexMut<usize> for Matrix {
	fn index_mut(&mut self, i: usize) -> &mut [Precision] {
		&mut self.data[i * self.columns .. i * self.columns + self.columns]
	}
}
