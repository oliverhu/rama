use std::{io::{BufReader, Read, Result}, fs::File};

pub trait FromBytes {
    fn from_bytes(bytes: [u8; 4]) -> Self;
}

impl FromBytes for f32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        f32::from_le_bytes(bytes)
    }
}

impl FromBytes for i32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        i32::from_le_bytes(bytes)
    }
}

pub(crate) fn read<T: FromBytes>(rd:&mut BufReader<File>) -> T {
    let mut buffer = [0u8; 4];
    rd.read_exact(&mut buffer).expect("error reading file");
    T::from_bytes(buffer)
}

pub fn read_vec<T: FromBytes>(rd: &mut BufReader<File>, size: usize) -> Vec<T> {
    (0..size).map(|_| read::<T>(rd)).collect()
}

pub fn read_n<R>(reader: R, bytes_to_read: usize) -> Result<Vec<u8>>
where
    R: Read,
{
    let mut buf = vec![];
    let mut chunk = reader.take(bytes_to_read as u64);
    let n = chunk.read_to_end(&mut buf)?;
    assert_eq!(bytes_to_read, n);
    Ok(buf)
}