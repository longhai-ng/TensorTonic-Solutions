import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    pe = np.zeros((seq_length, d_model))
    
    # Create a column vector for positions: shape (seq_length, 1)
    pos = np.arange(seq_length).reshape(-1, 1)
    
    # Calculate the division term for the frequencies: shape (d_model // 2,)
    # Using np.exp and np.log is numerically more stable than calculating powers of 10000 directly
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply the sine function to all even indices (0, 2, 4...)
    pe[:, 0::2] = np.sin(pos * div_term)
    
    # Apply the cosine function to all odd indices (1, 3, 5...)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe