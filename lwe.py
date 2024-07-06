import numpy as np 

def generate_mat(n = 10, m = 10, q = 61):
    """
    Generate the public key matrix, default is 10 by 10
    make sure the matrix is random and linearly independent
    
    """
    matrix = np.random.randint(low = 0, high = q-1 , size = (m,n) )
    # if you wanna get full rank matrix
    # rank = np.linalg.matrix_rank(matrix)
    # count=0
    # while rank < m :
    #     count += 1
    #     print(count )
    #     matrix = np.random.randint(low = 0, high = q-1 , size = (m,n) )
    #     rank = np.linalg.matrix_rank(matrix)
    return matrix
def generate_secret_key(n = 10, q = 61):
    """
    Generate the secret vector of the user.
    the size should mach n, the columns of public key
    
    """
    secret_key = np.random.randint(low = 0, high = q-1, size=(n,1))
    return secret_key
def generate_error_vector( m = 10, sigma = 2.0):
    """
    Generate the random error vector selected from 
    gaussian normal distribution with mean of zero and low sigma (standart deviation)
    """
    error_vector = np.random.normal( 0, sigma, m ).astype(int).reshape(m,1)
    return error_vector
def generate_public_key(secret_key, n = 10, m = 10, q = 61):
    matrix = generate_mat(n = n, m = m, q = q)
    #secret_key = generate_secret_key( n = n, q = q)
    error = generate_error_vector(m = m)
    b = (np.matmul(matrix, secret_key) + error) % q
    public_key = np.hstack((matrix,b))
    print("Matrix A:\n", matrix)
    print("Secret key s:\n", secret_key)
    print("Error vector e:\n", error)
    print("Vector b (A * s + e mod q):\n", b)
    print("Public key (A | b):\n", public_key)
    return (public_key, q)
def send_message( public_key, num_of_eq , bit):
    array, q = public_key
    m,n = array.shape
    equation_indices = np.random.randint(low = 1, high = m,size = num_of_eq)
    sum = np.zeros(n, dtype = int)
    for i in equation_indices:
        sum = (sum + array[i]) % q
    sum[-1] = (sum[-1] + bit * (q // 2)) % q
    return sum
def decode_message(secret_key, sum, public_key):
    array, q = public_key
    actual_solution = np.dot(sum[ :-1], secret_key) % q
    encoded_bit = (sum[-1] - actual_solution) % q
    #decoded_bit = 1 if encoded_bit > (q // 2) - encoded_bit else 0
    decoded_bit = 1 if encoded_bit > (q // 2) else 0
    return decoded_bit 

message = 1
m = 10
n = 10
q = 89
num_of_equations = 5
secret_key = generate_secret_key(n,q)
public_key = generate_public_key(secret_key,n, m, q)
sum = send_message(public_key=public_key, num_of_eq = num_of_equations, bit = message)
d= decode_message(secret_key = secret_key, sum = sum, public_key=public_key)
print("message:", message)
print("decoded message", d)