import numpy as np
import math

class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + self._k + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]


class SumProduct:
    def __init__(self, p):
        self._potentials = p
        # TODO: EDIT HERE
        # add whatever data structures needed
        self._N = p.chain_length()
        self._K = p.num_x_values()
        self._vertical_f_to_x = np.asmatrix(np.ones((self._K + 1, self._N + 1)))
        for k in range(1, self._K + 1):
            for n in range(1, self._N + 1):
                self._vertical_f_to_x[k, n] = self._potentials.potential(n, k)
        
        self._left_right_f_to_x = np.asmatrix(np.ones((self._K + 1, self._N + 1)))
        self._right_left_f_to_x = np.asmatrix(np.ones((self._K + 1, self._N + 1)))
        
        # left to right
        for node in range(1, self._N):
            # calculate horizontal x -> f
            prev_horiz_message = self._left_right_f_to_x[:, node - 1]
            u_x_to_f = np.multiply(prev_horiz_message, self._vertical_f_to_x[:, node])

            # calculate horizontal f -> x
            for k in range(1, self._K + 1):
                sum = 0
                for c in range(1, self._K + 1):
                    sum = sum + (self._potentials.potential(node + self._N, c, k) * u_x_to_f[c, 0])
                
                self._left_right_f_to_x[k, node] = sum


        # right to left
        for node in range(self._N, 1, -1):
            # calculate horizontal f <- x
            prev_horiz_message = self._right_left_f_to_x[:, node]
            u_x_to_f = np.multiply(prev_horiz_message, self._vertical_f_to_x[:, node])

            # calculate horizontal x <- f
            for k in range(1, self._K + 1):
                sum = 0
                for c in range(1, self._K + 1):
                    sum = sum + (self._potentials.potential(node + self._N - 1, k, c) * u_x_to_f[c, 0])

                self._right_left_f_to_x[k, node - 1] = sum
        

    def marginal_probability(self, x_i):
        # TODO: EDIT HERE
        # should return a python list of type float, with its length=k+1, and the first value 0
  
        # final probability calculation
        temp = np.multiply(self._left_right_f_to_x[:, x_i - 1], self._right_left_f_to_x[:, x_i])
        result = np.multiply(temp, self._vertical_f_to_x[:, x_i])
        result[0, 0] = 0
        output = []
        sum = 0
        for x in range(0, self._K + 1):
            sum = sum + result[x, 0]
            output.append(result[x, 0])
        output[:] = [x / float(sum) for x in output]
        return output


class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        # TODO: EDIT HERE
        # add whatever data structures needed
        self._N = p.chain_length()
        self._K = p.num_x_values()
        self._vertical_f_to_x = np.asmatrix(np.ones((self._K + 1, self._N + 1)))
        for k in range(1, self._K + 1):
            for n in range(1, self._N + 1):
                self._vertical_f_to_x[k, n] = math.log(self._potentials.potential(n, k))
                
        self._left_right_f_to_x = np.asmatrix(np.zeros((self._K + 1, self._N + 1)))
        self._right_left_f_to_x = np.asmatrix(np.zeros((self._K + 1, self._N + 1)))
        self._backtracker = np.asmatrix(np.zeros((self._K + 1, self._N + 1)))


    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        # TODO: EDIT HERE

        # left to right
        for node in range(1, x_i):
            # calculate horizontal x -> f
            prev_horiz_message = self._left_right_f_to_x[:, node - 1]
            u_x_to_f = np.add(prev_horiz_message, self._vertical_f_to_x[:, node])

            # calculate horizontal f -> x
            for k in range(1, self._K + 1):
                max_sum = float('-inf')
                max_c = -1
                for c in range(1, self._K + 1):
                    temp = math.log(self._potentials.potential(node + self._N, c, k)) + u_x_to_f[c, 0]
                    if temp > max_sum:
                        max_sum = temp
                        max_c = c
                
                self._left_right_f_to_x[k, node] = max_sum
                self._backtracker[k, node] = max_c


        # right to left
        for node in range(self._N, x_i, -1):
            # calculate horizontal f <- x
            prev_horiz_message = self._right_left_f_to_x[:, node]
            u_x_to_f = np.add(prev_horiz_message, self._vertical_f_to_x[:, node])

            # calculate horizontal x <- f
            for k in range(1, self._K + 1):
                max_sum = float('-inf')
                max_c = -1
                for c in range(1, self._K + 1):
                    temp = math.log(self._potentials.potential(node + self._N - 1, k, c)) + u_x_to_f[c, 0]
                    if temp > max_sum:
                        max_sum = temp
                        max_c = c

                self._right_left_f_to_x[k, node - 1] = max_sum
                self._backtracker[k, node] = max_c

        # final probability calculation
        temp = np.add(self._left_right_f_to_x[:, x_i - 1], self._right_left_f_to_x[:, x_i])
        result = np.add(temp, self._vertical_f_to_x[:, x_i])
        #result[0, 0] = 0
        
        max_k_val = float('-inf')
        max_k = -1
        for k in range(1, self._K + 1):
            if result[k, 0] > max_k_val:
                max_k_val = result[k, 0]
                max_k = k

        self._assignments[x_i] = max_k
        k = max_k
        for n in range(x_i + 1, self._N + 1):
            self._assignments[n] = int(self._backtracker[k, n])
            k = int(self._backtracker[k, n])
        k = max_k
        for n in range(x_i - 1, 0, -1):
            self._assignments[n] = int(self._backtracker[k, n])
            k = int(self._backtracker[k, n])
        
        normalization_const = self.get_normalization_constant(x_i)

        return max_k_val - math.log(normalization_const)


    def get_normalization_constant(self, x_i):
        vertical_f_to_x = np.asmatrix(np.ones((self._K + 1, self._N + 1)))
        for k in range(1, self._K + 1):
            for n in range(1, self._N + 1):
                vertical_f_to_x[k, n] = self._potentials.potential(n, k)
        
        left_right_f_to_x = np.asmatrix(np.ones((self._K + 1, self._N + 1)))
        right_left_f_to_x = np.asmatrix(np.ones((self._K + 1, self._N + 1)))
        
        # left to right
        for node in range(1, self._N):
            # calculate horizontal x -> f
            prev_horiz_message = left_right_f_to_x[:, node - 1]
            u_x_to_f = np.multiply(prev_horiz_message, vertical_f_to_x[:, node])

            # calculate horizontal f -> x
            for k in range(1, self._K + 1):
                sum = 0
                for c in range(1, self._K + 1):
                    sum = sum + (self._potentials.potential(node + self._N, c, k) * u_x_to_f[c, 0])
                
                left_right_f_to_x[k, node] = sum


        # right to left
        for node in range(self._N, 1, -1):
            # calculate horizontal f <- x
            prev_horiz_message = right_left_f_to_x[:, node]
            u_x_to_f = np.multiply(prev_horiz_message, vertical_f_to_x[:, node])

            # calculate horizontal x <- f
            for k in range(1, self._K + 1):
                sum = 0
                for c in range(1, self._K + 1):
                    sum = sum + (self._potentials.potential(node + self._N - 1, k, c) * u_x_to_f[c, 0])

                right_left_f_to_x[k, node - 1] = sum

        temp = np.multiply(left_right_f_to_x[:, x_i - 1], right_left_f_to_x[:, x_i])
        result = np.multiply(temp, vertical_f_to_x[:, x_i])
        result[0, 0] = 0
        sum = 0
        for x in range(0, self._K + 1):
            sum = sum + result[x, 0]
        return sum
