from keras.optimizers import Optimizer
from keras.src import ops

class CustomAdam(Optimizer):
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 name="custom_adam",
                 **kwargs):
        super().__init__(learning_rate=learning_rate,
                         name=name,
                         **kwargs)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    # Create moment vectors - optimizer state variables m and v and attach them for each model weight.
    # TensorFlow will call this func automatically
    def build(self, var_list):
        # var_list is weights + bias of each layers in DL architecture

        if self.built:
            return
        super().build(var_list)

        # Create Adam state variables for each weight
        self.m = []
        self.v = []
        for var in var_list:
            self.m.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="m"
                )
            )
            self.v.append(
                self.add_variable_from_reference(
                    reference_variable=var, name="v"
                )
            )

    # Update variables in Adam algorithm
    def update_step(self, grad, var, learning_rate):
        # grad: gradient
        # var: the model weight being updated
        lr = ops.cast(learning_rate, var.dtype)
        grad = ops.cast(grad, var.dtype)

        # Bias correction
        t = ops.cast(self.iterations + 1, var.dtype)

        beta_1 = ops.cast(self.beta_1, var.dtype)
        beta_2 = ops.cast(self.beta_2, var.dtype)
        
        # m: first moment
        m = self.m[self._get_variable_index(var)]
        # v: second moment
        v = self.v[self._get_variable_index(var)]

        # Update biased first moment estimate
        # m is a TF obj not variable, so call assign method instead of =
        # mt​=beta_1​mt−1​+(1−beta_1​)g
        self.assign(m, beta_1 * m + (1.0 - beta_1) * grad)

        # Update biased second raw moment estimate
        # vt​=beta_2​vt−1​+(1−beta_2​)g^2
        self.assign(v, beta_2 * v + (1.0 - beta_2) * ops.square(grad))

        # Compute bias-corrected
        # m_hat​=m/1−beta_1^t​​
        # v_hat​=m/1−beta_2^t​​
        m_hat = m / (1.0 - ops.power(beta_1, t))
        v_hat = v / (1.0 - ops.power(beta_2, t))

        # Apply update
        # assign_sub: subtract a param amount then assign to the variable
        # theta = theta - alpha*m_hat/(sqrt(v_hat) + epsilon)
        self.assign_sub(var, lr * m_hat / (ops.sqrt(v_hat) + self.epsilon))


    # Used to save/load the optimizer
    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon
            }
        )
        return base_config
    
