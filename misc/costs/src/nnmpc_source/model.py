class Model:
    def __init__(self):
        self.T = 3.2 # planning horizon, sec
        self.dt = 0.3 # time step, sec
        self.dt_sim = 0.1 # simulation time step, sec
        self.N = int(self.T/self.dt) # number of time steps
        self.a_max = 2
        self.a_min = -2
        self.v_max = 3
        self.v_min = 0
        self.steer_max = 0.5
        self.steer_min = -0.5
        self.width = 1
        self.length = 1
        self.N_action = 20 # number of samples
        self.goal = []

        # Prediction
        self.predictor_type = "sgan" # {constant-vel, sgan, ...}
        # self.sgan_model = "model_v7_2.pt"
        self.sgan_model = "default.pt"
        self.T_obs = 8 


        # Optimization
        self.lambda_v = 1
        self.lambda_div = 100
