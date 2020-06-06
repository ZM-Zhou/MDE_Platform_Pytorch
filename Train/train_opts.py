class TrainOpts():
    def __init__(self):
        self.rand_seed = 2020
        self.no_cuda = False

        self.num_workers = 0
        self.is_shuffle = False

        self.optim = "Adam"
        self.learning_rate = 1e-4
        self.scheduler = "Step"
        self.scheduler_step_size = [15]
        self.scheduler_rate = 0.1
        self.num_epochs = 20
        self.batch_size = 12

        self.log_dir = "/data/Train_Log"
        self.log_frequency = 20
        self.visual_frequency = 30
        self.visual_stop = 300
        self.save_frequency = 2
        self.load_weights_folder = None
        self.check_grad = False

