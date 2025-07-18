class ActivationAnalyzer():

    def __init__(self):
        self.activations = {}

    def get_activation(self, idx):
            def hook(module, input, output):
                self.activations[idx] = output
            return hook

    def attach_forward_hooks(self, model):
        index = 0


        for layer in model.model.layers:
            layer.self_attn.register_forward_hook(self.get_activation(index))
            index += 1

