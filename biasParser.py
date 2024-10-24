import glob


class biasParser:
    def __init__(self, bias_dir, kwargs = {}):
        self.bias_dir = bias_dir
        self.run_paths = glob.glob(f"{bias_dir}/run*")
        self.function = None

    def set_function(self, function, kwargs):
        self.function = function
        self.kwargs = kwargs

    def run(self):
        results = []
        for run_path in self.run_paths:
            result = self.function(run_path)
            if result is None:
                continue
            results.append(result)
        return results
    
    def double_run(self):
        results = []
        prev_run_path = self.run_paths[0]
        for run_path in self.run_paths[1:]:
            result = self.function(prev_run_path, run_path, **self.kwargs)
            prev_run_path = run_path
            if results is None:
                continue
            results.append(result)
        return results