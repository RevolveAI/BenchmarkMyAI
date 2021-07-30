
import wandb
import copy


class WandB:
    def __init__(self, project_name, run_name):
        self.project_name = project_name
        self.run_name = run_name
        self._instance_ = None
    def init(self):
        self._instance_ = wandb.init(project=self.project_name, name=self.run_name)
    def plot(self, benchmarks):
        if benchmarks['std'] is None:
            benchmarks['std'] = 0
        wandb.log(benchmarks)
    def draw_table(self, benchmarks):
        table = wandb.Table(columns=list(benchmarks.keys()))
        table.add_data(*list(benchmarks.values()))
        wandb.log({"Benchmarks": table})
    def plot_and_table(self, benchmarks):
        bm = copy.deepcopy(benchmarks)
        self.plot(bm['benchmark'])
        bm.update(benchmarks['benchmark'])
        _ = bm.pop('benchmark')
        _ = bm.pop('memory_info')
        if benchmarks['std'] is None:
            benchmarks['std'] = 'null'
        self.draw_table(bm)
    def close(self):
        self._instance_.finish()