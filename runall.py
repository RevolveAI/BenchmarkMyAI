
import argparse

parser = argparse.ArgumentParser(description='Available arguments for models benchmarking')

parser.add_argument('--batch_size', default=1, type=int, help='batch size for input data')
parser.add_argument('--device', default='CPU:0', help='device on which model will evaluated e.g. CPU:0 or GPU:0')
parser.add_argument('--wandb', default=False, type=bool, help='Optional: True if want to add all the results in wandb (weights and biases)')
parser.add_argument('--project_name', default='benchmarks', help='optional name for wandb project')

args = parser.parse_args()

import rbm
from sys import exc_info

def runAll():
    stars_length = 25
    h_sep = '='
    v_sep = '|'
    model_names = rbm.models.models_names()
    for model in model_names:
        try:
            benchmarker = rbm.utils.Benchmark(model=model, batch_size=args.batch_size, device=args.device)
            benchmarks = benchmarker.execute(wandb=args.wandb, project_name=args.project_name)
        except KeyboardInterrupt:
            raise KeyboardInterrupt('KeyboardInterrupt')
            break
        except:
            head = h_sep*stars_length + ' Benchmark Error ' + h_sep*stars_length
            print(head)
            _p_model = v_sep + ' model:' + model
            print(_p_model + ' ' * (len(head)-len(_p_model)-1) + v_sep)
            _p_error = f"{v_sep} Error: {exc_info()[0]} {exc_info()[1]}"
            print(_p_error + ' '*(len(head)-len(_p_error)-1) + v_sep)
        else:
            head = h_sep*stars_length + ' Benchmark Results ' + h_sep*stars_length
            print(head)
            benchmarks.update(benchmarks['benchmark'])
            _ = benchmarks.pop('benchmark')
            ls = list()
            for key, val in benchmarks.items():
                _p_result = f'{v_sep} {key}: {val}'
                ls.append(_p_result + ' '*(len(head)-len(_p_result)-1) + v_sep)
            print('\n'.join(ls))
        foot = h_sep*len(head)
        print(foot)


if __name__ == "__main__":

    runAll()


