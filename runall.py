
import argparse

parser = argparse.ArgumentParser(description='Available arguments for models benchmarking')

parser.add_argument('--batch_size', default=1, type=int, help='batch size for input data')
parser.add_argument('--device', default=None, help='device on which model will evaluated e.g. CPU:0 or GPU:0 default is prefered device GPU>CPU')
parser.add_argument('--wandb', default=False, type=bool, help='Optional: True if want to add all the results in wandb (weights and biases)')
parser.add_argument('--project_name', default='benchmarks', help='optional name for wandb project')
parser.add_argument('--runonly', default=[], nargs='+', help='will only execute benchmarks for the models listed in this argument')
parser.add_argument('--runexcept', default=[], nargs='+', help='will execute benchmarks for the models except listed in this argument')


args = parser.parse_args()

import rbm
from sys import exc_info

def runAll():
    stars_length = 25
    h_sep = '='
    v_sep = '|'
    if len(args.runonly) > 0:
        model_names = args.runonly
    else:
        model_names = rbm.models.models_names()
        if len(args.runexcept) > 0:
            model_names = list(np.setdiff1d(model_names, args.runexcept))
    for model in model_names:
        try:
            benchmarker = rbm.utils.Benchmark(model=model, batch_size=args.batch_size, device=args.device)
            benchmarks = benchmarker.execute(wandb=args.wandb, project_name=args.project_name)
        except KeyboardInterrupt:
            raise KeyboardInterrupt('KeyboardInterrupt')
            break
        except:
            print()
            head = h_sep*stars_length + ' Benchmark Error ' + h_sep*stars_length
            print(head)
            _p_model = v_sep + ' model:' + model
            print(_p_model + ' ' * (len(head)-len(_p_model)-1) + v_sep)
            _p_error = f"{v_sep} Error: {exc_info()[0]} {exc_info()[1]}"
            print(_p_error + ' '*(len(head)-len(_p_error)-1) + v_sep)
        else:
            print()
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
        print()


if __name__ == "__main__":

    runAll()


