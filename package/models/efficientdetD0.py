
#%%
import subprocess
import os
#%%
def executeInfer(model_name='efficientdet-d0', batch_size=16):
    #%%
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models/efficientdet/')
    #%%
    process = subprocess.Popen(['python', model_path + '/model_inspect.py', '--runmode', 'bm',
                                '--model_name', model_name, '--batch_size', str(batch_size), 
                                '--hparams', "mixed_precision=True"],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    result = stdout.decode("utf-8")
    #%%
    try:
        inference_time_batch = float(result.split('Per batch inference time:  ')[1].split('\n')[0])
        fps = float(result.split('FPS:  ')[1])
    except:
        return [False, {'result': result, 'error': stderr}]
    #%%
    return [True, {'inference_time_batch': inference_time_batch,
            'fps': fps}]