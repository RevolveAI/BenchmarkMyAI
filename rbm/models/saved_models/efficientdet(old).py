
#%%
import subprocess
import os
import tensorflow as tf
#%%
def executeInfer(model_name, batch_size, image_size):
    '''
    

    Parameters
    ----------
    model_name : TYPE: str
        DESCRIPTION.Like efficientdet-d0, efficientdet-d1, ..., efficientdet-d7
    batch_size : TYPE: int
        DESCRIPTION.
    image_size : TYPE: tuple
        DESCRIPTION. (244, 244)

    Returns
    -------
    list
        DESCRIPTION.

    '''
    #%%
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models/efficientdet/')
    #%%
    process = subprocess.Popen(['python', model_path + '/model_inspect.py', '--runmode', 'bm',
                                '--model_name', model_name, '--batch_size', str(batch_size), 
                                '--hparams', f"image_size={image_size[0]}x{image_size[1]},mixed_precision=True"],
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
                   'fps': fps,
                   'framework': 'TensorFlow ' + tf.__version__,
                   'name': model_name}]