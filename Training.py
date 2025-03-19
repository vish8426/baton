import os

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'


labels = [{'name':'forward', 'id':1},
          {'name':'left', 'id':2},
          {'name':'right', 'id':3},
          {'name':'stop', 'id':4},
          {'name':'back', 'id':5},
          {'name':'fast', 'id':6},
          {'name':'slow', 'id':7}
         ]

ORIGIN_DIR = os.getcwd()


with open(ANNOTATION_PATH + '\label_map.pbtxt', 'w') as f:
    for label in labels:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

cmd = 'python ' + SCRIPTS_PATH + '/generate_tfrecord.py -x ' +IMAGE_PATH + '/train -l '+ANNOTATION_PATH + '/label_map.pbtxt -o '+ANNOTATION_PATH + '/train.record'
print(cmd)
# os.system(cmd)

cmd = 'python '+SCRIPTS_PATH+ '/generate_tfrecord.py -x '+ IMAGE_PATH + '/test -l ' +ANNOTATION_PATH + '/label_map.pbtxt -o '+ANNOTATION_PATH + '/test.record'
print(cmd)
# os.system(cmd)
