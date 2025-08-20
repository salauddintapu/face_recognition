import argparse
from arcface.config import get_config
from mtcnn import MTCNN
from arcface.Learner import face_learner
from arcface.utils import prepare_facebank

parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("-s", "--save", help="whether save",action="store_true")
parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=.75, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
args = parser.parse_args()

conf = get_config(False)

mtcnn = MTCNN()
print('arcface loaded')

learner = face_learner(conf, True)
learner.threshold = args.threshold
if conf.device.type == 'cpu':
    learner.load_state(conf, 'cpu_final.pth', True, True)
else:
    learner.load_state(conf, 'final.pth', True, True)
learner.model.eval()

#register faces from data/facebank directory
print('Registration In Progress. Please Wait...')

targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)

print('Registration Process has been finished successfully! Congratulations!!!')