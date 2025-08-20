from PIL import Image
import argparse
from arcface.config import get_config
from mtcnn import MTCNN
from arcface.Learner import face_learner
from arcface.utils import load_facebank

parser = argparse.ArgumentParser(description='for face verification')
parser.add_argument("-s", "--save", help="whether save",action="store_true")
parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=.75, type=float)
parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
args = parser.parse_args()

conf = get_config(False)


class RECOGNITION:
    def __init__(self) -> None:
        self.mtcnn = MTCNN()
        self.learner = face_learner(conf, True)
        self.learner.threshold = args.threshold
        if conf.device.type == 'cpu':
            self.learner.load_state(conf, 'cpu_final.pth', True, True)
        else:
            self.learner.load_state(conf, 'final.pth', True, True)
        self.learner.model.eval()
        self.targets, self.names = load_facebank(conf)

    # returns a name and corresponding bounding box
    def face_verify(self, frame):
        res = {
            'names':'None',
            'bboxes':'None',
            'scores':'None'
        }
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        
        try:
            bboxes, faces = self.mtcnn.align_multi(image, 10, 30)
        
            try:
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice
                results, score = self.learner.infer(conf, faces, self.targets, args.tta)
                
                names = []
                for idx, bbox in enumerate(bboxes):
                    if args.score:
                        return res               
                    
                    else:
                        if float('{:.2f}'.format(score[idx])) > .9:
                            name = self.names[0]
                            names.append(name)
                        else:
                            name = self.names[results[idx]+1]
                            names.append(name)
                print(type(bboxes), type(score), results.detach().cpu().numpy().tolist())
                res.update({'names':str(names), 'bboxes':str(bboxes.tolist()), 'scores':str(score.detach().cpu().numpy().tolist())})
                return res

            except:
                return res

        except Exception as e:
            print(e)
            return res