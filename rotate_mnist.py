from PIL import Image
import os 
import random
if __name__ == "__main__": 
    for root,dirs,files in os.walk('./mnist_png_files'):
        for f in files:
            if f.endswith('.png'):
                img = Image.open(root+'/'+f)
                for i in range(5):
                    rotation = random.randint(0,360)
                    rotImg = img.rotate(rotation)
                    
                    pathSplit = root.split('/')
                    pathSplit.append(f)
                    pathSplit[1] +='_rot'
                    pathSplit[-1] = pathSplit[-1].split('.')[0] + '_' + str(rotation) + '.png'
                    newPath = '/'.join(pathSplit)
                    print(newPath)
                    rotImg.save(newPath)
                    
                    
