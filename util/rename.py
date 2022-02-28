import os
import sys

if __name__ == '__main__':
    path = sys.argv[1]
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir.endswith('.mp4'):
                video = os.path.join(root, dir)
                label = dir.replace('.mp4', '')
                if dir.__contains__('_'):
                    label = label.split('_')[0]
                out = os.path.join(root, label)
                cmd = f'mv {video} {out}'
                os.system(cmd)
