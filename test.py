from utils import makeDetection

def main():
    modelpaths = ['.\\model\\svc-rbf-232f03.pkl', '.\\model\\svc-linear-8b953d.pkl', '.\\model\\svc-poly-5a474b.pkl']
    rawpaths = ['.\\test-rbf\\raw', '.\\test-linear\\raw', '.\\test-poly\\raw']
    boxedpaths = ['.\\test-rbf\\boxed', '.\\test-linear\\boxed', '.\\test-poly\\boxed']

    cnt = len(modelpaths)
    for i in range(cnt):
        m = modelpaths[i]
        r = rawpaths[i]
        b = boxedpaths[i]
        print(f'Using {m} for detection ......')
        makeDetection(m, r, b)
        print('')

if __name__ == '__main__':
    main()