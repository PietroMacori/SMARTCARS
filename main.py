import math
import pickle
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file = open("myFile3", "rb")
    data = pickle.load(file)
    plt.axes(projection = 'polar')


    list_fin = []
    for p in data[0]:
        ang = 0
        if p[1]<0:
            ang = 2*math.pi+p[1]
        else:
            ang = p[1]
        if ang>0.36 and ang < 2.3:
            continue

        if ang>4.25 and ang < 4.97 and p[0]>11.5 :
            continue


        if ang>2.8 and ang < 3.3 and p[0]>11 and p[0]<13 :
            continue
        
        list_fin.append((p[0],ang))

    afile = open("./filter2", "wb")
    pickle.dump(list_fin,afile )
    afile.close()
