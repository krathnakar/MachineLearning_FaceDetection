# -*- coding: utf-8 -*-
import numpy as np

class SemiNaive:
    def __init__(self, face,numCutOff, group):
        cutoff = numCutOff
        face = np.array([[1,1,1],[1 ,0 ,1],[0 ,1 ,1],[1 ,1 ,1]])
        nonface = np.array([[1,1,1,0],[0,1,0,1],[0,1,1,0]]).T
        print face
        print nonface
        CTClass1 = occArrayFunc(self, face, cutoff, group)
        CPTClass1 = CoPairTabClass(self, face, cutoff, group)
        CTClass2 = occArrayFunc(self, nonface, cutoff, group)
        CPTClass2 = CoPairTabClass(self, nonface, cutoff, group)
        FindProb2(self, CPTClass1, CPTClass2, CTClass1, CTClass2, group)


def occArrayFunc(self, face, cutoff, group):

    occArray = []
    for x in face:#x= pixels
        #create variable for all possible values below
        occ1=0.0
        occ0=0.0
        for each in x[0:group]:#restricting to Cutoff->2 samples
            #change below loops according to possible values
            if each:
                occ1+=1
            else:
                occ0+=1
        temp = [occ0, occ1]#row in occurence table
        occArray.append(temp)#occurence table
    return occArray

def CoPairTabClass(self, face, cutoff, group):
    face =face.T
    pixels = len(face[0])
    #print face
    coPairTabClass = np.zeros((pixels,pixels,cutoff,cutoff))


    for i in range(pixels):
        for j in range(pixels):
            for each in face[0:cutoff]:#restricting to Cutoff->2 samples
                coPairTabClass[i][j][each[i]][each[j]]+=1
    #print "one"
    #print coPairTabClass
    return coPairTabClass


def FindProb2(self, CPTClass1, CPTClass2, CTClass1, CTClass2, group):
    print CPTClass1.shape
    epsilon = np.finfo(np.float).eps

    C1 = np.zeros((CPTClass1.shape[0],CPTClass1.shape[1]))
    res2 = 0.0

    for i in range(len(CPTClass1[0])):
        #res2 = 0.0
        for j in range(len(CPTClass1[1])):
            res2 = 0.0
            pxixj = 0.0
            pxixjw1 = 0.0
            pxixjw2 = 0.0
            pxiw1 = 0.0
            pxjw1 = 0.0
            pxiw2 = 0.0
            pxjw2 = 0.0
            res = 0.0
            for each in range(CPTClass1.shape[2]):
                res2 = 0.0
                for each2 in range(CPTClass1.shape[3]):
                    #res=0.0
                    ri1 = CTClass1[i][each]
                    rj1 = CTClass1[j][each2]
                    ri2 = CTClass2[i][each]
                    rj2 = CTClass2[j][each2]

                #for each2 in range(CPTClass1.shape[3]):
                    q1 = CPTClass1[i][j][each][each2]
                    q2 = CPTClass2[i][j][each][each2]
                    pxixj = (q1 + q2) / (group + group)
                    pxixjw1 = max(q1 / group, epsilon)
                    pxixjw2 = max(q2 / group, epsilon)
                    pxiw1 = ri1/group
                    pxjw1 = rj1/group
                    pxiw2 = ri2/group
                    pxjw2 = rj2/group

                    b_num = pxiw1*pxjw1
                    b_den = max(pxiw2*pxjw2, epsilon)

                    a = np.log(pxixjw1/pxixjw2)
                    b = np.log(max(b_num/b_den, epsilon))
                    res += pxixj * abs(a - b)
                res2 += res
            #res2 = res
            C1[i][j] = float("{0:.3f}".format(res2))
    print C1


def FindProb(self, CPTClass1, CPTClass2, CTClass1, CTClass2, group):
    temp = CTClass1
    for x in range(len(CTClass1)):
        for y in range(len(CTClass1[x])):
            temp[x][y] = CTClass1[x][y]/group#group->samples of faces
    pxiw1 = temp
    temp = CTClass1
    for x in range(len(CTClass1)):
        for y in range(len(CTClass1[x])):
            temp[x][y] = CTClass1[x][y]/group#group->samples of faces
    pxjw1 = temp


    temp = CTClass2
    for x in range(len(CTClass2)):
        for y in range(len(CTClass2[x])):
            temp[x][y] = CTClass2[x][y]/group#group->samples of nonfaces
    pxiw2 = temp
    temp = CTClass2
    for x in range(len(CTClass2)):
        for y in range(len(CTClass2[x])):
            temp[x][y] = CTClass2[x][y]/group#group->samples of nonfaces
    pxjw2 = temp

    temp = CPTClass1


SemiNaive(5,2,2)