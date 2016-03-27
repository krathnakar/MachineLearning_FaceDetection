# -*- coding: utf-8 -*-
import numpy as np
import sys




class SemiNaive:
    def __init__(self, face_shape,numCutOff, group):
        self.cutoff = numCutOff
        self.group = group
        #face = np.array([[1,1,1],[1 ,0 ,1],[0 ,1 ,1],[1 ,1 ,1]])#pixel*sample
        #nonface = np.array([[1,1,1,0],[0,1,0,1],[0,1,1,0]]).T
        #print face
        #print nonface
        self.pixels = face_shape

    def train(self, face, nonface, model):
        cutoff = self.cutoff
        group = self.group
        pixels = self.pixels
        #print face
        #print "pix"+str(pixels)
        #print face.shape
        #face = np.array(face).T
        #print face.shape
        #nonface = np.array(nonface).T
        #print nonface
        CTClass1 = occArrayFunc(self, face, cutoff, group)
        #print "CTClass1"+str(np.array(CTClass1).shape)
        CPTClass1 = CoPairTabClass(self, face, cutoff, group)
        CTClass2 = occArrayFunc(self, nonface, cutoff, group)
        CPTClass2 = CoPairTabClass(self, nonface, cutoff, group)
        ProbTabel = FindProb2(self, CPTClass1, CPTClass2, CTClass1, CTClass2, group)
        SubGroup = findSubGroup(ProbTabel, group)
        faces_probTabSubGroup = probTabSubGroup(pixels, cutoff, group, SubGroup, np.array(face).T)
        nfaces_probTabSubGroup = probTabSubGroup(pixels, cutoff, group, SubGroup, np.array(nonface).T)
        #test_img(cutoff, np.array(face).T, SubGroup, faces_probTabSubGroup, nfaces_probTabSubGroup)
        #test_img(cutoff, np.array(nonface).T, SubGroup, faces_probTabSubGroup, nfaces_probTabSubGroup)
        model.setSubgroupSize(group)
        model.setDataSize(pixels)#length of faces.shape[0]
        model.setNumClass1(len(face))#no.of samples faces
        model.setNumClass2(len(nonface))#no. of samples nfaces
        model.setSubgroups(SubGroup)#s
        model.setTableClass1(faces_probTabSubGroup)
        model.setTableClass2(nfaces_probTabSubGroup)
        model.setGoodness("commented")
        model.write()

    def loadModel(self, model):

        ref = model.read()

        self.subgroupSize = model.subgroupSize
        self.pixels = model.dataSize
        self.numClass1 = model.numClass1
        self.numClass2 = model.numClass2
        self.subGroups = model.subgroups
        self.tableClass1 = model.tableClass1
        self.tableClass2 = model.tableClass2
        self.goodness = model.goodness



    def test(self, faces):#, subgroup, faces_table, nfaces_table):
        #print subgroup.shape
        cutoff = self.cutoff
        subgroup = self.subGroups
        faces_table = self.tableClass1
        nfaces_table = self.tableClass2

        #epsilon = np.finfo(np.float).eps
        epsilon = sys.float_info.epsilon
        #print faces
        #faces = faces[2]#change this later
        #print faces

        final_value = []
        for each_face in faces:
            pos_values = []
            npos_values = []
            for each_row in range(subgroup.shape[0]):#goes through rows of subgroup
                #print subgroup[each_row]
                temp = []
                for each_col in subgroup[each_row]:#takes each column VALUES in each row
                    temp.append(each_face[each_col])#appends sample circles to temp- matrix 3 in section6
                #print temp
                find_col = fromDigits(temp, cutoff)#finds column in matrix 2 in section6
                #print find_col
                pos_values.append(faces_table[each_row][find_col])#, epsilon))#positive prob. table
                npos_values.append(nfaces_table[each_row][find_col])#, epsilon))#negative prob. table
            #print pos_values
            #print npos_values
            #finding log equation value
            total_value = 0
            #for n,i in enumerate(npos_values):
            #    if i == 0:
            #        npos_values[n] = epsilon**1000
            #pos_values = np.array(pos_values)
            #npos_values = np.array(npos_values)
            #print npos_values.shape
            for itr in range(len(pos_values)):

                npos_values[itr] = max(epsilon**3,npos_values[itr])
                den = max(npos_values[itr], epsilon)
                num = max(pos_values[itr], epsilon)
                if npos_values[itr] == 0:
                    den = 0.0000000000001
                #print pos_values
                #print npos_values
                total_value += np.log(num/den)

            final_value.append(total_value)
        #print "complete"
        final_value = np.array(final_value)

        return final_value


def occArrayFunc(self, face, cutoff, group):

    occArray = []
    for x in face:#x= pixels
        #create variable for all possible values below
        occ1=0.0
        occ0=0.0
        occ2=0.0
        occ3=0.0
        occ4=0.0
        occ5=0.0
        for each in x[0:cutoff]:#restricting to Cutoff->2 samples
            #change below loops according to possible values
            if each==5:
                occ5+=1
            elif each==4:
                occ4+=1
            elif each==3:
                occ3+=1

            elif each==2:
                occ2+=1
            elif each==1:
                occ1+=1
            elif each==0:
                occ0+=1
        temp = [occ0, occ1, occ2, occ3, occ4, occ5]#row in occurence table
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
    #print np.array(CTClass1).shape
    #print CPTClass1.shape
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
    #print C1
    return C1


def findSubGroup(C1, group):
    pass
    temp = C1
    #print temp.shape
    #print group
    submatrix = np.zeros((temp.shape[0], group))
    for each in range(group):
        for col in range(temp.shape[0]):
            if each==0:
                submatrix[col][each] = col
                #print submatrix
                temp[col][col] = 0.0
            else:
                #print C1[col]
                #temparr = np.max(C1[col])
                ind = np.argmax(C1[col])
                submatrix[col][each] = ind
                temp[col][ind] = 0.0
                #ind = C1[col].index(max(C1[col]))
                #print ind
                #submatrix[col][each] = max(i for i in C1[col])
    #print temp
    return submatrix


def probTabSubGroup(pixels, cutoff, group, subgroup, faces):
    pass
    PTSubMatrix = np.zeros((pixels, cutoff**group))
    #print PTSubMatrix
    #temp = []
    for i in range(pixels):
        temp = []
        for j in range(group):
            temp.append(subgroup[i][j])
            pass

        #insert_col = []
        for each_face in range(group):
            temp_face = []
            for each in range(len(temp)):
                temp_face.append(faces[each_face][temp[each]])
            #print temp_face #second table circles in section5
            insert_col = fromDigits(temp_face, cutoff)
            PTSubMatrix[i][insert_col] += 1
    #print PTSubMatrix
    return PTSubMatrix


def fromDigits(digits, b):
    """Compute the number given by digits in base b."""
    n = 0
    for d in digits:
        n = b * n + d
    return n
