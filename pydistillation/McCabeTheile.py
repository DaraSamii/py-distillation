import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root



class McCabeTheile:
    def __init__(self, y_func):
        '''
        MaCabe-Theili method for solving distillation probles

        y_func is a function `y_func(x) --> y`
        
        y_func(0) == 0
        y_func(1) == 1 
        '''
        self.y_func = y_func
        self.PlotConfig()

        self.MultipleFeed = False
        self.SideStream = False


##########################################################################
    def PlotConfig(self, figsize=(10,10), YXColor = 'k', YXlineWidth=2, initPC = 'r',AP = 'y',OpC='b',StageP = 'g',StageLine='r'):
        '''
        this function Costumize matplotlib plots

        figsize: figure size[tuple (heigh, width)]
        YXColor: color string[exampe: red:'r', black:'k'] Color of YX and XX curves
        YXlineWidth: line width of YX and XX curves
        initPC: color of xW,zF,xD, Points
        AP: color of q-line and Operating line intersections
        OpC: color of operating line
        StageP: color of points where stages intersect with YX and XX
        StageLine: color of stage lines

        '''
        self.figsize = figsize
        self.YXColor = YXColor
        self.YXlineWidth = YXlineWidth
        self.initPC = initPC
        self.AP = AP
        self.OpC = OpC
        self.StageP = StageP
        self.StageLine = StageLine


##########################################################################
    def raw_plot(self):
        '''
        this function plots YX and XX curve
        '''
        # creating figure
        self.fig, self.ax = plt.subplots(nrows=1,ncols=1,figsize = self.figsize)

        # curve points
        Xs = np.linspace(0,1,200)
        Ys = self.y_func(Xs)

        # setting figure axes limits
        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)

        #ploting YX and XX curve
        self.ax.plot(Xs, Ys, color = self.YXColor, linewidth=self.YXlineWidth)
        self.ax.plot(Xs, Xs, color = self.YXColor, linewidth=self.YXlineWidth)

        self.ax.set_ylabel('y')
        self.ax.set_xlabel('x')

########################################################################## 
    def _lineSlope(self,a,b):
        '''
        this function return a line function`a*x + b` 
        line(x) --> y
        '''
        def line(x):
            return a* x + b

        return line 


##########################################################################
    def _linePoints(slef,A,B):
        '''
        A helper function which creates a line function ax + b from 2 given points

        A: (xA[float], yA[float])
        B: (xB[float], yB[float])
        '''
        xA, yA = A
        xB, yB = B

        #sloope of the line
        a = (yB - yA)/(xB - xA)

        #creating line function
        def func(x):
            return a*(x - xA) + yA

        return func


##########################################################################
    def DistillationFrac(self, F, zF, xD, xW, F2=None, zF2=None,S=None, xS=None):
        '''
        F = D + W
        F.zF = D.xD + W.xW

        given: F, zF, xD, xW
        returns D,W
        '''

        a = np.array([[1.0, 1.0], [xD, xW]])
        b = np.array([F, F*zF])

        if F2 !=None and zF2 !=None:                #adding F2 and zF2 to mass balance
            b = np.array([F + F2, F*zF + F2*zF2])

        if S !=None and xS !=None:                  #subtracting S and xS to mass balance
            b = np.array([F - S, F*zF - S*xS])

        result = np.linalg.solve(a, b)

        D = result[0]
        W = result[1]

        return (D,W)

##########################################################################
    def _drawLine(self, P1, P2, style, linewidth = 1):
        '''
        A helper function which draw a line form P1 to P2
        
        P1: Point 1 (x[float], y[float])
        P2: Point 2 (x[float], y[float])
        '''

        self.ax.plot([P1[0], P2[0]], [P1[1], P2[1]],
                                         style,linewidth= linewidth)



##########################################################################
    def _drawPoint(self, x, y ,color='r'):
        '''
        helper function draws Point(x,y) in the given `ax`
        '''
        
        self.ax.plot([x],[y],color + 'o')

##########################################################################
    def _Intersection(self, func1, func2):
        '''
        a helper function which finds intersection of two lines
        return (x,y) of th intersection point
        '''
        def FUNC(x):
            return func1(x) - func2(x)

        rootX = root(FUNC, 1)['x'][0]

        return (rootX, func1(rootX))


##########################################################################
    def _qline(self,q,zF):
        '''
        thid helper function return line function for q-line
        q: quality of the feed between 0 and 1
        zF: feed fraction, between and 0 and 1
        '''
        if q == 1:      #aproximetry slope of inf
            q = 0.999

        a = q/(q-1)
        b = -zF/(q-1)
        
        def func(x):
            return a*x + b
        
        return func



##########################################################################
    def addData(self, xD, xW, zF, F,q,R, zF2=None, F2 = None, q2 = None, xS = None, S = None, qS=None):
        '''
        add the data for furthur Computations
        this module supports only 3 options
        -simple
        -multi Feed
        -Side stream
        * side stream and multi feed both can't be included
        '''
        self.xD = xD
        self.xW = xW
        self.zF = zF
        self.F = F
        self.q = q
        self.R = R

        if F2 != None and S != None:
            raise Exception("only one of `side stream` or `multi feed` can be included")

        #addig multi feed data
        if zF2 != None and F2 != None and q2 != None:
            self.MultipleFeed = True
            self.zF2 = zF2
            self.F2 = F2
            self.q2 = q2

            # sorting f1 f2 
            (self.zF,self.F,self.q),(self.zF2,self.F2,self.q2) = sorted([(self.zF,self.F,self.q),(self.zF2, self.F2,self.q2)],reverse=True)
            
            self.qline2 = self._qline(self.q2, self.zF2)

        # adding Side Stream data
        elif xS != None and S != None:
            self.xS = xS
            self.S = S
            self.qS = qS
            if self.qS == None:
                self.qS = 1.0
            self.SideStream = True
            self.qsline = self._qline(self.qS, self.xS)

        (self.D, self.W) = self.DistillationFrac(F, zF, xD, xW,F2,zF2, S, xS) #Compute W,D

        self.qline = self._qline(self.q, self.zF)   #main feed q-line

            
##########################################################################
    def plot_Data(self,plotOpLine = True):
        '''
        this function displays intial condition of Ponchon-Saverit method
        '''
        self.raw_plot()

        self._drawPoint(self.xW,self.xW, color = self.initPC)
        self._drawPoint(self.xD, self.xD, color = self.initPC)
        self._drawPoint(self.zF, self.zF, color = self.initPC)

        qyIntersectPoint = self._Intersection(self.y_func, self.qline)
        self._drawPoint(qyIntersectPoint[0],qyIntersectPoint[1], color=self.initPC)
        self._drawLine((self.zF,self.zF),qyIntersectPoint,style = self.initPC +':')


        self.RectLine = self._lineSlope(self.R/(self.R+1), self.xD/(self.R+1))
    
        if self.MultipleFeed == True:
            self.SideStream = False
            self._drawPoint(self.zF2, self.zF2, color = self.initPC)


            q2yIntersectPoint = self._Intersection(self.y_func, self.qline2)
            self._drawPoint(q2yIntersectPoint[0],q2yIntersectPoint[1], color=self.initPC)
            self._drawLine((self.zF2,self.zF2),q2yIntersectPoint,style = self.initPC +':')

            A = self._Intersection(self.RectLine, self.qline)
            self.qOpPoint = A
            if plotOpLine == True:
                self._drawPoint(A[0],A[1],color=self.AP)
                self._drawLine(A, (self.xD,self.xD),style=self.OpC)


            L = self.R * self.D
            L_prime = L + self.q*self.F

            G = (1+self.R)*self.D
            G_prime = G + (self.q-1)*self.F

            self.midslope = L_prime/G_prime
            b = A[1] - self.midslope*A[0]
            self.q1q2line = self._lineSlope(self.midslope,b)
            
            B = self._Intersection(self.qline2,self.q1q2line)
            if plotOpLine == True:
                self._drawPoint(B[0],B[1],color=self.AP)
                self._drawLine(A, B,style=self.OpC)          

                self._drawLine(B, (self.xW,self.xW),style=self.OpC)

            def OL(x):
                if A[0] < x:
                    return self.RectLine(x)
                elif A[0]> x and x> B[0]:
                    return self.q1q2line(x)
                elif B[0]> x:
                    return self._linePoints(B, (self.xW,self.xW))(x)

            self.OpLine = OL



        if self.SideStream == True:
            self.MultipleFeed = False
            qsyIntersectPoint = self._Intersection(self.y_func, self.qsline)
            self._drawPoint(qsyIntersectPoint[0],qsyIntersectPoint[1], color=self.initPC)
            self._drawLine((self.xS,self.xS),qsyIntersectPoint,style = self.initPC +':')

            A = self._Intersection(self.RectLine, self.qsline)
            self.qOpPoint = A
            if plotOpLine == True:
                self._drawPoint(A[0],A[1],color=self.AP)
                self._drawLine(A, (self.xD,self.xD),style=self.OpC)

            self.midslope = (self.R - (self.S/self.D))/(self.R + 1)
            b = A[1] - self.midslope*A[0]
            self.sline = self._lineSlope(self.midslope, b)
            
            B = self._Intersection(self.qline,self.sline)
            if plotOpLine == True:
                self._drawPoint(B[0],B[1],color=self.AP)
                self._drawLine(A, B,style=self.OpC)          

                self._drawLine(B, (self.xW,self.xW),style=self.OpC)

            def OL(x):
                if A[0] < x:
                    return self.RectLine(x)
                elif A[0]> x and x> B[0]:
                    return self.sline(x)
                elif B[0]> x:
                    return self._linePoints(B, (self.xW,self.xW))(x)

            self.OpLine = OL


        if self.SideStream == False and self.MultipleFeed == False:
            A = self._Intersection(self.RectLine, self.qline)
            self.qOpPoint = A
            if plotOpLine == True:
                self._drawPoint(A[0],A[1],color=self.AP)
                self._drawLine(A, (self.xD,self.xD),style=self.OpC)
                self._drawLine(A, (self.xW,self.xW),style=self.OpC)

            def OL(x):
                if A[0] < x:
                    return self.RectLine(x)
                elif A[0]> x:
                    return self._linePoints(A, (self.xW,self.xW))(x)

            self.OpLine = OL

    def _YXsolver(self,X):
        '''
        Helper function for finding the x of y equal to given X
        Simlarly to drawing a horizantal line from (X,X) and finding the intersection point with (Y,nX)
        '''
        Y = X
        def func(x):
            return self.y_func(x) - Y
        
        nX = root(func, 0.01)['x'][0]
        return nX


##########################################################################
    def ComputeN(self, maxIter=20):
        """
        Computes number of stages needed for distillation
        """
        self.plot_Data()

        X = self.xD
        OpY = self.xD
        countN = 0
        while X > self.xW:
            countN += 1
            nX = self._YXsolver(OpY)
            nXY = self.y_func(nX)
            self._drawPoint(nX, nXY,color = self.StageP)
            self._drawLine((X,OpY),(nX, nXY),style=self.StageLine+'-')
            
            OpY = self.OpLine(nX)

            #last Stage
            if nX < self.xW:
                self._drawLine((nX,nXY),(nX,nX),style = self.StageLine+'-') 
                break
            else:  
                self._drawLine((nX,OpY),(nX,nXY),style = self.StageLine+'-')
                X = nX

            if countN > maxIter:    #max Iteration loop break
                break

        data = {
            "Count Stages": countN,
            "F": self.F,
            "D": self.D,
            "W": self.W,
            "zF": self.zF,
            "xW": self.xW,
            "xD": self.xD,
            "R": self.R,
            "q": self.q,
            "Rectifying Slope": self.R/(self.R+1),
            "Stripping Slope": self._Slopefinder((self.xW,self.xW),self.qOpPoint),
            "Q-Line Slope": self.q/(self.q-1),
            "has Multiple Feed" : False,
            "has Side Stream": False
        }

        if self.MultipleFeed == True:
            data["zF2"] = self.zF2
            data["F2"] = self.F2
            data["q2"] = self.q2 
            data['Q2-Line Slope']=self.q2/(self.q2 - 1)
            data["mid Op-line Slope"] = self.midslope
            data["has Multiple Feed"] = True
        
        if self.SideStream == True:
            data["mid Op-line Slope"] = self.midslope
            data["has Side Stream"] = True 
            data["S"] = self.S
            data["xS"]: self.Xs

        return data
            

##########################################################################
    def findNmin(self):
        '''
        Computes Minimum number of stages needed for distillation
        '''
        self.plot_Data(plotOpLine=False)

        X = self.xD
        OpY = self.xD
        countN = 0
        while X > self.xW:
            countN += 1
            nX = self._YXsolver(OpY)
            nXY = self.y_func(nX)
            self._drawPoint(nX, nXY,color = self.StageP)
            self._drawLine((X,OpY),(nX, nXY),style=self.StageLine+'-')
            
            OpY = nX

            #last Stage
            if nX < self.xW:
                self._drawLine((nX,nXY),(nX,nX),style = self.StageLine+'-') 
                break
            else:  
                self._drawLine((nX,OpY),(nX,nXY),style = self.StageLine+'-')
                X = nX

        return countN
        
##########################################################################
    def _Slopefinder(self,A, B):
        '''
        helper function computes slope of line from A and B points
        '''
        xA, yA = A
        xB, yB = B

        #sloope of the line
        a = (yB - yA)/(xB - xA)
        return a

    def findRmin(self):
        '''
        Computes R minimum
        '''
        self.raw_plot()

        self._drawPoint(self.xW,self.xW, color = self.initPC)
        self._drawPoint(self.xD, self.xD, color = self.initPC)
        self._drawPoint(self.zF, self.zF, color = self.initPC)

        A = self._Intersection(self.y_func, self.qline)
        self._drawPoint(A[0],A[1], color=self.initPC)
        self._drawLine((self.zF,self.zF),A,style = self.initPC +':')

        self._drawLine((self.xW, self.xW),A,style = self.StageLine+'-')
        self._drawLine((self.xD, self.xD),A,style = self.StageLine+'-')

        recSlope = self._Slopefinder(A,(self.xD, self.xD))

        R = recSlope/(1 - recSlope)
        return R