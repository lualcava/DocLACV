#!/usr/bin/python
# -*- coding: latin-1 -*- 
"""
@author: Luis-Alexander Calvo-Valverde
"""

###############################################################################
# Imports
###############################################################################

import os
import sys
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import math
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.lda import LDA
from sklearn import preprocessing
from pykalman import KalmanFilter
from sklearn import grid_search
from sklearn import cross_validation
from collections import Counter
from scipy import stats
import itertools


###############################################################################
#  Funciones
###############################################################################

###############################################################################
def calcule_precision_recall(pOrigRespuestas, pOrigResultado):
    # Revisar: estas funciones de precision_score y recall_score si el pOrigRespuestas
    # tiene solo dos etiquetas da un error porque cree que es binaria 
    try: 
        pOrig = copy.deepcopy(pOrigRespuestas.round(0))
        pResu = copy.deepcopy(pOrigResultado.round(0))
        tPrecision = precision_score(pOrig, pResu, average='weighted')
    except ValueError:
        tEtiq = []
        for tIn in pOrig:
            if tIn not in tEtiq:
                tEtiq.append(tIn)
        try:
            tPrecision1 = precision_score(pOrig, pResu, average='weighted', pos_label=tEtiq[0])
            if len(tEtiq) > 1:
                tPrecision2 = precision_score(pOrig, pResu, average='weighted', pos_label=tEtiq[1])        
                tDiv = 2.0
            else:
                tDiv = 1.0
                tPrecision2 = 0.0
            tPrecision = (tPrecision1 + tPrecision2) / tDiv
            # las promedia
        except:
            tPrecision = 0
    try: 
        tRecall = recall_score(pOrig, pResu, average='weighted')
    except ValueError:
        tEtiq = []
        for tIn in pOrig:
            if tIn not in tEtiq:
                tEtiq.append(tIn)
        try:
            tRecall1 = recall_score(pOrig, pResu, average='weighted', pos_label=tEtiq[0])
            if len(tEtiq) > 1:
                tRecall2 = recall_score(pOrig, pResu, average='weighted', pos_label=tEtiq[1])        
                tDiv = 2.0
            else:
                tDiv = 1.0
                tRecall2 = 0.0
            tRecall = (tRecall1 + tRecall2) / tDiv
            # las promedia
        except:
            tRecall = 0
    return [tPrecision, tRecall]

###############################################################################
def calcule_Accuracy(pOrigRespuestas, pOrigResultado):  
    tResult = 0.0
    tAciertos = 0
    tLargo = len(pOrigRespuestas)
    for tCont in range(tLargo):
        if pOrigRespuestas[tCont] == pOrigResultado[tCont]:
            tAciertos += 1
    tResult = float(tAciertos)/float(tLargo)
    return tResult
        
###############################################################################
# Discretiza en los rangos de pRangosDis 
def DiscreticeEnRangos(pMatriz, pRangosDis):
    tRespuesta = np.array([])
    for tM in range(len(pMatriz)):
        for tIn in range(len(pRangosDis-1)):
            if (pMatriz[tM]>=pRangosDis[tIn]) and (pMatriz[tM]<=pRangosDis[tIn+1]) :
                tDato = tIn
                tRespuesta.append(tDato)
                break
    return tRespuesta
     
###############################################################################
# Calcula el RMSE    
def deme_rmer(pOriginal, pPredecido):
    tLargo = len(pOriginal)
    tTotal = 0
    for tI in range(tLargo):
        tTotal = tTotal + pow((float(pPredecido[tI])- float(pOriginal[tI])), 2)
    rmse = math.sqrt(tTotal/float(tLargo)) 
    return rmse

###############################################################################
# Calcula el R2  - Coeficiente de determinación
def calculeR2(pXorig, pYpred):
    # 1    stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(pXorig,pYpred)
    return r_value*r_value
    
    '''
    # Witten
    tn = len(pXorig)
    aMedia = np.sum(pXorig)/ tn
    pMedia = np.sum(pYpred)/ tn
    spa = np.sum( (pYpred-pMedia)*(pXorig-aMedia)) / (tn-1)
    sp = np.sum((pYpred - pMedia)**2) / (tn-1)
    sa = np.sum((pXorig - aMedia)**2) / (tn-1)    
    results =  spa / np.sqrt( sp * sa)       
    r2 = results**2
    return r2
    
    # 3 Inicial
    tn = len(pXorig)
    ybar = np.sum(pXorig)/ tn
    sse = np.sum((pXorig - pYpred)**2)  # Suma de cuadrados del error o residual
    ssr = np.sum((pYpred - ybar)**2)  # Suma de Cuadrados debido a la regresión
    sst = sse + ssr   # Suma de cuadrados total
    results =  (ssr / sst)  
    return results
    '''
    

###############################################################################
# Escala
def escaleDatosT(pM, tMaxColumnasDatos, pEscaleY ):
    global gEscalaParaVarAPredecir , gMetodoParaEscalar, gMinMaxScaler
    #  0:mean,rango   1:mean,max   2:mean,stdar  3:minMax con (-1,1)
    gEscalaParaVarAPredecir = copy.deepcopy(pM[:,(tMaxColumnasDatos-1) : ]).transpose()[0]  
    if pEscaleY :
        pMatriz = copy.deepcopy(pM[:,:])
    else:
        pMatriz = copy.deepcopy(pM[:,:(tMaxColumnasDatos-1)])
    if gMetodoParaEscalar == 0:
        xMedia = np.mean(pMatriz,axis=0)
        xRango =  pMatriz.max(axis=0) - pMatriz.min(axis=0)
        tResult = (pMatriz - xMedia) / xRango
    elif gMetodoParaEscalar == 1:    
        xMedia = np.mean(pMatriz,axis=0)
        xMax =  np.max(pMatriz, axis=0) 
        tResult = (pMatriz - xMedia) / xMax
    elif gMetodoParaEscalar == 2:    
        xMedia = np.mean(pMatriz,axis=0)
        xStd =  np.std(pMatriz, axis=0) 
        tResult = (pMatriz - xMedia) / xStd   
    elif gMetodoParaEscalar == 3:    
        xMin = pMatriz.min(axis=0)
        xRango =  pMatriz.max(axis=0) - pMatriz.min(axis=0)
        tResult = (pMatriz - xMin) / xRango        
    elif gMetodoParaEscalar == 4:    
        tMinMax = preprocessing.MinMaxScaler() 
        tResult = tMinMax.fit_transform(pMatriz) 
    elif gMetodoParaEscalar == 5:    
        xMin = pMatriz.min(axis=0)
        xRango =  pMatriz.max(axis=0) - pMatriz.min(axis=0)
        tTemp1 = (pMatriz - xMin) / xRango   
        tTemp2 = 2 * tTemp1
        tResult = -1 + tTemp2        
    tLargoF = len(tResult)
    tLargoC = len(tResult[0])    
    for tFil in range(tLargoF):
        for tCol in range(tLargoC):
            if np.isnan( tResult[tFil][tCol]):
                tResult[tFil][tCol] = 0.5
                print("Nan",tFil, tCol )
                sys.exit(40)
    if not ( pEscaleY ):        
        tResult = np.hstack((tResult,pM[: , (tMaxColumnasDatos-1) : ]))
    return tResult

###############################################################################
#  El método de escalar y el de desEscalar debe usar el mismo método
def desescaleT(pMatriz):
    global gEscalaParaVarAPredecir, gMetodoParaEscalar, gMinMaxScaler
    #  0:mean,rango   1:mean,max   2:mean,std  3:minMax con (-1,1)
    if gMetodoParaEscalar == 0:
        xMedia = np.mean(gEscalaParaVarAPredecir)
        xRango =  gEscalaParaVarAPredecir.max() - gEscalaParaVarAPredecir.min()
        tResult = (pMatriz * xRango) + xMedia
    elif gMetodoParaEscalar == 1:
        xMedia = np.mean(gEscalaParaVarAPredecir)
        xMax =  np.max(gEscalaParaVarAPredecir) 
        tResult = (pMatriz * xMax) + xMedia 
    elif gMetodoParaEscalar == 2:    
        xMedia = np.mean(gEscalaParaVarAPredecir)
        xStd =  np.std(gEscalaParaVarAPredecir) 
        tResult = (pMatriz * xStd) + xMedia     
    elif gMetodoParaEscalar == 3:    
        xMin = pMatriz.min(axis=0)
        xRango =  pMatriz.max(axis=0) - pMatriz.min(axis=0)
        tResult = (pMatriz * xRango) + xMin
        #tResult = gMinMaxScaler.inverse_transform(pMatriz)   
    elif gMetodoParaEscalar == 4:    
        tResult = -1
        #  Revisar no implementado
    elif gMetodoParaEscalar == 5:    
        xMin = pMatriz.min(axis=0)
        xRango =  pMatriz.max(axis=0) - pMatriz.min(axis=0)
        tResult = xMin + (xRango * ((pMatriz+1)/2))
        #  Revisar no implementado         
    return tResult
    
###############################################################################
# aplica la Echo State Network, ya teniendo los parámetros básicos
def CalculeConESN(pDataTrain, pDataTest, pCantidadVarAPredecir, pNeuronas, 
                  pLeakingRate, pInitLenPorcentaje ):    

    ######################################################
    #  Importante: parámetros para ESN fijos en este punto
    pTestLen = len(pDataTest) 
    pTrainLen = len(pDataTrain) - 1
    pNumColumnas = len(pDataTrain[0])
    pInSize = pNumColumnas - pCantidadVarAPredecir
    pOutSize = pCantidadVarAPredecir
    pInitLen = int(round(float(pTestLen)*pInitLenPorcentaje))   
    ######################################################
    
    pDataX = copy.deepcopy(pDataTrain[:, :pInSize])
    pDataY = copy.deepcopy(pDataTrain[:, pInSize])
 
    Win = (np.random.rand(pNeuronas, 1 + pInSize) - 0.5) * 1.0
    W = np.random.rand(pNeuronas, pNeuronas) - 0.5
    # normalizing and setting spectral radius (correct, slow):
    # Calcula y obtiene el valor propio mÃ¡ximo de la matriz de pesos W
    # Se puede optimizar usando la funciÃ³n:
    rhoW = max(abs(np.linalg.eigvals(W)))
    W *= 1.25 / rhoW
    # allocated memory for the design (collected states) matrix
    
    X = np.zeros((1 + pInSize + pNeuronas, pTrainLen - pInitLen))    
    
    # set the corresponding target matrix directly
    # Yt = data[None, initLen + 1:trainLen + 1]
    Yt = copy.deepcopy(pDataY[None, pInitLen + 1:pTrainLen + 1])
    # En Yt quedan las respuestas originales
    # run the reservoir with the data and collect X
    x = np.zeros((pNeuronas, 1))
    for t in range(pTrainLen):
        u = np.array([pDataX[t]])
        u = u.T
        t10 = np.vstack((np.array([[1]]), u))
        t13 = np.dot(Win, t10) + np.dot(W, x)
        t15 = pLeakingRate * np.tanh(t13)
        x = (1 - pLeakingRate) * x + t15
        if t >= pInitLen:
            t16 = np.vstack(([[1]], u, x))
            t17 = t - pInitLen
            X[:, t17] = copy.deepcopy(t16[:, 0])
        #        X[:, t - initLen] = vstack((1, u, x))[:, 0]
    # train the output
    reg = 1e-8  # regularization coefficient
    # Calculando transpuesta de la matriz
    X_T = X.T
    #Calculando los pesos de salida
    Wout = np.dot(np.dot(Yt, X_T), np.linalg.inv(np.dot(X, X_T) + \
                                    reg * np.eye(1 + pInSize + pNeuronas)))
    # run the trained ESN in a generative mode. no need to initialize here, 
    # because x is initialized with training data and we continue from there.
    Y = np.zeros((pOutSize, pTestLen))
    # en Y quedarán las respuetas al apicar el Test
    for t in range(pTestLen):
        u = np.array([pDataTest[t]])
        u = u.T
        x = (1 - pLeakingRate) * x 
        x += pLeakingRate * np.tanh(np.dot(Win, np.vstack(([[1]], u))) + np.dot(W, x))
        y = np.dot(Wout, np.vstack(([[1]], u, x)))
        Y[:, t] = copy.deepcopy(y)
        # generative mode:
        #u = y
        ## this would be a predictive mode:
    tRespuesta = copy.deepcopy(Y[0, 0:pTestLen])
    return tRespuesta

###############################################################################
# Elije los mejores parámetros para ESN
def ESN_Optimice_Params(pParameters, pDatos, pTamTrainingSet, pCantidadVarAPredecir ):
    tElMejorR2 = float(-np.inf)
    tOptNeu = 0.0
    tOptLRe = 0.0
    tOptInit = 0.0
    pDataTrain = copy.deepcopy(pDatos[:pTamTrainingSet,:])
    pDataTest = copy.deepcopy(pDatos[pTamTrainingSet:,:-pCantidadVarAPredecir])    
    py_train = copy.deepcopy(pDatos[pTamTrainingSet:,-pCantidadVarAPredecir:]) 
    py_train = [tCont0[0] for tCont0 in py_train]
    for tContN in pParameters['neuronas']:
        for tContL in pParameters['leakingrate']:
            for tContI in pParameters['InitLen']:               
                tResultadoESN = CalculeConESN(pDataTrain, pDataTest, pCantidadVarAPredecir, tContN, 
                  tContL, tContI )
                tR2 = calculeR2(py_train, tResultadoESN)
                if tR2 > tElMejorR2:
                    tElMejorR2 = tR2
                    tOptNeu = tContN
                    tOptLRe = tContL
                    tOptInit = tContI
    tRespuesta = {'neuronas': tOptNeu, 'leakingrate': tOptLRe, 'InitLen': tOptInit}
    return tRespuesta
    
###############################################################################
# Ejecute el nuevo método propuesto
def newMethod_entrene(pTrainingSet, pRespuestasTrainSet):
    ####################
    #  tResult
    #   0 Media de cada columna
    #   1 Desviación estándar de cada columna
    #   2 Coeficiencie de Variación de la variable
    #   3 Efecto medio de cada columna con respecto al valor predecido
    #   4 Efecto de la variable sobre el total del punto 3
    #   5 Desviación estándar del Efecto 3
    #   6 Coeficiente de variación del Efecto  3 / 5
    ####################
    tFilas = len(pTrainingSet)
    tColumnas = len(pTrainingSet[0])
    tResult = np.array([])
    # 0 la media
    tResult = np.mean(pTrainingSet,axis=0)
    # 1  la desviacion estandar
    tResult = np.vstack((tResult,np.std(pTrainingSet,axis=0)))
    # 2 el coeficiente de variacion
    tResult = np.vstack((tResult,tResult[1] / tResult[0]))
    # 3 Aporte del Efecto
    tResult = np.vstack((tResult,np.zeros(tColumnas)))
    # 4 Aporte promedio del Efecto
    tResult = np.vstack((tResult,np.zeros(tColumnas)))
    # 5  desviacion estandar del efecto
    tResult = np.vstack((tResult,np.zeros(tColumnas)))
    # 6  coeficiente de variación del efecto
    tResult = np.vstack((tResult,np.zeros(tColumnas)))
    
    tTodosLosEfectos = []
    for tFil in range(tFilas):
        tTemp = np.zeros(tColumnas)
        for tCol in range(tColumnas):
            tTemp[tCol] = pRespuestasTrainSet[tFil] * pTrainingSet[tFil][tCol]    
        tTotalTemp =  np.sum(tTemp)
        tEfectoTemp = np.zeros(tColumnas)
        for tCol in range(tColumnas):
            tEfectoTemp[tCol] =    tTemp[tCol] / tTotalTemp
        tTodosLosEfectos.append(tEfectoTemp)
        tResult[3] = tResult[3] + tTemp
        tResult[4] = tResult[4] + tEfectoTemp
    tTodosLosEfectos= np.array(tTodosLosEfectos)
    tResult[3] = tResult[3] / tFilas
    tResult[4] = tResult[4] / tFilas
    tResult[5] = np.std(tTodosLosEfectos,axis=0)      
    tResult[6] = tResult[5] / tResult[3]
    promedioRespuesta = np.mean(pRespuestasTrainSet)
    return ( tResult, promedioRespuesta )

###############################################################################
def newMethod_prediga(pEntrenadoNewMethodTT, pCrossValidationSet, pLimite):
    pEntrenadoNewMethod = copy.deepcopy(pEntrenadoNewMethodTT[0])
    pPromedioRes = copy.deepcopy(pEntrenadoNewMethodTT[1])
    tFilas = len(pCrossValidationSet)
    tColumnas = len(pCrossValidationSet[0])    
    tResult = []
    for tFil in range(tFilas):
        tTemp = []
        for tCol in range(tColumnas):
            if pEntrenadoNewMethod[6][tCol] <= pLimite:
                tValor = pEntrenadoNewMethod[4][tCol] * pPromedioRes
                tTemp.append(tValor)                        
            else:
                tDif =    pCrossValidationSet[tFil][tCol]  - pEntrenadoNewMethod[0][tCol]  
                tValor1 =  pEntrenadoNewMethod[4][tCol] * pPromedioRes
                tValor2 = tValor1 + ((tDif / pEntrenadoNewMethod[0][tCol]) * tValor1 )
                tTemp.append(tValor2)                        
        tTotal = np.sum(tTemp)
        tResult.append(tTotal)        
    tResult = np.array(tResult)    
    return tResult

###############################################################################
# Ejecute el nuevo método propuesto  2
def newMethod_2_entrene(pTrainingSet, pRespuestasTrainSet):
    ####################
    #  tResult
    #   0 Media de cada columna
    #   1 Desviación estándar de cada columna
    #   2 Coeficiencie de Variación de la variable
    #   3 Efecto medio de cada columna con respecto al valor predecido
    #   4 Efecto de la variable sobre el total del punto 3
    #   5 Desviación estándar del Efecto 3
    #   6 Coeficiente de variación del Efecto  5 / 3
    #   7 Valor máximo de la variable
    ####################
    tFilas = len(pTrainingSet)
    tColumnas = len(pTrainingSet[0])
    tResult = np.array([])
    # 0
    tResult = np.mean(pTrainingSet,axis=0)
    # 1
    tResult = np.vstack((tResult,np.std(pTrainingSet,axis=0)))
    # 2
    tResult = np.vstack((tResult,tResult[1] / tResult[0]))
    # 3
    tResult = np.vstack((tResult,np.zeros(tColumnas)))
    # 4
    tResult = np.vstack((tResult,np.zeros(tColumnas)))
    # 5
    tResult = np.vstack((tResult,np.zeros(tColumnas)))
    # 6
    tResult = np.vstack((tResult,np.zeros(tColumnas)))
    # 7
    tResult = np.vstack((tResult,np.max(pTrainingSet,axis=0))) 
    
    tTodosLosEfectos = []
    tListaMinimos = []
    for tFil in range(tFilas):
        tTemp = np.zeros(tColumnas)
        for tCol in range(tColumnas):
            tTemp[tCol] = pRespuestasTrainSet[tFil] * pTrainingSet[tFil][tCol]    
        tTotalTemp =  np.sum(tTemp)
        tEfectoTemp = np.zeros(tColumnas)
        tTotalAporteMax = 0
        for tCol in range(tColumnas):
            tEfectoTemp[tCol] =    tTemp[tCol] / tTotalTemp
            if tResult[7][tCol] != 0:
                tTotalAporteMax += ( abs(tResult[7][tCol]-
                                pTrainingSet[tFil][tCol]) / tResult[7][tCol] )
        tTodosLosEfectos.append(tEfectoTemp)
        tResult[3] = tResult[3] + tTemp
        tResult[4] = tResult[4] + tEfectoTemp
        tListaMinimos.append([tTotalAporteMax, pRespuestasTrainSet[tFil]])
        tResult[7] = tResult[4] + tEfectoTemp     
    tTodosLosEfectos= np.array(tTodosLosEfectos)
    tResult[3] = tResult[3] / tFilas
    tResult[4] = tResult[4] / tFilas
    tResult[5] = np.std(tTodosLosEfectos,axis=0)      
    tResult[6] = tResult[5] / tResult[3]
    promedioRespuesta = np.mean(pRespuestasTrainSet)
    if promedioRespuesta != 0:
        CVRespuesta = np.std(pRespuestasTrainSet) / promedioRespuesta
    else:
        CVRespuesta = 0
    tListaMinimos.sort()
    return ( tResult, promedioRespuesta, tListaMinimos, CVRespuesta )


###############################################################################
def newMethod_2_prediga(pEntrenadoNewMethodTT, pCrossValidationSet, pLimite):
    pEntrenadoNewMethod = pEntrenadoNewMethodTT[0]
    pPromedioRes = pEntrenadoNewMethodTT[1]
    pListaMinimos = pEntrenadoNewMethodTT[2]
    CVRespuesta = pEntrenadoNewMethodTT[3]
    tFilas = len(pCrossValidationSet)
    tColumnas = len(pCrossValidationSet[0])    
    tResult = []
    bCVVarieblePredMenorLimite = CVRespuesta <= pLimite
    tNumColumnas = len (pCrossValidationSet[0]) # todas las filas son del mismo largo
    for tFil in range(tFilas):
        tTemp = []
        tParaEfectos = np.zeros(tNumColumnas)
        for tInd in range(tNumColumnas):
            tParaEfectos[tInd] = pCrossValidationSet[tFil][tInd] * pPromedioRes
        tTotalDeEfectos = np.sum(tParaEfectos)
        tParaEfectoPorcentual = np.zeros(tNumColumnas)
        for tInd in range(tNumColumnas):
            tParaEfectoPorcentual[tInd] = tParaEfectos[tInd] / tTotalDeEfectos
        tActuaEntreMax = np.zeros(tNumColumnas)
        for tInd in range(tNumColumnas):
            tDiv = pEntrenadoNewMethod[7][tInd]   # el máximo de la columna
            if tDiv != 0:
                tActuaEntreMax[tInd]= abs(tDiv-pCrossValidationSet[tFil][tInd]) / tDiv
            else:
                tActuaEntreMax[tInd] = 0       
        tTotalDiferencia = np.sum(tActuaEntreMax)
        # Se inicializa con el máximo por si no lo encuentra
        tPosNuevoValor = len(pListaMinimos) - 1
        tPromedioMejor = pListaMinimos[tPosNuevoValor][1]
        for tInd in range(len(pListaMinimos)):
            if tTotalDiferencia < pListaMinimos[tInd][0]:
                tPosNuevoValor = tInd
                tPromedioMejor = pListaMinimos[tInd][1]
                break        
        for tCol in range(tColumnas):
            bCVEfectoMenorLimite = pEntrenadoNewMethod[6][tCol] <= pLimite
            tValVariable = pCrossValidationSet[tFil][tCol]
            tEfectoVarSobrePred = pEntrenadoNewMethod[4][tCol]
            tMediaVariable = pEntrenadoNewMethod[0][tCol]            
            if bCVVarieblePredMenorLimite:
                if bCVEfectoMenorLimite:
                    tValor = tValVariable / tMediaVariable *  tEfectoVarSobrePred * pPromedioRes                                       
                else:
                    tValor = tValVariable / tMediaVariable *  tParaEfectoPorcentual[tCol] * pPromedioRes
            else:
                if bCVEfectoMenorLimite:
                    tValor = tValVariable / tMediaVariable *  tEfectoVarSobrePred * tPromedioMejor                    
                else:
                    tValor = tValVariable / tMediaVariable * tParaEfectoPorcentual[tCol] * tPromedioMejor                    
            tTemp.append(tValor)                        
        tTotal = np.sum(tTemp)
        tResult.append(tTotal)        
    tResult = np.array(tResult)    
    return tResult

###############################################################################
def newMethod_2_SeleccioneParam(pEntrenadoNewMethodTT, pParameters,
                                pX_train, py_train ):
    tElMejorR2 = float(-np.inf)
    tElMejorPar =  0
    for tContP in pParameters:
        tResultado = newMethod_2_prediga(pEntrenadoNewMethodTT,pX_train,tContP)
        tR2 = calculeR2(py_train, tResultado)
        if tR2 > tElMejorR2:
            tElMejorR2 = tR2
            tElMejorPar = tContP
    return tElMejorPar 

###############################################################################
#  Usando regresión      ( Xt *  X )'-1   *  Xt  *  y
def newMethod_3_entrene(pTrainingSet, pRespuestasTrainSet):
    tTrans = pTrainingSet.transpose()
    tParte1 = np.dot(tTrans , pTrainingSet)
    tParte2 = np.linalg.inv( tParte1) 
    tParte3 = np.dot(tParte2 , tTrans )
    tResult = np.dot(tParte3 , pRespuestasTrainSet)
    return tResult
    
###############################################################################
def newMethod_3_prediga(pEntrenadoNewMethodTT, pCrossValidationSet):
    tResult = np.dot(pCrossValidationSet , pEntrenadoNewMethodTT)
    return tResult
        
    
###############################################################################
def GD_computeCostMulti(X, y, theta):
    #  X ya trae un 1 en toda la primera columna para el theta 0
    m = len(y) 
    J = 0.0
    for j in range(m):
        temp1 = np.dot( theta.transpose() , X[j].transpose() )
        temp2 = temp1 - y[j]
        temp3 = np.power( temp2 , 2)
        J = J + temp3
    J = 1.0 / (2.0 * float(m) ) * J;
    return J

###############################################################################
def GD_entrene(X, y, theta, alpha, num_iters):
    m = float(len(y))
    J_history = np.zeros((num_iters, 1))
    for iter in range (num_iters):
        temp1 = np.dot(X , theta)
        temp2 = ( temp1 - y).transpose()
        temp3 = np.dot(temp2 , X).transpose()
        theta = theta - ( alpha * (1.0/m)* temp3)
        J_history[iter] = GD_computeCostMulti(X, y, theta)
        ###################################
        if iter > 0:
            if (J_history[iter-1] - J_history[iter]) < 0.0000001:
                print("ya no mejora, iter: ", iter)
                break
        ###################################
    return [theta, J_history]

###############################################################################
def GD_prediga(pEntrenadoNewMethodTT, pCrossValidationSet):
    # en 0 el theta, en  1l J_history
    tResult = np.dot(pCrossValidationSet , pEntrenadoNewMethodTT[0])
    return tResult

###############################################################################
# Implementa Dynamic Time Warping
def path_cost(x, y, accumulated_cost, distances):
    path = [[len(x)-1, len(y)-1]]
    cost = 0
    i = len(y)-1
    j = len(x)-1
    while i>0 and j>0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                i = i - 1
            elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                j = j-1
            else:
                i = i - 1
                j= j- 1
        path.append([j, i])
    path.append([0,0])
    for [y, x] in path:
        cost = cost +distances[x, y]
    return path, cost    

#######################
# Hay dos mètodo para calcular el resultado final
# uno es tomar el valor de la tupla màs parecida y la otra es por
# peso relativo de la variable
#######################
def CalculeDTW(pTrainingSet, pRespuestasTrainSet, pCrossValidationSet):    
    # cercano, peso
    tTipoCalculo = "peso"
    tLargoCross = len(pCrossValidationSet)
    tLargoTrain = len(pTrainingSet)    
    tResult = []
    for tContCross in range(tLargoCross):
        tListaCostos = []
        for tContTrain in range(tLargoTrain):
            x = pCrossValidationSet[tContCross]
            y = pTrainingSet[tContTrain]
            distances = np.zeros((len(y), len(x)))
            for i in range(len(y)):  # Distancia Euclideana
                for j in range(len(x)):
                    distances[i,j] = (x[j]-y[i])**2          
            accumulated_cost = np.zeros((len(y), len(x)))
            accumulated_cost[0,0] = distances[0,0]
            for i in range(1, len(x)):
                accumulated_cost[0,i] = distances[0,i] + accumulated_cost[0, i-1]    
            for i in range(1, len(y)):
                accumulated_cost[i,0] = distances[i, 0] + accumulated_cost[i-1, 0]           
            for i in range(1, len(y)):
                for j in range(1, len(x)):
                    accumulated_cost[i, j] = min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]) + distances[i, j]
            path = [[len(x)-1, len(y)-1]]
            i = len(y)-1
            j = len(x)-1
            while i>0 and j>0:
                if i==0:
                    j = j - 1
                elif j==0:
                    i = i - 1
                else:
                    if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                        i = i - 1
                    elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                        j = j-1
                    else:
                        i = i - 1
                        j= j- 1
                path.append([j, i])
            path.append([0,0])
            path_x = [point[0] for point in path]
            path_y = [point[1] for point in path]
            path, cost = path_cost(x, y, accumulated_cost, distances)
            tListaCostos.append([cost, tContCross, tContTrain])
        tListaCostos = sorted(tListaCostos) # el menor quede de primero
        if tTipoCalculo == "cercano":
            tResult.append(pRespuestasTrainSet[tListaCostos[0][2]])
        else:   # peso
            tTuplaTrain = pTrainingSet[tListaCostos[0][2]]
            tTuplaCross = pCrossValidationSet[tListaCostos[0][1]]
            tValorTrain = pRespuestasTrainSet[tListaCostos[0][2]]
            tLargoTupla = len(tTuplaTrain)
            tTotal1 = 0
            tValores1 = np.zeros(tLargoTupla)
            for tCont3 in range(tLargoTupla):
                tValores1[tCont3] = tTuplaTrain[tCont3] * tValorTrain
                tTotal1 = tTotal1 = tValores1[tCont3]
            tTotal2 = 0
            tValores2 = np.zeros(tLargoTupla)
            for tCont4 in range(tLargoTupla):
                tValores2[tCont4] = tTuplaCross[tCont4] * tValorTrain
                tTotal2 = tTotal2 = tValores2[tCont4]          
            if tTotal1 == 0:
                tPeso = 0
            else:
                tPeso = tValorTrain * tTotal2 / tTotal1
            tResult.append(tPeso)
    return tResult

###############################################################################
# Kalman Filter
def Calcule_Kalman_Filter(pTrainingSet, pRespuestasTrainSet, pCrossValidationSet):
    tCantidadEstados = len(pTrainingSet[0])
    tLargoCross = len(pCrossValidationSet)
    tResult = []
    tTraining = np.array(pRespuestasTrainSet).reshape((len(pTrainingSet), 1))
    my_data = np.concatenate((pTrainingSet,tTraining), axis=1)
    measurements = np.asarray(my_data)  
    kf = KalmanFilter(n_dim_obs=len(measurements[0]), n_dim_state=tCantidadEstados )
    kf = kf.em(measurements, n_iter=10, em_vars="all")
    snPredicho = list(kf.sample(tLargoCross))        
    tResult.append(snPredicho[1][:, -1:])
    tTemp1 = np.array(tResult)
    tTemp2 = tTemp1.flatten()
    tResult = list(tTemp2)
    return tResult

######################################################################
#  Entropía y Ganancia de informacion
######################################################################

def calculeEntropia_NS(p, n): 
    p = float(p)
    n = float(n)
    if p == 0:
        respuesta = ( -(n/(p+n)) *  math.log((n/(p+n)), 2))
    elif n == 0:
        respuesta = (-( p / (p+n) ) * math.log((p/(p+n)) , 2))
    else:
        respuesta = (-( p / (p+n) ) * math.log((p/(p+n)) , 2))   + ( -(n/(p+n)) *  math.log((n/(p+n)), 2))
    return respuesta

######################################################################

def calculeLimiteEntropia_NS(N, E, E1, E2, k, k1,k2):
    respuesta = (math.log(N-1,2)/N) + ( math.log((3**k)-2,2)-k*E + k1*E1 + k2*E2 )/N 
    return respuesta

######################################################################    

def demeCantidadTotal_NS(pLista, pDiccionario):
    tRespuesta = 0
    for tValor in pLista:
        tRespuesta += pDiccionario[tValor]
    return tRespuesta
    
######################################################################    

def EntropiasRangos_NS(pListaValores, pFrecuencias):
    respuesta = {}   
    todos = copy.deepcopy(pListaValores)   
    todos.sort()
    medio = len(todos) // 2
    valorCorte = todos[medio]    
    respuesta["medio"] = valorCorte
    respuesta["inferioresEiguales"] = []
    respuesta["superiores"] = []
    tN = len(todos)
    for tCont in range(tN):      
        if todos[tCont] <= valorCorte:
            respuesta["inferioresEiguales"].append(todos[tCont])
        else:
            respuesta["superiores"].append(todos[tCont])
    respuesta["nInferioresEiguales"] = demeCantidadTotal_NS( respuesta["inferioresEiguales"], pFrecuencias )
    respuesta["nSuperiores"] = demeCantidadTotal_NS( respuesta["superiores"] , pFrecuencias)
    respuesta["Entropia"] = calculeEntropia_NS(respuesta["nInferioresEiguales"], 
                                            respuesta["nSuperiores"]  )
    return respuesta       

######################################################################
def inserteOrdenado(pLista, pValor):
    # Precondición es que la lista viene ordenada
    # No se insertan repetidos
    pNuevaL = []
    pLargo = len(pLista)
    if pLargo == 0:
        pNuevaL.append(pValor)
    elif pValor in pLista:
        pNuevaL = copy.deepcopy(pLista)
    else:
        for tCont in range(pLargo):
            if pLista[tCont] < pValor:
                pNuevaL.append(pLista[tCont])
                if tCont == (pLargo-1):
                    pNuevaL.append(pValor)
            else:
                pNuevaL.append(pValor)
                pNuevaL = pNuevaL + pLista[tCont:]
                break
    return pNuevaL

######################################################################

def calculeAporteSeccion_NS(pRangos, tLista, pEntropiaTodos, pN, pMinimoAporteInformacion, tFrecuencias):
    tEntropia1 = EntropiasRangos_NS(tLista, tFrecuencias)    
    nInferioresEigualesI = tEntropia1["nInferioresEiguales"]
    nSuperiores = tEntropia1["nSuperiores"]
    tEntropiaI = EntropiasRangos_NS(tEntropia1["inferioresEiguales"], tFrecuencias)    
    tEntropiaInferioresEiguales = tEntropiaI["Entropia"]
    tEntropiaS = EntropiasRangos_NS(tEntropia1["superiores"], tFrecuencias)    
    tEntropiaSuperiores = tEntropiaS["Entropia"] 
    tAporteI = pEntropiaTodos -(float(nInferioresEigualesI)/float(pN))*tEntropiaInferioresEiguales -(float(nSuperiores)/float(pN))*tEntropiaSuperiores                 
    #print("Aporte de informacion:" , tAporteI) 
    if tAporteI < pMinimoAporteInformacion:
        pRangos = inserteOrdenado(pRangos, tEntropia1["medio"])
        if len(tEntropia1["inferioresEiguales"]) > 3:
            pRangos = calculeAporteSeccion_NS(pRangos , tEntropia1["inferioresEiguales"], pEntropiaTodos, pN, pMinimoAporteInformacion, tFrecuencias)
        if len(tEntropia1["superiores"]) > 3:            
            pRangos = calculeAporteSeccion_NS(pRangos , tEntropia1["superiores"], pEntropiaTodos, pN, pMinimoAporteInformacion, tFrecuencias)       
    return pRangos

###############################################################################

def hagaDiscretizacion_NS(tDatosCol):    
    # Lo hace de manera no-supervisada, sin considerar las clases de la
    # variable a predecir
    tRespuestas = {}
    tRangos = []
    tFrecuencias = Counter(tDatosCol)
    N = len(list(tDatosCol))
    k = len(tFrecuencias)
    print("N: ", N)
    print("k: ", k)            
    tValores = list(tFrecuencias)
    tPrimero = min(tValores)
    tRangos.append(tPrimero) # para que el valor menor quede como límite inferior
    print("Cantidad de clases: ", k)
    tEntropia = EntropiasRangos_NS(tValores, tFrecuencias)    
    entropiaTodos = tEntropia["Entropia"]      
    nInferioresEigualesI = tEntropia["nInferioresEiguales"]
    nSuperiores = tEntropia["nSuperiores"]
    print("Entropia de todo el conjunto: ", entropiaTodos)
    tEntropiaI = EntropiasRangos_NS(tEntropia["inferioresEiguales"], tFrecuencias)    
    tEntropiaInferioresEiguales = tEntropiaI["Entropia"]
    tEntropiaS = EntropiasRangos_NS(tEntropia["superiores"], tFrecuencias)    
    tEntropiaSuperiores = tEntropiaS["Entropia"] 
    print("Entropia InferioresEIguales", tEntropiaInferioresEiguales)
    print("Entropia Superiores", tEntropiaSuperiores)        
    tAporteI = entropiaTodos -(float(nInferioresEigualesI)/float(N))*tEntropiaInferioresEiguales -(float(nSuperiores)/float(N))*tEntropiaSuperiores     
    print("Aporte de información:" , tAporteI) 
    k1 = len(Counter(tEntropia["inferioresEiguales"]))
    k2 = len(Counter(tEntropia["superiores"]))
    print("k1: ", k1)
    print("k2: ", k2)
    minimoAporteInformacion = calculeLimiteEntropia_NS(N, entropiaTodos, tEntropiaInferioresEiguales, tEntropiaSuperiores, k, k1, k2)    
    print("El aporte de información del rango debe ser mayor a: ", minimoAporteInformacion)                 
    
    print(tRangos)
    print(tEntropia["medio"] )
    print(max(tValores))
    
    tRangos = inserteOrdenado(tRangos, tEntropia["medio"])
    if tAporteI < minimoAporteInformacion:
        tRangos = calculeAporteSeccion_NS(tRangos , tEntropia["inferioresEiguales"], entropiaTodos, N, minimoAporteInformacion, tFrecuencias)
        tRangos = calculeAporteSeccion_NS(tRangos , tEntropia["superiores"], entropiaTodos, N, minimoAporteInformacion, tFrecuencias)       
    tUltimo = max(tValores)
    tRangos.append(tUltimo) # para que el valor mayor quede como límite mayor
    tCuantosRangos = len(tRangos)
    
    print("Rangos: ", tCuantosRangos)
    
    tLista = []
    for tI in range(tCuantosRangos-1): # Los valores discretizados inician en 0 y son n-1
        tLista.append(tI)
    tCopiaDatosCol = tDatosCol.copy()
    tRespuestas["Valores"] = pd.cut(x=tDatosCol,bins=tRangos, right=True, 
                          labels=tLista, include_lowest=True)
    tRespuestas["Rangos"] = pd.cut(x=tCopiaDatosCol,bins=tRangos, right=True, 
                          labels=tLista, include_lowest=True, retbins = True)
    return tRespuestas

###############################################################################

def calculeEntropia_S(p, n): 
    p = float(p)
    n = float(n)
    if p == 0:
        respuesta = ( -(n/(p+n)) *  math.log((n/(p+n)), 2))
    elif n == 0:
        respuesta = (-( p / (p+n) ) * math.log((p/(p+n)) , 2))
    else:
        respuesta = (-( p / (p+n) ) * math.log((p/(p+n)) , 2))   + ( -(n/(p+n)) *  math.log((n/(p+n)), 2))
    return respuesta

######################################################################

def calculeLimiteEntropia_S(N, E, E1, E2, k, k1,k2):
    respuesta = (math.log(N-1,2)/N) + ( math.log((3**k)-2,2)-k*E + k1*E1 + k2*E2 )/N 
    return respuesta

######################################################################    

def demeCantidadTotal_S(pLista, pDiccionario):
    tRespuesta = 0
    for tValor in pLista:
        tRespuesta += pDiccionario[tValor]
    return tRespuesta
    
######################################################################    

def EntropiasRangos_S(pListaValores, pFrecuencias):
    respuesta = {}   
    todos = copy.deepcopy(pListaValores)   
    todos.sort()
    medio = len(todos) // 2
    valorCorte = todos[medio]    
    respuesta["medio"] = valorCorte
    respuesta["inferioresEiguales"] = []
    respuesta["superiores"] = []
    tN = len(todos)
    for tCont in range(tN):      
        if todos[tCont] <= valorCorte:
            respuesta["inferioresEiguales"].append(todos[tCont])
        else:
            respuesta["superiores"].append(todos[tCont])
    respuesta["nInferioresEiguales"] = demeCantidadTotal_S( respuesta["inferioresEiguales"], pFrecuencias )
    respuesta["nSuperiores"] = demeCantidadTotal_S( respuesta["superiores"] , pFrecuencias)
    respuesta["Entropia"] = calculeEntropia_S(respuesta["nInferioresEiguales"], 
                                            respuesta["nSuperiores"]  )
    return respuesta       

######################################################################

def calculeAporteSeccion_S(pRangos, tLista, pEntropiaTodos, pN, pMinimoAporteInformacion, tFrecuencias):
    tEntropia1 = EntropiasRangos_S(tLista, tFrecuencias)    
    nInferioresEigualesI = tEntropia1["nInferioresEiguales"]
    nSuperiores = tEntropia1["nSuperiores"]
    tEntropiaI = EntropiasRangos_S(tEntropia1["inferioresEiguales"], tFrecuencias)    
    tEntropiaInferioresEiguales = tEntropiaI["Entropia"]
    tEntropiaS = EntropiasRangos_S(tEntropia1["superiores"], tFrecuencias)    
    tEntropiaSuperiores = tEntropiaS["Entropia"] 
    tAporteI = pEntropiaTodos -(float(nInferioresEigualesI)/float(pN))*tEntropiaInferioresEiguales -(float(nSuperiores)/float(pN))*tEntropiaSuperiores                 
    #print("Aporte de informacion:" , tAporteI) 
    if tAporteI < pMinimoAporteInformacion:
        pRangos = inserteOrdenado(pRangos, tEntropia1["medio"])
        if len(tEntropia1["inferioresEiguales"]) > 3:
            pRangos = calculeAporteSeccion_S(pRangos , tEntropia1["inferioresEiguales"], pEntropiaTodos, pN, pMinimoAporteInformacion, tFrecuencias)
        if len(tEntropia1["superiores"]) > 3:            
            pRangos = calculeAporteSeccion_S(pRangos , tEntropia1["superiores"], pEntropiaTodos, pN, pMinimoAporteInformacion, tFrecuencias)       
    return pRangos

###############################################################################

def hagaDiscretizacion_S(tDatosCol, pDatosClases):    
    # Lo hace de manera supervisada, considerarando las clases de la
    # variable a predecir
    tRespuestas = {}
    tRangos = []
    tFrecuencias = Counter(tDatosCol)  
    tClasesVal = {}
    N = len(list(tDatosCol))
    for tCont1 in range(N):
        if tDatosCol[tCont1] in tClasesVal:
            if pDatosClases[tCont1] not in tClasesVal[tDatosCol[tCont1]]:
                tClasesVal[tDatosCol[tCont1]].append( pDatosClases[tCont1])
        else:
            tClasesVal[tDatosCol[tCont1]] = [pDatosClases[tCont1]]
    #print(tClasesVal)       
    tValor_Clases = {}     
    for tValor in tClasesVal:
        tValor_Clases[tValor] = len(tClasesVal[tValor])
    tNumClases = Counter(pDatosClases)
  
    print(len(tFrecuencias))
    print(len(tValor_Clases))      
    print(len(tNumClases))
    
    k = len(tNumClases)
    
    print("N: ", N)       
    
    tValores = list(tFrecuencias)
    tPrimero = min(tValores)
    tRangos.append(tPrimero) # para que el valor menor quede como límite inferior
    
    print("Cantidad de clases: ", k)
    
    tEntropia = EntropiasRangos_S(tValores, tValor_Clases )    
    entropiaTodos = tEntropia["Entropia"]      
    nInferioresEigualesI = tEntropia["nInferioresEiguales"]
    nSuperiores = tEntropia["nSuperiores"]

    print("Entropia de todo el conjunto: ", entropiaTodos)

    tEntropiaI = EntropiasRangos_S(tEntropia["inferioresEiguales"], tValor_Clases )    
    tEntropiaInferioresEiguales = tEntropiaI["Entropia"]
    tEntropiaS = EntropiasRangos_S(tEntropia["superiores"], tValor_Clases )    
    tEntropiaSuperiores = tEntropiaS["Entropia"] 

    print("Entropia InferioresEIguales", tEntropiaInferioresEiguales)
    print("Entropia Superiores", tEntropiaSuperiores)        

    tAporteI = entropiaTodos -(float(nInferioresEigualesI)/float(N))*tEntropiaInferioresEiguales -(float(nSuperiores)/float(N))*tEntropiaSuperiores     

    print("Aporte de información:" , tAporteI) 

    k1 = tEntropia["nInferioresEiguales"]
    k2 = tEntropia["nSuperiores"]
    
    print("k1: ", k1)
    print("k2: ", k2)
    
    minimoAporteInformacion = calculeLimiteEntropia_S(N, entropiaTodos, tEntropiaInferioresEiguales, tEntropiaSuperiores, k, k1, k2)    

    print("El aporte de información del rango debe ser mayor a: ", minimoAporteInformacion)                 
    
    print(tRangos)
    print(tEntropia["medio"] )
    print(max(tValores))
        
    if tAporteI < minimoAporteInformacion:
        tRangos = inserteOrdenado(tRangos, tEntropia["medio"])
        tRangos = calculeAporteSeccion_S(tRangos , tEntropia["inferioresEiguales"], entropiaTodos, N, minimoAporteInformacion, tValor_Clases)
        tRangos = calculeAporteSeccion_S(tRangos , tEntropia["superiores"], entropiaTodos, N, minimoAporteInformacion, tValor_Clases)       
    tUltimo = max(tValores)
    tRangos.append(tUltimo) # para que el valor mayor quede como límite mayor
    tCuantosRangos = len(tRangos)
    
    print("Rangos: ", tCuantosRangos)

    #print(tRangos)

    tLista = []
    for tI in range(tCuantosRangos-1): # Los valores discretizados inician en 0 y son n-1
        tLista.append(tI)
    tCopiaDatosCol = tDatosCol.copy()
    tRespuestas["Valores"] = pd.cut(x=tDatosCol,bins=tRangos, right=True, 
                          labels=tLista, include_lowest=True)
    tRespuestas["Rangos"] = pd.cut(x=tCopiaDatosCol,bins=tRangos, right=True, 
                          labels=tLista, include_lowest=True, retbins = True)
    return tRespuestas
    
###############################################################################
###############################################################################
    
# Ejecuta los cálculos para los datos indicados
def EjecuteSVR_ESN(tDirectorio, tLugar, tArchivoEntrada, tArchivoConfig, pTipoProceso,
                   pTipoPatron,  pTipoAlgoritmo, pTipoMetodo ):
    global gGeneral, gEscalaParaVarAPredecir, gCantidadRangosVarAPredecir, gTipoDiscretizacion
    global gEscalaTres, gEscalaCinco, gEscaleDatosEntreAyB, gEscaleY, gMuestreAvance
    global gNumIteraciones, gCantidadVarAPredecir, gMaxPeriodsAntes, gMaxSemenasPrediccion
    global gSVR_C, gSVR_gamma, g_degree,  gLeakingRate, gLimiteNM, gNumFigura, gGenereArchivoDiscretizado
    global gHagaEntropia, gHagaRoughSet, gHagaReducida, gHagaNecesarias, gHagaPares, gResultadosSiguiente
    global gHagaValidacion, gCuantosPeriodosValidar, gListaValoresReales, gResultadoValidacion
    global gDiscreticeEspecial, gGenereArchivoConPatronDefinido, gNombreArchivoConPatronDefinido
    global gIncluyaPredichoEnPatron, gTrabajarComoTasaCambio, gGenereArchivosDePatrones
    global gOptimiceParametros, gInitLenPorcentaje, gHagaCombinatoria
    
    tNombrArchDiscretizado = "Discretizado-"+ tLugar +  " - "+ str(time.time()) + ".txt"
    os.chdir(tDirectorio)
    #######################################
    # Archivos de bitácora
    tNombreBitacora = "MLAPC " + tLugar +  " - "+ str(time.time()) + ".txt"
    tBitacora = open(tNombreBitacora,"w")
    tBitacora.write("**************************************************\n")
    tBitacora.write("Por: Luis-Alexander Calvo-Valverde\n")
    tBitacora.write("Proyecto asociado a su tesis doctoral en DOCINADE\n")
    tBitacora.write("Derechos reservados\n")
    tBitacora.write("**************************************************\n\n")
    tBitacora.write(str(time.strftime("%a, %d %b %Y %H:%M:%S")) + "\n\n" ) 
    tVariableDiscretizadora = 'rangosdiscretizar'
    tVariableNecesaria = 'necesaria'
    tVariableEntropia = "entropia"
    tVariableRoughSet = "roughset"
    tVariableReducida = "reducida"
    tVariableAPredecir = "predecir"  # este es el nombre de la columna en Config que guarda la variable a predecir
    tNombreVariableAPredecir = ""  # Este es el nombre físico de la variable a predecir
    tRangosVariableAPredecir = []  # Guarda los rangos definidos para discretizar la variable a Predecir
    tVariableMinimo = "minimo"
    tVariableMaximo = "maximo"
    tVariableDiscretice = "discretice"  # indica si se discretiza o no.  Si es "s" usa el valor
                                        # en la variable tVariableDiscretizadora
    tBitacora.write("Archivo de datos: "+ tArchivoEntrada+ "\n")
    tBitacora.write("Archivo de configuración: "+ tArchivoConfig+ "\n")
    tBitacora.write("Escalas: "+str(gEscalaTres) + "\n")
    tBitacora.write("Escalas: "+ str(gEscalaCinco)+ "\n")
    tGenereGraficos = False
    
    #######################################
    #  Patrones
    # En este momento permite máximo dos dígitos
    tBasePat = "S-Period-P"
    tPatrones = []
    for tPatI1 in range(1,gMaxPeriodsAntes+1):  # el +1 es porque python es base 0
        for tPatI2 in range(1,gMaxSemenasPrediccion+1):
            #if tPatI1 >= tPatI2: # Es para los casos que no tiene sentido predecir más periodos que los que se tienen
            tPatrones.append(str("%2.0f" % tPatI1)+ tBasePat + str("%2.0f" % tPatI2))
    tBitacora.write("Patrones aplicados: "+ str(tPatrones)+ "\n")
    tBitacora.write("Total de Patrones aplicados: "+ str(len(tPatrones))+ "\n")    
    
    #######################################
    #  Algoritmos
    tAlgoritmo1 = "SVR model con Kernel rbf"
    tAlgoritmo2 = "SVR model con Kernel linear"
    tAlgoritmo3 = "SVR model con Kernel sigmoid"  
    tAlgoritmo4 = "ESN"
    tAlgoritmo5 = "LinearRegression"  
    tAlgoritmo6 = "ElasticNet" 
    tAlgoritmo7 = "Ridge" 
    tAlgoritmo8 = "LDA" 
    tAlgoritmo9 = "newMethod" 
    tAlgoritmo10 = "newMethod2" 
    tAlgoritmo11 = "newMethod3" 
    tAlgoritmo12 = "gradientDescent" 
    tAlgoritmo13 = "BayesianRidge"     
    tAlgoritmo14 = "DynamicTimeWarping"     
    tAlgoritmo15 = "Kalman_Filter"
    tAlgoritmo16 = "SVR model con Kernel poly"


    tAlgoritmos = [ tAlgoritmo2 ]   
    
    tAlgoritmos = [ tAlgoritmo1, tAlgoritmo3, tAlgoritmo4, tAlgoritmo5,
                    tAlgoritmo6, tAlgoritmo7,  tAlgoritmo10,
                    tAlgoritmo13, tAlgoritmo14, tAlgoritmo15 ]

    tAlgoritmos = [ tAlgoritmo5, tAlgoritmo6, tAlgoritmo7,
                   tAlgoritmo10 ]                 

 
    
    #######################################
    #  Para cuando se llama desde la línea de comandos con el Algoritmo            
    if gCuantosParam == 2:
        tAlgoritmos = [ gListaParam[1]  ]
    #######################################
                   
    tBitacora.write("Algoritmos aplicados: "+ str( tAlgoritmos)+ "\n")
    
    #######################################
    #  Configuración de parámetros importantes (fijos)
    tSVR_C = gSVR_C
    tSVR_gamma = gSVR_gamma 
    t_degree = g_degree
    pLeakingRate = gLeakingRate
    pLimiteNM = gLimiteNM
    tNumFigura = gNumFigura
    gGenereArchivoDiscretizado = gGenereArchivoDiscretizado

    
    ###############################################################################
    # Cargar archivo de  datos y preparar DataFrame
    tMatrizDatos = pd.io.parsers.read_csv(tArchivoEntrada, sep=gSeparadorLista, header = 0)  # En fila 0 títulos
    tNumFilas = len(tMatrizDatos.index)
    tNumColumnas = len(tMatrizDatos.columns)
    tMatrizDatos.index = list(range(0,tNumFilas))  # le pone numero a las filas iniciando en 0
    tBitacora.write("Cantidad de filas: "+ str(tNumFilas)+ "\n")
    tBitacora.write("Cantidad de columnas: "+ str(tNumColumnas)+ "\n")
    tBitacora.write("Cantidad de variables a predecir: "+ str(gCantidadVarAPredecir)+ "\n")
    if gMuestreAvance:
        print("Datos de Entrada")
        print("filas", tNumFilas)
        print("columnas", tNumColumnas)
        print("Cantidad de variables a predecir: ", gCantidadVarAPredecir)
     
    ###############################################################################
    # Carga datos de Configuracion
    tMatrizConfig = pd.io.parsers.read_csv(tArchivoConfig, sep=gSeparadorLista, header = 0, index_col=0)  # En fila 0 títulos
    tNumFilasConfig = len(tMatrizConfig.index)
    tNumColumnasConfig = len(tMatrizConfig.columns)
      
    #######################################
    #  Métodos (Variables) a utilizar
    tMetodo1 = "Todas"  #  Se pone de primera pues respadla la base de datos.
    tMetodo2 = "NecesariasConfig"
    tMetodo3 = "Entropia"
    tMetodo4 = "RoughSet"    
    tVariablesUtilizar = [tMetodo1]
    if gHagaNecesarias:
        tVariablesUtilizar.append( tMetodo2 )        
    if gHagaEntropia:
        tVariablesUtilizar.append( tMetodo3 )
    if gHagaRoughSet:
        tVariablesUtilizar.append( tMetodo4 )     
    tListaReducidas = []
    # De una en Una de las Reducidas
    tMetodo5= "Unica*"
    if gHagaReducida:        
        for tNombreFil, tDatosFil in tMatrizConfig.iterrows():
            tValorSel = tMatrizConfig.loc[tNombreFil, tVariableReducida]
            if (tValorSel != "n"):
                if tMatrizConfig.loc[tNombreFil, tVariableAPredecir] != "s":
                    if (tMetodo5+tValorSel) not in tVariablesUtilizar:
                        tListaReducidas.append(tValorSel)
                        tVariablesUtilizar.append(tMetodo5+tValorSel)    # 1, 2, 3, 4, ...            
    # Pares
    tMetodo6 = "Par*"
    if gHagaPares:
        for tCont1 in range(len(tListaReducidas)-1):
            for tCont2 in range(tCont1+1,len(tListaReducidas) ):
                tVariablesUtilizar.append(tMetodo6+tListaReducidas[tCont1]+"*"+tListaReducidas[tCont2])      
            
    # Hace la combinatoria de las reducidas, a partir de 3 en 3, sin llegar a todas
    tMetodo7 = "Com*"
    if gHagaCombinatoria:
        tContenidoVar = tMatrizConfig[tVariableReducida]
        tTodasVar = []
        for tI1 in tContenidoVar:
            try:
                tValor=int(tI1)
                tTodasVar.append(tI1)
            except:
                pass
        tMayor= len(tTodasVar)
        for tVoy in range(3,tMayor):
            for tSubset in itertools.combinations(tTodasVar, tVoy):
                tTexto = "Com"
                for tI2 in tSubset:
                    tTexto = tTexto + "*" + str(tI2)
                tVariablesUtilizar.append(tTexto)

    ###############################################################################
    # Validando Outliers y contando missing values
    tBitacora.write("\n *************************** \n")  
    tTotOutliers = 0.0
    tTotMissing = 0.0
    tTotDatos = 0.0
    for tNombreFil, tDatosFil in tMatrizDatos.iterrows():
        tNombresCol = tMatrizDatos.columns
        for tCol in tNombresCol:
            tValor = tMatrizDatos.loc[tNombreFil, tCol]
            tTotDatos += 1
            if pd.notnull(tValor):
                if (tValor < tMatrizConfig.loc[tCol, tVariableMinimo]) or (tValor > tMatrizConfig.loc[tCol, tVariableMaximo]):
                    tMatrizDatos.loc[tNombreFil, tCol] = np.nan
                    tBitacora.write("Fila: "+str(tNombreFil)+" Columna: "+str(tCol) + " Valor: "+ str(tValor)+ "\n")
                    tTotOutliers = tTotOutliers + 1
                    if pd.isnull(tMatrizDatos.loc[tNombreFil, tCol]):
                        if gMuestreAvance:
                            print("Valor puesto en nulo: ", tNombreFil, " , ",tCol, " valor: ", tValor)
            else:
               tTotMissing += 1 
    tBitacora.write("Total de outliers: " + str(tTotOutliers) + "\n")  
    tBitacora.write("Porcentaje de Datos outliers: " + str(tTotOutliers/tTotDatos) + "\n")           
    tBitacora.write("*************************** \n")   

    tBitacora.write("\n *************************** \n")   
    tBitacora.write("Total de missing values: " + str(tTotMissing) + "\n")   
    tBitacora.write("Porcentaje de Missing Values: " + str(tTotMissing/tTotDatos) + "\n")          
    tBitacora.write("*************************** \n")      
    
    tBitacora.write("\n *************************** \n") 
    tBitacora.write("Total de Datos: " + str(tTotDatos) )      
    tBitacora.write("\n *************************** \n\n") 
    
    
    ###############################################################################
    # Completando datos
    #tMetodoInterpolar = 'nearest'  # no requiere order
    #tMetodoInterpolar = 'polynomial'  # en este caso hay que agregar order=?
    #tMetodoInterpolar = 'index'  # no requiere order
    tMetodoInterpolar= "spline"   # en este caso hay que agregar order=?
    tConfigGuardar = "order=1"
    for tNombreCol, tDatosCol in tMatrizDatos.iteritems():
        tMatrizDatos[tNombreCol] = tDatosCol.interpolate(method=tMetodoInterpolar, order=1)
    tBitacora.write("\n *************************** \n") 
    tBitacora.write("Método utilizado para interpolar: " + tMetodoInterpolar +
                    " ( "+ tConfigGuardar+ " )"+ "\n\n") 
    
    #############################################################################     
    #  Esta parte es para asegurarse que no quedaron valores nulos        
    for tNombreFil, tDatosFil in tMatrizDatos.iterrows():
        for tValor in tDatosFil:
            if pd.isnull(tValor):
                    print(tValor)                
                    print(tNombreFil)
                    print(tDatosFil)
                    print("Error hay valores Nulos presentes")
                    sys.exit(1)

    ############################################################################# 
    #  Pone la variable a predecir en la  ultima columna de la derecha
    for tNombreFil, tDatosFil in tMatrizConfig.iterrows():
        tValorPre = tMatrizConfig.loc[tNombreFil, tVariableAPredecir]
        if tValorPre == "s":
            tNombreVariableAPredecir = tNombreFil
            gCantidadRangosVarAPredecir = tMatrizConfig.loc[tNombreFil, tVariableDiscretizadora]
            tTempM = pd.DataFrame(tMatrizDatos[tNombreFil])
            del tMatrizDatos[tNombreFil]
            tMatrizDatos = tMatrizDatos.join(tTempM)   
    
    ############################################################################# 
    #  Se crea el archivo para guardar resultados, se incluye en el nombre 
    # la variable a predecir    
    gSVRoutfile = open( "MLSFBP-"+ tLugar +  " - "+ tNombreVariableAPredecir+" - "+
                    str(time.time()) + ".txt", "w") # Para RNN
    gSVRoutfile.write("Metodo;Patron;Algoritmo;Detalles;Train;Test;Num_Columnas;RMSE;R2;std-RMSE;CV-RMSE;std-R2;Accuracy;CV-R2;Precision;Recall"+"\n")
    
    ###############################################################################
    # Discretizando
    if gDiscreticeEspecial:  # Discretiza Atributos independiente de lo que diga el archivo de configuración 
        for tNombreCol, tDatosCol in tMatrizDatos.iteritems():
            if tNombreCol != tNombreVariableAPredecir:
                print(tNombreCol)
                tCopiaDatosCol = tDatosCol.copy()
                #tFactorizado = hagaDiscretizacion_NS(tDatosCol)
                tFactorizado = hagaDiscretizacion_S(tDatosCol, tMatrizDatos[tNombreVariableAPredecir])                
                tBitacora.write("\n\n Discretizacion: "+ tNombreCol + "\nCantidad de Rangos: " + 
                     str(len(tFactorizado["Rangos"])) + "\n" + str(tFactorizado["Rangos"]) + "\n")
                tMatrizDatos[tNombreCol] = copy.deepcopy(tFactorizado["Valores"])
    else:
        tDecisionDiscretizar = {}
        for tNombreCol, tDatosCol in tMatrizDatos.iteritems():
            tCuantosRangos = tMatrizConfig.loc[tNombreCol, tVariableDiscretizadora]
            tDecisionDiscretizar[tNombreCol]  = tMatrizConfig.loc[tNombreCol, tVariableDiscretice]
            if (tDecisionDiscretizar[tNombreCol] == "s") :   
                tLista = []
                for tI in range(tCuantosRangos): # Los valores discretizados inician en 0
                    tLista.append(tI)
                tCopiaDatosCol = tDatosCol.copy()
                tFactorizado = pd.cut(tDatosCol,tCuantosRangos,  labels=tLista)
                tRangosDis = pd.cut(tCopiaDatosCol,tCuantosRangos,  labels=tLista, retbins = True)    
                tBitacora.write("\n\n Discretizacion: "+ tNombreCol + "\nCantidad de Rangos: " + 
                    str(tCuantosRangos) + "\n"  + str(tRangosDis) + "\n")
                tMatrizDatos[tNombreCol] = copy.deepcopy(tFactorizado)
                if tNombreCol == tNombreVariableAPredecir:
                    tRangosVariableAPredecir = np.array(tRangosDis[1]) # lista de rangos utilizados
            else:
                tBitacora.write("\n\n Variable: "+ tNombreCol + " NO se discretiza" + "\n")            
    
    #########################
    # Genera el archivo descretizado        
    if gGenereArchivoDiscretizado:
        tArchDiscretizado = open(tNombrArchDiscretizado, "w")
        tNombresCol = tMatrizDatos.columns
        tTit = ""
        for tIcol in tNombresCol:
            tTit = tTit + tIcol + ";"
        tTit = tTit[:-1]   # para quitar el último ; agregado
        tArchDiscretizado.write(tTit+"\n")    
        for tNombreFil, tDatosFil in tMatrizDatos.iterrows():
            tTit = ""
            for tValor in tDatosFil:
                tTit = tTit + str(tValor)+";"
            tTit = tTit[:-1]    
            tArchDiscretizado.write(tTit + "\n")
        tArchDiscretizado.close()
      

    ###############################################################################
    # Preparando la Estructura de Datos para guardar los esultados de algoritmos
    tResultados = {}
    for tMetodo in tVariablesUtilizar:
        tResultados[tMetodo] = {}
        for tPat in tPatrones:
            tResultados[tMetodo][tPat] = {}
            for tA in tAlgoritmos:
                tResultados[tMetodo][tPat][tA] = [ [], 0.0, [], 0.0, 0.0, [], 0.0, 0.0, 0.0, 0.0, "",[],0.0,[],0.0 ]
                # 0  Lista de RMSE
                # 1  Promedio de los RMSE
                # 2  Lista de R2
                # 3  Promedio de los R2
                # 4  std de rmse
                # 5  Lista de los Accuracy
                # 6  coeficiente de variacion rmse
                # 7  std de r2
                # 8  Promedio de los Accuracy  -  
                #        the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true
                # 9  coeficiente de variacion de r2
                # 10 información adicional de parámetros
                # 11 Lista de Precision
                # 12 Promedio de los Precision
                # 13 Lista de Recall
                # 14 Promedio de los Recall             
               
    # Se reapalda la base de datos con todas la variables pues algunos método implican
    # eliminar variables
    tRespaldoDatos = tMatrizDatos.copy()
    tCuantasValidaciones = 1    
    if pTipoProceso == "Siguiente":
        # Modifica las variables para correr solo una vez y con lo requerido
        tAlgoritmos = [ pTipoAlgoritmo ]
        tVariablesUtilizar = [ pTipoMetodo ]
        tPatrones = [ pTipoPatron ]
        gNumIteraciones = 1
        tSiguienteReal =  []  # Usada para ir guardando los valores reales a usar
        if gHagaValidacion :
            tCuantasValidaciones = gCuantosPeriodosValidar 
    #tValorAnteriorPredicho = 0
    for tValidaciones in range(tCuantasValidaciones):        
        ###############################################################################
        # Reduciendo las variables que correspondan
        for tMetodo in tVariablesUtilizar:
            tBitacora.write("\n *************************************************")
            tBitacora.write("Método: "+ tMetodo+ "\n")
            if gMuestreAvance:
                print("Metodo: ", tMetodo)    
            if tMetodo == "Todas":     
                tMatrizDatos = tRespaldoDatos.copy()
                pass  # Se dejan todas las variables        
            elif tMetodo == "NecesariasConfig":
                tMatrizDatos = tRespaldoDatos.copy()
                for tNombreCol, tDatosCol in tMatrizDatos.iteritems():
                    tValorSel = tMatrizConfig.loc[tNombreCol, tVariableNecesaria]
                    if tValorSel != "s":
                        del tMatrizDatos[tNombreCol]
                tNumColumnas = len(tMatrizDatos.columns)
            elif tMetodo == "Entropia":
                tMatrizDatos = tRespaldoDatos.copy()
                for tNombreCol, tDatosCol in tMatrizDatos.iteritems():
                    tValorSel = tMatrizConfig.loc[tNombreCol, tVariableEntropia]
                    if tValorSel != "s":
                        del tMatrizDatos[tNombreCol]
                tNumColumnas = len(tMatrizDatos.columns)
            elif tMetodo == "RoughSet":        
                tMatrizDatos = tRespaldoDatos.copy()
                for tNombreCol, tDatosCol in tMatrizDatos.iteritems():
                    tValorSel = tMatrizConfig.loc[tNombreCol, tVariableRoughSet]
                    if tValorSel != "s":
                        del tMatrizDatos[tNombreCol]
                tNumColumnas = len(tMatrizDatos.columns)
            elif tMetodo[:6] == tMetodo5:  # Reducida, viene la palabra Unica
                tNumeroVar = tMetodo[6:]
                tMatrizDatos = tRespaldoDatos.copy()
                for tNombreCol, tDatosCol in tMatrizDatos.iteritems():
                    tValorSel = tMatrizConfig.loc[tNombreCol, tVariableReducida]                
                    if (tValorSel != tNumeroVar) and (tNombreCol != tNombreVariableAPredecir):
                        del tMatrizDatos[tNombreCol]
                tNumColumnas = len(tMatrizDatos.columns)
            elif tMetodo[:4] == tMetodo6:  #  Par
                tNombresVar = tMetodo.split("*")
                tNombresVar = tNombresVar[1:]   # Se elimina la primera que es la palabra "Par"  
                tMatrizDatos = tRespaldoDatos.copy()
                for tNombreCol, tDatosCol in tMatrizDatos.iteritems():
                    tValorSel = tMatrizConfig.loc[tNombreCol, tVariableReducida]  
                    if (tValorSel not in tNombresVar) and (tNombreCol != tNombreVariableAPredecir):                
                        del tMatrizDatos[tNombreCol]
                tNumColumnas = len(tMatrizDatos.columns)
            elif tMetodo[:4] == tMetodo7:  #  Combinatoria
                tNombresVar = tMetodo.split("*")            
                tNombresVar = tNombresVar[1:]
                tMatrizDatos = tRespaldoDatos.copy()
                for tNombreCol, tDatosCol in tMatrizDatos.iteritems():
                    tValorSel = tMatrizConfig.loc[tNombreCol, tVariableReducida]  
                    if (tValorSel not in tNombresVar) and (tNombreCol != tNombreVariableAPredecir):                
                        del tMatrizDatos[tNombreCol]
                tNumColumnas = len(tMatrizDatos.columns)                
            else:
                print("Error método desconocido")
                sys.exit(2)       
              
            ###############################################################################
            # Preparando estructura de datos para correr algoritmos
            tMatrizDatosEnNumpy = np.array(tMatrizDatos, dtype=np.float)   
            # Lo pasa a Numpy pues no todos los algoritmos soportan pandas. Flotante
            # Obtiene los nombres de las columnas
            tNombresColumnas = tMatrizDatos.columns.values.tolist()
          
            ##########################################################
            # Escalando los datos de las Variables para los algoritmos
            if gEscaleDatosEntreAyB:   
            # Se escala X , no se escala la variable a Predecir (y)   
                tTodosEscalados = escaleDatosT(tMatrizDatosEnNumpy, tNumColumnas , True)
                tMatrizDatosEnNumpy = escaleDatosT(tMatrizDatosEnNumpy, tNumColumnas , gEscaleY)   
                            
            ###############################################################################
            # Definiendo patrones a aplicar
            for tIpatron in tPatrones:    
                tBitacora.write("\n *****************")
                tBitacora.write("Patrón: "+ str( tIpatron)+ "\n")
                if gMuestreAvance:
                    print("Patron: ", tIpatron)                               
                tMaxColumnasDatos = tNumColumnas - gCantidadVarAPredecir
                tMatrizDatosNP = np.array([], dtype=np.float)
                tPatPeriods = int(tIpatron[0:2])  # en las dos primera posiciones esta cuantas Periods cubre un patron
                tPatAdelante = int(tIpatron[(len(tIpatron)-2):])  # en las dos ultimas posiciones esta cuantas Periods adelante predice          
                tIFila = tNumFilas - tPatPeriods - tPatAdelante
                while (tIFila >= 0):
                    tFilaNueva = np.array([])
                    tPosFil = tIFila
                    tContadorTit = 0
                    tListaNom = []
                    for tINumBloque in range(tPatPeriods):
                        tContadorTit = tContadorTit + 1
                        for tCont11 in range(len(tNombresColumnas)-1):
                            tListaNom.append(tNombresColumnas[tCont11]+str(tContadorTit))
                        if gIncluyaPredichoEnPatron:
                            tListaNom.append( tNombresColumnas[-1]+str(tContadorTit))
                            if gEscaleDatosEntreAyB:                                
                                tElemento = np.copy(tMatrizDatosEnNumpy[tPosFil])
                                #print(tElemento[-1])
                                # Este en comentario
                                tElemento[-1] = tTodosEscalados[tPosFil][-1]
                                #print(tElemento[-1])
                                ###############                                
                                tFilaNueva = np.hstack((tFilaNueva,tElemento))
                            else:
                                tElemento = np.copy(tMatrizDatosEnNumpy[tPosFil])
                                tFilaNueva = np.hstack((tFilaNueva,tElemento))
                        else:
                            tElemento = np.copy(tMatrizDatosEnNumpy[tPosFil][:-1])
                            tFilaNueva = np.hstack((tFilaNueva,tElemento))
                        tPosFil = tPosFil + 1
                    if gTrabajarComoTasaCambio:
                        if tIFila > 0:
                            tN1 = float(tMatrizDatosEnNumpy[tIFila+tPatPeriods-1,tMaxColumnasDatos:])
                            tN2 = float(tMatrizDatosEnNumpy[tIFila+tPatPeriods+tPatAdelante-1,tMaxColumnasDatos:])
                            if tN1 != 0:
                                tValorB = [ (tN2 -  tN1) / tN1 ]
                            else:
                                tValorB = [ (tN2 -  tN1) / 1 ]
                        else:
                            tValorB = [ 0.0 ]
                        #tPatronSig = list(tMatrizDatosEnNumpy[tIFila+tPatPeriods+tPatAdelante-1,tMaxColumnasDatos:])                        
                        tPatronSig = tValorB
                    else:
                        tPatronSig = list(tMatrizDatosEnNumpy[tIFila+tPatPeriods+tPatAdelante-1,tMaxColumnasDatos:])
                    tFilaNueva = np.hstack((tFilaNueva,tPatronSig))     
                    tListaNom.append(tNombresColumnas[-1])
                    if len(tMatrizDatosNP) == 0:
                        tMatrizDatosNP = np.array([ tFilaNueva ] )
                    else:
                        tMatrizDatosNP = np.vstack((tFilaNueva,tMatrizDatosNP))
                    tIFila = tIFila - 1
                    
                if gGenereArchivosDePatrones:
                    ###############################################################################
                    # Genera un archivos con todos los patrones definidos, en caso de que se solicite
                    gGuardarPatron = pd.DataFrame(tMatrizDatosNP, columns = tListaNom)                    
                    gGuardarPatron.to_excel("Patrones\\"+gNombreArchivoConPatronDefinido + "- "+tIpatron+ ".xlsx",
                                            sheet_name="Resultado", engine="openpyxl", index=False)                        
                if pTipoProceso == "Siguiente":
                    tUltimaFila = []
                    tContU = tPatPeriods
                    while tContU > 0:
                        if gIncluyaPredichoEnPatron:
                            tUltimaFila = np.hstack((tUltimaFila,tMatrizDatosEnNumpy[-tContU]))
                        else:
                            tUltimaFila = np.hstack((tUltimaFila,tMatrizDatosEnNumpy[-tContU][:-1]))                            
                        tContU -= 1
                    tUltimaFila = np.array( [ tUltimaFila ] )
                    tCuantosND = tPatAdelante  # Cuantos periodos inicialmente serán ND (No Disponible)

                ###############################################################################
                # Genera un archivo con un patron definido, en caso de que se solicite
                if gGenereArchivoConPatronDefinido:
                   gGenereArchivoConPatronDefinido = False  # Para que no lo vuelva a hacer
                   np.savetxt(gNombreArchivoConPatronDefinido + "- "+tIpatron+".txt", tMatrizDatosNP, delimiter=";")   

                 
                ###############################################################################
                # Definiendo Training set y Cross Validation set
                if pTipoProceso == "Siguiente":
                    tTotalDatos = len(tMatrizDatosNP) #  ahora es cuántos registros quedan
                    tMaxColumnasDatos = len(tMatrizDatosNP[0]) - gCantidadVarAPredecir
                    tTamCrossValidationSet = 1   #  el siguiente a predecir
                    tTamTrainingSet = tTotalDatos   # Se entrena con todos los datos disponibles
                else: 
                    tTotalDatos = len(tMatrizDatosNP) #  ahora es cuántos registros quedan
                    tMaxColumnasDatos = len(tMatrizDatosNP[0]) - gCantidadVarAPredecir
                    tTamCrossValidationSet = tTotalDatos // gNumIteraciones   
                    tTamTrainingSet = tTotalDatos - tTamCrossValidationSet
                tBitacora.write("Total de datos: "+ str( tTotalDatos)+ "\n")
                tBitacora.write("Número de Folds: "+ str( gNumIteraciones)+ "\n")
                tBitacora.write("Tamaño del Training Set: " +str( tTamTrainingSet)+ "\n")
                tBitacora.write("Tamaño del Cross Validation Set: " + str( tTamCrossValidationSet)+ "\n")
                tBitacora.write("\n" )
                       
                ###############################################################################
                # Optimiza parámetros en caso de solicitarse                            
                if gOptimiceParametros:
                    
                    if "ElasticNet" in tAlgoritmos:      
                        X_train = copy.deepcopy(tMatrizDatosNP[:, :-1])
                        y_train = copy.deepcopy(tMatrizDatosNP[:, -1: ])
                        y_train = [tCont1[0] for tCont1 in y_train]
                        tParameters = {'alpha':(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1.0), 
                                          'l1_ratio':(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1.0)}                                    
                        tSeleccione = linear_model.ElasticNet(alpha=0.5, fit_intercept=True, normalize=True, copy_X=True)
                        tFold = cross_validation.KFold(n=len(tMatrizDatosNP), n_folds=gNumIteraciones, shuffle=False, random_state=None)
                        clf = grid_search.GridSearchCV(tSeleccione, tParameters, cv =tFold, scoring='r2')
                        clf.fit( X_train, y_train ) 
                        tOPalpha1 = clf.best_params_['alpha']
                        tOPl1_ratio1 = clf.best_params_['l1_ratio']
                        print("ElasticNet")                        
                        print(tOPalpha1)
                        print(tOPl1_ratio1)
                        
                    if "newMethod2" in tAlgoritmos:
                        X_train = copy.deepcopy(tMatrizDatosNP[:, :-1])
                        y_train = copy.deepcopy(tMatrizDatosNP[:, -1: ])
                        y_train = copy.deepcopy([tCont1[0] for tCont1 in y_train])
                        tParameters = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                        tEntrenadoNewMethod = newMethod_2_entrene(X_train, y_train)   
                        tOLimiteNM2 = newMethod_2_SeleccioneParam(tEntrenadoNewMethod, tParameters,X_train, y_train )
                        print("NewMethod2")
                        print(tOLimiteNM2)
                        
                    if "ESN" in tAlgoritmos:                        
                        pOptDatos = copy.deepcopy(tMatrizDatosNP)
                        pListaNeuronas = [  int( round( tTamTrainingSet * 0.1 )),
                                            int( round( tTamTrainingSet * 0.2 )),
                                            int( round( tTamTrainingSet * 0.3 )),
                                            int( round( tTamTrainingSet * 0.4 )),
                                            int( round( tTamTrainingSet * 0.5 )),
                                            int( round( tTamTrainingSet * 0.6 )),
                                            int( round( tTamTrainingSet * 0.7 )),
                                            int( round( tTamTrainingSet * 0.8 )),
                                            int( round( tTamTrainingSet * 0.9 )) ]
                        pListaLeakingRate = [ 0.02, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.5,0.9]
                        pListaInitLen = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ]
                        tParameters = {'neuronas': pListaNeuronas, 'leakingrate': pListaLeakingRate,
                                       'InitLen': pListaInitLen}
                        tOptPar = ESN_Optimice_Params(tParameters,pOptDatos, tTamTrainingSet, gCantidadVarAPredecir )                        
                        tOptNeuronas3 = tOptPar['neuronas']
                        tOptLeakingRate3 = tOptPar['leakingrate']
                        tOptInitLenPorcentaje3 = tOptPar['InitLen']
                        print("ESN")
                        print(tOptNeuronas3)
                        print(tOptLeakingRate3)
                        print(tOptInitLenPorcentaje3)

                    if "SVR model con Kernel linear" in tAlgoritmos:   
                        X_train = copy.deepcopy(tMatrizDatosNP[:, :-1])
                        y_train = copy.deepcopy(tMatrizDatosNP[:, -1: ])
                        y_train = [tCont1[0] for tCont1 in y_train]
                        tParameters = {'C':[0.001, 0.1, 0.5, 1,5, 10, 20, 30, 50, 65,  80, 100, 1000, 10000, 100000], 
                                       'epsilon':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9]}
                        tSeleccione = SVR( kernel='linear')                        
                        tFold = cross_validation.KFold(n=len(tMatrizDatosNP), n_folds=gNumIteraciones, shuffle=False, random_state=None)
                        clf = grid_search.GridSearchCV(tSeleccione, tParameters, cv =tFold, scoring='r2')
                        clf.fit( X_train, y_train ) 
                        tOpttSVR_C4 = clf.best_params_['C']
                        tOpttEpsilon4 = clf.best_params_['epsilon']     
                        print("SVR linear")
                        print(tOpttSVR_C4)
                        print(tOpttEpsilon4)           
                        
                    if "SVR model con Kernel rbf" in tAlgoritmos:   
                        X_train = copy.deepcopy(tMatrizDatosNP[:, :-1])
                        y_train = copy.deepcopy(tMatrizDatosNP[:, -1: ])
                        y_train = [tCont1[0] for tCont1 in y_train]
                        tParameters = {'C':[0.001, 0.1, 0.5, 1,5, 10, 20, 30, 50, 65,  80, 100, 1000, 10000, 100000], 
                                       'epsilon':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9], 
                                        'gamma' :[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1.0]}
                        tSeleccione = SVR( kernel='rbf')                        
                        tFold = cross_validation.KFold(n=len(tMatrizDatosNP), n_folds=gNumIteraciones, shuffle=False, random_state=None)
                        clf = grid_search.GridSearchCV(tSeleccione, tParameters, cv =tFold, scoring='r2')
                        clf.fit( X_train, y_train ) 
                        tOpttSVR_C5 = clf.best_params_['C']
                        tOpttEpsilon5 = clf.best_params_['epsilon']                                     
                        tOpttSVR_gamma5 = clf.best_params_['gamma']     
                        print("SVR rbf")                                
                        print(tOpttSVR_C5)
                        print(tOpttEpsilon5)
                        print(tOpttSVR_gamma5)

                    if "SVR model con Kernel sigmoid" in tAlgoritmos:   
                        X_train = copy.deepcopy(tMatrizDatosNP[:, :-1])
                        y_train = copy.deepcopy(tMatrizDatosNP[:, -1: ])
                        y_train = [tCont1[0] for tCont1 in y_train]
                        tParameters = {'C':[0.001, 0.1, 0.5, 1,5, 10, 20, 30, 50, 65,  80, 100, 1000, 10000, 100000], 
                                       'epsilon':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9], 
                                        'coef0' : [0.0, 0.1, 0.2, 0.5, 0.7, 0.9, 10], 
                                        'gamma' :[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1.0]}
                        tSeleccione = SVR( kernel='sigmoid')                        
                        tFold = cross_validation.KFold(n=len(tMatrizDatosNP), n_folds=gNumIteraciones, shuffle=False, random_state=None)
                        clf = grid_search.GridSearchCV(tSeleccione, tParameters, cv =tFold, scoring='r2')
                        clf.fit( X_train, y_train ) 
                        tOpttSVR_C6 = clf.best_params_['C']
                        tOpttEpsilon6 = clf.best_params_['epsilon']   
                        tOpttCoef06 = clf.best_params_['coef0']                                     
                        tOpttSVR_gamma6 = clf.best_params_['gamma']    
                        print("SVR sigmoid")
                        print(tOpttSVR_C6)
                        print(tOpttEpsilon6)
                        print(tOpttCoef06)
                        print(tOpttSVR_gamma6)

                    if "SVR model con Kernel poly" in tAlgoritmos:   
                        X_train = copy.deepcopy(tMatrizDatosNP[:, :-1])
                        y_train = copy.deepcopy(tMatrizDatosNP[:, -1: ])
                        y_train = [tCont1[0] for tCont1 in y_train]
                        tParameters = {'C':[0.001, 0.1, 0.5, 1,5, 10, 20, 30, 50, 65,  80, 100, 1000, 10000, 100000], 
                                       'epsilon':[0.0, 0.1, 0.2, 0.3, 0.3, 0.5, 0.6, 0.7, 0.8,0.9], 
                                        'coef0' : [0.0, 0.1, 0.2, 0.5, 0.7, 0.9, 10], 
                                        'gamma' :[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1.0], 
                                       'degree':(1,2,3)}
                        tSeleccione = SVR( kernel='poly')                        
                        tFold = cross_validation.KFold(n=len(tMatrizDatosNP), n_folds=gNumIteraciones, shuffle=False, random_state=None)
                        clf = grid_search.GridSearchCV(tSeleccione, tParameters, cv =tFold, scoring='r2')
                        clf.fit( X_train, y_train ) 
                        tOpttSVR_C7 = clf.best_params_['C']
                        tOptt_degree7 = clf.best_params_['degree']
                        tOpttEpsilon7 = clf.best_params_['epsilon']                                     
                        tOpttSVR_gamma7 = clf.best_params_['gamma']   
                        tOpttCoef07 = clf.best_params_['coef0'] 
                        print("SVR poly")
                        print(tOpttSVR_C7)
                        print(tOptt_degree7)
                        print(tOpttEpsilon7)
                        print(tOpttSVR_gamma7)
                        print(tOpttCoef07)

                    if "Ridge" in tAlgoritmos:      
                        X_train = copy.deepcopy(tMatrizDatosNP[:, :-1])
                        y_train = copy.deepcopy(tMatrizDatosNP[:, -1: ])
                        y_train = [tCont1[0] for tCont1 in y_train]
                        tParameters = {'alpha':(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9, 1.0) }                                    
                        tSeleccione = linear_model.Ridge(fit_intercept=True, normalize=True, copy_X=True)
                        tFold = cross_validation.KFold(n=len(tMatrizDatosNP), n_folds=gNumIteraciones, shuffle=False, random_state=None)
                        clf = grid_search.GridSearchCV(tSeleccione, tParameters, cv =tFold, scoring='r2')
                        clf.fit( X_train, y_train ) 
                        tOPalpha8 = clf.best_params_['alpha']
                        print("Ridge")                        
                        print(tOPalpha8)

                else: # No optimiza parámetros
                    # Elastic Net
                    tOPalpha1 = 0.5
                    tOPl1_ratio1 = 0.5                    
                    
                    # New Method 2
                    tOLimiteNM2 = pLimiteNM                             
                    
                    # ESN
                    tOptNeuronas3 = int( round( tTamTrainingSet * 0.4 ))
                    tOptLeakingRate3 = pLeakingRate
                    tOptInitLenPorcentaje3 = gInitLenPorcentaje
                    
                    #SVR-linear
                    tOpttSVR_C4 = tSVR_C
                    tOpttEpsilon4 = 0.1
                    
                    #SVR-rbv                    
                    tOpttSVR_C5 = tSVR_C
                    tOpttEpsilon5 = 0.1
                    tOpttSVR_gamma5 = tSVR_gamma
                    
                    #SVR-sigmoid
                    tOpttSVR_C6 = tSVR_C
                    tOpttEpsilon6 = 0.1
                    tOpttSVR_gamma6 = tSVR_gamma                    
                    tOpttCoef06 = 0.0

                    #SVR-poly
                    tOpttSVR_C7 = tSVR_C
                    tOpttEpsilon7 = 0.1
                    tOpttSVR_gamma7 = tSVR_gamma                    
                    tOpttCoef07 = 0.0    
                    tOptt_degree7 = t_degree
                    
                    # Ridge
                    tOPalpha8 = 0.1
                    
                    
                ###############################################################################
                # Aplicando algoritmos                        
                for tVez in range(gNumIteraciones):
                    if gMuestreAvance:
                        print("Iteracion: ", tVez)
                    tBitacora.write("Iteración: "+ str( tVez)+ "\n")
                    
                    ###############################################################################        
                    # Preparación del Training set y del Cross Validation set
                    if pTipoProceso == "Siguiente":
                        tCrossValidationSet = copy.deepcopy(tUltimaFila)
                        tTrainingSet = tMatrizDatosNP.copy()
                        pTrainESN = tTrainingSet.copy()
                        pCrossESN = tCrossValidationSet.copy()
                        tRespuestasTrainSet = copy.deepcopy(tTrainingSet[:,tMaxColumnasDatos:])
                        tRespuestasTrainSet = np.ndarray.flatten(tRespuestasTrainSet)   
                        tTrainingSet = copy.deepcopy(tTrainingSet[ : , : tMaxColumnasDatos])
                    else:
                        tParteA = copy.deepcopy(tMatrizDatosNP[0: (tTamCrossValidationSet*tVez)])
                        tCrossValidationSet = copy.deepcopy(tMatrizDatosNP[ (tTamCrossValidationSet*tVez) : ((tTamCrossValidationSet*tVez)+tTamCrossValidationSet)])
                        tParteB = copy.deepcopy(tMatrizDatosNP[(tTamCrossValidationSet*tVez)+tTamCrossValidationSet:])
                        tTrainingSet = np.vstack((tParteA,tParteB))  
                        pTrainESN = tTrainingSet.copy()
                        tRespuestasTrainSet = copy.deepcopy(tTrainingSet[:,tMaxColumnasDatos:])
                        tRespuestasTrainSet = np.ndarray.flatten(tRespuestasTrainSet)
                        tRespuestasCrossSet = copy.deepcopy(tCrossValidationSet[:,tMaxColumnasDatos :])
                        tRespuestasCrossSet = np.ndarray.flatten(tRespuestasCrossSet)    
                        tTrainingSet = copy.deepcopy(tTrainingSet[ : , : tMaxColumnasDatos])
                        tCrossValidationSet = copy.deepcopy(tCrossValidationSet[ : , : tMaxColumnasDatos])
                        pCrossESN = tCrossValidationSet.copy()                    
                    
                    
                    for tAlgoritmo in tAlgoritmos:   
                        #####################################################
                        # Oara poder comparar tiempos de corrida entre algoritmos                   
                        if gMuestreAvance:
                            print("Ini: ",time.strftime("%a, %d %b %Y %H:%M:%S"))
                            #####################################################
                        if gMuestreAvance:
                            print("Calculando: ", tAlgoritmo," - ", tIpatron," - ", tMetodo)
                        #print("el training: ", tTrainingSet[0])
                        ###############################################################################        
                        if tAlgoritmo == "LDA":   
                            tModelo = LDA(n_components=None, priors=None) 
                            tModelo.fit(tTrainingSet, tRespuestasTrainSet)
                            tResultado = tModelo.predict(tCrossValidationSet)
                            tDetalleAlg = str(tModelo.get_params(deep=True))+" - "+str(tModelo.coef_)
                            
                        ###############################################################################        
                        elif tAlgoritmo == "LinearRegression":   
                            tModelo = linear_model.LinearRegression(fit_intercept=True, normalize=True, copy_X=True) 
                            tModelo.fit(tTrainingSet, tRespuestasTrainSet)
                            tResultado = tModelo.predict(tCrossValidationSet)
                            tDetalleAlg = str(tModelo.get_params(deep=True))+" - "+str(tModelo.coef_)
                            
                        ###############################################################################        
                        elif tAlgoritmo == "ElasticNet":   
                            tModelo = linear_model.ElasticNet(alpha=tOPalpha1, l1_ratio=tOPl1_ratio1, 
                                            fit_intercept=True, normalize=True, copy_X=True)
                            tModelo.fit(tTrainingSet, tRespuestasTrainSet)
                            tResultado = tModelo.predict(tCrossValidationSet)
                            tDetalleAlg = str(tModelo.get_params(deep=True))+" - "+str(tModelo.coef_)
                                    
                        ###############################################################################        
                        elif tAlgoritmo == "Ridge":   
                            tModelo = Ridge(alpha=tOPalpha8, fit_intercept=True, normalize=True, copy_X=True)
                            tModelo.fit(tTrainingSet, tRespuestasTrainSet)
                            tResultado = tModelo.predict(tCrossValidationSet)
                            tDetalleAlg = str(tModelo.get_params(deep=True))+" - "+str(tModelo.coef_)
                                    
                        ###############################################################################        
                        elif tAlgoritmo == "SVR model con Kernel rbf":   
                            tAjusteSVR_rbf = SVR(kernel='rbf', C=tOpttSVR_C5, gamma=tOpttSVR_gamma5,
                                                 epsilon = tOpttEpsilon5)    
                            tAjusteSVR_rbf.fit(tTrainingSet, tRespuestasTrainSet)
                            tResultado = tAjusteSVR_rbf.predict(tCrossValidationSet)
                            tDetalleAlg = str(tAjusteSVR_rbf.get_params(deep=True))+" "+str(tAjusteSVR_rbf.support_vectors_)
        
                        ###############################################################################
                        elif tAlgoritmo == "SVR model con Kernel linear":
                            #tAjusteSVR_lin = SVR(kernel='linear')
                            tAjusteSVR_lin = SVR(kernel='linear', C=tOpttSVR_C4, 
                                                 tol = 1e-3 , epsilon = tOpttEpsilon4, shrinking = True)
                            tAjusteSVR_lin.fit(tTrainingSet, tRespuestasTrainSet)
                            tResultado = tAjusteSVR_lin.predict(tCrossValidationSet)
                            tDetalleAlg = str(tAjusteSVR_lin.get_params(deep=True))+" "+str(tAjusteSVR_lin.support_vectors_)                                

                        ###############################################################################
                        elif tAlgoritmo == "SVR model con Kernel sigmoid":          
                            tAjusteSVR_sig = SVR(kernel='sigmoid', C=tOpttSVR_C6, gamma=tOpttSVR_gamma6, 
                                                 coef0 = tOpttCoef06, epsilon = tOpttEpsilon6) 
                            tAjusteSVR_sig.fit(tTrainingSet, tRespuestasTrainSet)
                            tResultado = tAjusteSVR_sig.predict(tCrossValidationSet) 
                            tDetalleAlg = str(tAjusteSVR_sig.get_params(deep=True))+" "+str(tAjusteSVR_sig.support_vectors_)                 

                        ###############################################################################
                        elif tAlgoritmo == "SVR model con Kernel poly":          
                            tAjusteSVR_sig = SVR(kernel='poly', C=tOpttSVR_C7, gamma=tOpttSVR_gamma7, 
                                                 coef0 = tOpttCoef07, epsilon = tOpttEpsilon7, degree=tOptt_degree7) 
                            tAjusteSVR_sig.fit(tTrainingSet, tRespuestasTrainSet)
                            tResultado = tAjusteSVR_sig.predict(tCrossValidationSet) 
                            tDetalleAlg = str(tAjusteSVR_sig.get_params(deep=True))+" "+str(tAjusteSVR_sig.support_vectors_)                 
       
                        ###############################################################################
                        elif tAlgoritmo == "newMethod":       
                            tEntrenadoNewMethod = newMethod_entrene(tTrainingSet, tRespuestasTrainSet)
                            tResultado = newMethod_prediga(tEntrenadoNewMethod, tCrossValidationSet, pLimiteNM)                         
                            tDetalleAlg = "Limite: "  + str(pLimiteNM)     
    
                        ###############################################################################
                        elif tAlgoritmo == "newMethod2":          
                            tEntrenadoNewMethod = newMethod_2_entrene(tTrainingSet, tRespuestasTrainSet)
                            tResultado = newMethod_2_prediga(tEntrenadoNewMethod, tCrossValidationSet,tOLimiteNM2)                         
                            tDetalleAlg = "Limite: "  + str(tOLimiteNM2)                             
    
                        ###############################################################################
                        elif tAlgoritmo == "newMethod3":
                            tEntrenadoNewMethod = newMethod_3_entrene(tTrainingSet, tRespuestasTrainSet)
                            tResultado = newMethod_3_prediga(tEntrenadoNewMethod, tCrossValidationSet)                         
                            tDetalleAlg = "Regresión (Xt * X)'-1 * Xt * y: "  + str(tEntrenadoNewMethod)                             
    
                        ###############################################################################
                        elif tAlgoritmo == "gradientDescent":
                            tAlpha = 0.001;
                            tNum_iters = 3000;
                            X = np.hstack((np.ones( (len(tTrainingSet),1)),tTrainingSet))
                            tTheta = np.zeros((len(X[0]), 1))
                            tTemp = copy.deepcopy(tRespuestasTrainSet)
                            y = tTemp.reshape(len(X), 1)
                            tEntrenadoNewMethod = GD_entrene(X, y, tTheta, tAlpha, tNum_iters)
                            XTest = np.hstack((np.ones(( len(tCrossValidationSet),1)),tCrossValidationSet))
                            tResultado = GD_prediga(tEntrenadoNewMethod, XTest)                         
                            tDetalleAlg = "Regresión con gradient descent. Theta: "  + str(tEntrenadoNewMethod[1])+str(" Alpha- ")+str(tAlpha)+str(" NumIteraciones- ")+str(tNum_iters)
                            tResultado = tResultado.transpose()
                            tResultado = tResultado[0]                        

                        ###############################################################################        
                        elif tAlgoritmo == "BayesianRidge":   
                            tModelo = linear_model.BayesianRidge(normalize=False, copy_X=True)
                            tModelo.fit(tTrainingSet, tRespuestasTrainSet)
                            tResultado = tModelo.predict(tCrossValidationSet)
                            tDetalleAlg = str(tModelo.get_params(deep=True))+" - "+str(tModelo.coef_)
                            
                        ###############################################################################
                        elif tAlgoritmo == "DynamicTimeWarping":   
                            tResultado = CalculeDTW(tTrainingSet, tRespuestasTrainSet, tCrossValidationSet)
                            tDetalleAlg = "euclidean distance" 
                            
                        elif tAlgoritmo == "Kalman_Filter":   
                            tResultado = Calcule_Kalman_Filter(tTrainingSet, tRespuestasTrainSet, tCrossValidationSet)
                            tDetalleAlg = "all" 
                            
                        ###############################################################################
                        elif tAlgoritmo == "ESN":    
                            tBitacora.write("ESN"+ "\n")
                            tResultado = CalculeConESN(pTrainESN, pCrossESN, gCantidadVarAPredecir,
                                tOptNeuronas3, tOptLeakingRate3, tOptInitLenPorcentaje3)
                            tDetalleAlg =  "Neuronas= "+str(tOptNeuronas3)+" - Leaking rate="+str(tOptLeakingRate3)+" - Init Porcentaje="+str(tOptInitLenPorcentaje3)
                        else:
                            print("Algoritmo Desconocido: ",tAlgoritmo)
                            sys.exit(5)
                        # Luego de ejecutar el algoritmo
                            
                        #####################################################
                        # Oara poder comparar tiempos de corrida entre algoritmos                   
                        if gMuestreAvance:
                            print("Fin: ",time.strftime("%a, %d %b %Y %H:%M:%S"))
                        #####################################################
                                    
                        if pTipoProceso == "Siguiente":
                            tBitacora.write(tAlgoritmo+ "\n")          
                            tResultado = [ tResultado ]               
                            if gEscaleY and gEscaleDatosEntreAyB:
                                tResultadoUnico = np.array(desescaleT(tResultado)).transpose()
                            else:
                                tResultadoUnico = copy.deepcopy(tResultado[:])
                            if tDecisionDiscretizar[tNombreCol] == "s": 
                                if gCantidadRangosVarAPredecir == len(gEscalaTres):                    
                                    tMensajeResult = gEscalaTres[int(tResultadoUnico[0])]
                                elif gCantidadRangosVarAPredecir == len(gEscalaCinco):
                                    tMensajeResult = gEscalaCinco[int(tResultadoUnico[0])]     
                                else:
                                    tMensajeResult = "Escala no disponible"   
                            else:
                                tMensajeResult = "Valor no discretizado"
                            print("*************************************")                            
                            print(tLugar)
                            print("*************************************")  
                            print("Validacion numero: "+str(tValidaciones))
                            print("Archivo entrada: ",tArchivoEntrada)                                                  
                            print("El siguiente valor es: ", tResultadoUnico, " - ", tMensajeResult)
                            tBitacora.write("\nEl siguiente valor es: "+ str(tResultadoUnico) +"\n")
                            tBitacora.write("significado: "+ tMensajeResult +"\n")    
                            
                            gResultadosSiguiente.append(tResultadoUnico[0][0])
                            gListaValoresReales.append("ND")
                            if gHagaValidacion:
                                tSiguienteReal.append(tRespuestasTrainSet[-1])
                                if (tCuantosND - tValidaciones) > 0: # tValidaciones inicia en cero
                                    tValorRealT = "ND"
                                else:
                                    tValorRealT = tSiguienteReal[0]
                                    tSiguienteReal = tSiguienteReal[1:]                                    
                                tTempDic ={ "Discretizacion":[gTipoDiscretizacion], "EscalaX":[gEscaleDatosEntreAyB], 
                                              "EscalaY":[gEscaleY], "MetodoEscalar":[gMetodoParaEscalar], "Lugar":[tLugar],
                                              "Metodo": [tMetodo], "APatron": [tIpatron], "Algoritmo":[tAlgoritmo], 
                                              "Detalles": [tDetalleAlg], "Num_Train": [len(tTrainingSet)],
                                              "Num_Validacion": [tCuantasValidaciones-tValidaciones],  
                                              "Valor1Real":[tValorRealT],
                                              "Valor2Predicho":[tResultadoUnico[0][0]]}
                                tUnRV = pd.DataFrame(tTempDic)
                                gResultadoValidacion = gResultadoValidacion.append(tUnRV)
                            print("*************************************")
                        else:                            
                            tBitacora.write(tAlgoritmo+ "\n") 
                            if gEscaleY and gEscaleDatosEntreAyB:
                                tOrigRespuestas = np.array(desescaleT(tRespuestasCrossSet)).transpose()
                                tOrigResultado = np.array(desescaleT(tResultado)).transpose()                                                        
                            else: 
                                tOrigRespuestas = copy.deepcopy(tRespuestasCrossSet[:])
                                tOrigResultado = copy.deepcopy(tResultado[:])          
                            tRMSE = deme_rmer(tOrigRespuestas, tOrigResultado)
                            tR2 = calculeR2(tOrigRespuestas, tOrigResultado)
                            tResultados[tMetodo][tIpatron][tAlgoritmo][0].append(tRMSE)
                            tResultados[tMetodo][tIpatron][tAlgoritmo][2].append(tR2)                             
                            
                            
                            #tAccuracy = calcule_Accuracy(tOrigRespuestas, tOrigResultado)
                            tAccuracy = 0
                            
                            
                            #tMedidas = calcule_precision_recall(tOrigRespuestas, tOrigResultado)
                            tMedidas= [1,2]
                            
                            
                            
                            tPrecision = tMedidas[0]
                            tRecall = tMedidas[1]
                            tResultados[tMetodo][tIpatron][tAlgoritmo][5].append(tAccuracy)  
                            tResultados[tMetodo][tIpatron][tAlgoritmo][11].append(tPrecision)  
                            tResultados[tMetodo][tIpatron][tAlgoritmo][13].append(tRecall)  
                            tDetalleAlg = tDetalleAlg.replace(";"," ")
                            tDetalleAlg = tDetalleAlg.replace("\n"," ")                         
                            tResultados[tMetodo][tIpatron][tAlgoritmo][10] = copy.deepcopy(tDetalleAlg)                              
                if pTipoProceso != "Siguiente":
                    ###############################################################################
                    # Calcula promedio de RMSE
                    for a in tAlgoritmos:
                        total = 0.0
                        for c in range(gNumIteraciones):
                            total = total + tResultados[tMetodo][tIpatron][a][0][c]
                        tResultados[tMetodo][tIpatron][a][1] = total / gNumIteraciones
                        if len(tResultados[tMetodo][tIpatron][a][0]) != gNumIteraciones:
                            print("Error en la lista de RMSE")
                            sys.exit(3)
                                      
                    ###############################################################################
                    # Calcula el resto
                    for a in tAlgoritmos:                
                        total = 0.0
                        for c in range(gNumIteraciones):
                            total = total + tResultados[tMetodo][tIpatron][a][2][c]
                        if len(tResultados[tMetodo][tIpatron][a][2]) != gNumIteraciones:
                            print("Error en la lista de R2")
                            sys.exit(4)
                        tResultados[tMetodo][tIpatron][a][3] = total / gNumIteraciones
                        tResultados[tMetodo][tIpatron][a][4] = np.std(tResultados[tMetodo][tIpatron][a][0]) 
                        tResultados[tMetodo][tIpatron][a][6] = tResultados[tMetodo][tIpatron][a][4] / tResultados[tMetodo][tIpatron][a][1] 
                        tResultados[tMetodo][tIpatron][a][7] = np.std(tResultados[tMetodo][tIpatron][a][2])
                        tResultados[tMetodo][tIpatron][a][8] = np.mean(tResultados[tMetodo][tIpatron][a][5])  
                        tResultados[tMetodo][tIpatron][a][9] = tResultados[tMetodo][tIpatron][a][7] / tResultados[tMetodo][tIpatron][a][3] 
                        tResultados[tMetodo][tIpatron][a][12] = np.mean(tResultados[tMetodo][tIpatron][a][11] )
                        tResultados[tMetodo][tIpatron][a][14] = np.mean(tResultados[tMetodo][tIpatron][a][13] )
                        
                        gSVRoutfile.write(tMetodo+";"+str(tIpatron)+";"+a+";" +
                                tResultados[tMetodo][tIpatron][a][10]+";"+ str(len(tTrainingSet))+ ";"+
                                str(len(tCrossValidationSet)) +";"+ str(len(tTrainingSet[0])+gCantidadVarAPredecir)+";"+
                                str(tResultados[tMetodo][tIpatron][a][1])+ ";"+
                                str(tResultados[tMetodo][tIpatron][a][3]) + ";" +
                                str(tResultados[tMetodo][tIpatron][a][4]) + ";" + 
                                str(tResultados[tMetodo][tIpatron][a][6]) + ";" + 
                                str(tResultados[tMetodo][tIpatron][a][7]) + ";" + 
                                str(tResultados[tMetodo][tIpatron][a][8]) + ";" + 
                                str(tResultados[tMetodo][tIpatron][a][9]) + ";" +
                                str(tResultados[tMetodo][tIpatron][a][12]) + ";" +
                                str(tResultados[tMetodo][tIpatron][a][14]) + "\n")      
                        gGeneral = gGeneral.append( { "Discretizacion":gTipoDiscretizacion, "EscalaX":gEscaleDatosEntreAyB, "EscalaY":gEscaleY, 
                                "MetodoEscalar":gMetodoParaEscalar,                                             
                                "Lugar":tLugar,"Metodo": tMetodo, "Patron": tIpatron, 
                                "Algoritmo":a, "Detalles": tResultados[tMetodo][tIpatron][a][10], 
                                "Train": len(tTrainingSet), "Test": len(tCrossValidationSet), 
                                "Num_Columnas": (len(tTrainingSet[0])+gCantidadVarAPredecir), 
                                "RMSE":tResultados[tMetodo][tIpatron][a][1], 
                                "R2": tResultados[tMetodo][tIpatron][a][3],  
                                "std-RMSE": tResultados[tMetodo][tIpatron][a][4], 
                                "CV-RMSE":tResultados[tMetodo][tIpatron][a][6], 
                                "std-R2": tResultados[tMetodo][tIpatron][a][7], 
                                "Accuracy":tResultados[tMetodo][tIpatron][a][8], 
                                "CV-R2":tResultados[tMetodo][tIpatron][a][9] ,
                                "Precision":tResultados[tMetodo][tIpatron][a][12] ,
                                "Recall":tResultados[tMetodo][tIpatron][a][14]},  ignore_index=True )                
                        if gMuestreAvance:
                            print(tMetodo,a,tResultados[tMetodo][tIpatron][a][3])
                                   
                ###############################################################################
                # Graficando 
                if tGenereGraficos: 
                    tMatrizDatos3 = np.arange(gNumIteraciones)        
                    pl.grid(color='r', linestyle='-', linewidth=2)
                    pl.plot(tMatrizDatos3, tResultados[tMetodo][tIpatron][tAlgoritmo1][0], c='g', label='SVR rbf model')
                    pl.plot(tMatrizDatos3, tResultados[tMetodo][tIpatron][tAlgoritmo2][0], c='b', label='SVR linear model')
                    pl.plot(tMatrizDatos3, tResultados[tMetodo][tIpatron][tAlgoritmo3][0], c='y', label='SVR sigmoid model')
                    pl.xlabel('Iteracion')
                    pl.ylabel('RMSE')
                    pl.title("Resultados de RMSE con datos de " + tLugar+ "\n" +"Training Set: "+str(tTamTrainingSet)+
                            "\n"+"Total de datos: "+str(len(tMatrizDatosNP)) + "\n"+"Número de Folds: "
                            +str(gNumIteraciones) + "\n"+"Patron de Periods: " + str(tIpatron))
                    pl.legend(loc=2)
            
                    pl.savefig("Figuras\Figura - "+tMetodo+"- Patron "+str(tIpatron)+"-"+ str(tNumFigura) + ".png")
                    tNumFigura = tNumFigura + 1
                    pl.show()
        ###############################################################################
        if (pTipoProceso == "Siguiente") and gHagaValidacion :
            tBorrar = (tRespaldoDatos.index[-1])
            tRespaldoDatos = tRespaldoDatos.drop(tBorrar)
            tNumFilas = len(tRespaldoDatos.index)
    # fin de: for tValidaciones in range(tCuantasValidaciones):   
    
    if (pTipoProceso == "Siguiente") and gHagaValidacion :    
        gResultadosSiguiente.reverse()
        gListaValoresReales.reverse()

        
    if pTipoProceso != "Siguiente":
        tBitacora.write("********************************" + "\n" )    
        tBitacora.write("********************************" + "\n" )    
        tElMejorR2 = -1000
        tElMejorRMSE =  1000
        tElMejorR2 = 0
        tMejorMetR2 = ""
        tMejorPatR2 = ""
        tMejorAlgR2 = ""
        tElMejorRMSE = 0
        tMejorMetRMSE = ""
        tMejorPatRMSE = ""
        tMejorAlgRMSE = "" 
        for tMetodo in tVariablesUtilizar:
            tBitacora.write("\n Método: " + tMetodo + "\n" )
            for tPat in tPatrones:
                tBitacora.write("Patrón: " + tPat + "\n" )
                for tA in tAlgoritmos:
                    tBitacora.write("Algoritmo: " + tA + "\n" )
                    tBitacora.write("Detalles: " + str(tResultados[tMetodo][tPat][tA][10]) + "\n" )
                    tBitacora.write("Lista de RMSE: " + str(tResultados[tMetodo][tPat][tA][0]) + "\n" )
                    tBitacora.write("Lista de R2: " + str(tResultados[tMetodo][tPat][tA][2]) + "\n" )             
                    tBitacora.write("Promedio RMSE: " + str(tResultados[tMetodo][tPat][tA][1]) + "\n" )
                    tBitacora.write("Promedio R2: " + str(tResultados[tMetodo][tPat][tA][3]) + "\n" )
                    tBitacora.write("std de RMSE: " + str(tResultados[tMetodo][tPat][tA][4]) + "\n" )
                    tBitacora.write("Coeficiente de variacion de RMSE: " + str(tResultados[tMetodo][tPat][tA][6]) + "\n" )
                    tBitacora.write("std de R2: " + str(tResultados[tMetodo][tPat][tA][7]) + "\n" )
                    tBitacora.write("Accuracy: " + str(tResultados[tMetodo][tPat][tA][8]) + "\n" )
                    tBitacora.write("Coeficiente de variacion de R2: " + str(tResultados[tMetodo][tPat][tA][9]) + "\n" )
                    tBitacora.write("Precision: " + str(tResultados[tMetodo][tPat][tA][12]) + "\n" )
                    tBitacora.write("Recall: " + str(tResultados[tMetodo][tPat][tA][14]) + "\n" )                
                    if tElMejorR2 < tResultados[tMetodo][tPat][tA][3]:
                        tElMejorR2 = tResultados[tMetodo][tPat][tA][3]
                        tMejorMetR2 = tMetodo
                        tMejorPatR2 = tPat
                        tMejorAlgR2 = tA
                    if tElMejorRMSE > tResultados[tMetodo][tPat][tA][1]:
                        tElMejorRMSE = tResultados[tMetodo][tPat][tA][1]
                        tMejorMetRMSE = tMetodo
                        tMejorPatRMSE = tPat
                        tMejorAlgRMSE = tA    
        tBitacora.write("\n *****   Parámetros del modelo SVR   **** \n")
        tBitacora.write("gamma: "+ str(tSVR_gamma)+ "\n")
        tBitacora.write("C: "+ str(tSVR_C)+ "\n")
        tBitacora.write("************** El mejor R2 ******************" + "\n" )   
        tBitacora.write( "Método: "+ tMejorMetR2 +"\n" )    
        tBitacora.write( "Patrón: "+ tMejorPatR2 + "\n" )
        tBitacora.write( "Algoritmo: " + tMejorAlgR2+ "\n" )
        tBitacora.write( "R2: "+ str(tElMejorR2) + "\n" )            
        tBitacora.write("************** El mejor RMSE ******************" + "\n" )   
        tBitacora.write( "Método: "+ tMejorMetRMSE +"\n" )    
        tBitacora.write( "Patrón: "+ tMejorPatRMSE + "\n" )
        tBitacora.write( "Algoritmo: " + tMejorAlgRMSE + "\n" )
        tBitacora.write( "RMSE: "+ str(tElMejorRMSE) + "\n" )           
       
    ###############################################################################
    # Cerrando el programa
    tBitacora.write<("\n \n  FIN DEL PROGRAMA \n")
    tBitacora.write(str(time.strftime("%a, %d %b %Y %H:%M:%S")) + "\n\n" ) 
    gSVRoutfile.close()
    tBitacora.close()
#########################################################################
#########################################################################


#########################################################################    
def pareto_frontier(Xs, Ys, maxX = True, maxY = True):
    # Calcula el frente de pareto
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    p_front = [myList[0]] 
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: 
                p_front.append(pair) 
        else:
            if pair[1] <= p_front[-1][1]:
                p_front.append(pair) 
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    return p_frontX, p_frontY

#########################################################################    
def grafique_pareto_frontier(pNombre, pXlabel, pYLabel, pLoc, Xs, Ys, maxX = True, maxY = True):
    # Calcula el frente de pareto
    pl.clf()
    p_front = pareto_frontier(Xs, Ys, maxX = False, maxY = True)
    pl.scatter(Xs, Ys)
    pl.plot(p_front[0], p_front[1])
    pl.xlabel(pXlabel)
    pl.ylabel(pYLabel)
    pl.legend(loc=pLoc)
    pl.savefig(pNombre+".png")
    pl.show()

#########################################################################
def genereUnGrafico(pDatos, pColores, pTipos, pCol, pTipoGraf, pX, pY, pLoc, pNombre, pPareto):
    # Imprime y archiva un gráfico
    tColor = 0
    if len(pDatos>0):
        try:
            for tL in pTipos:
                tParte = pDatos[(pDatos[pCol] == tL )]
                if len(tParte) > 0:     
                    if tColor == 0:
                        t_ax = tParte.plot(kind=pTipoGraf, x=pX, y=pY, color=pColores[tColor], label=tL)
                        tColor += 1
                    else:
                        tParte.plot(kind=pTipoGraf, x=pX, y=pY, color=pColores[tColor], label=tL, ax=t_ax)
                        tColor += 1
                if tColor >= len(pColores):
                    tColor = 0
            pl.legend(loc=pLoc)
            t_ax.plot()            
            pl.savefig(pNombre+".png")
            if pPareto:
                pXequis = list(pDatos[pX])
                pYes = list(pDatos[pY])
                pNombrePar = pNombre + " - FRONTERA DE PARETO" 
                grafique_pareto_frontier(pNombrePar, pX, pY, pLoc, pXequis, pYes, maxX = True, maxY = True)
        except:
            print("Datos insuficientes para graficar: ", pNombre)

#########################################################################
def genereGraficos(pProyecto, pGeneralTmp, pLugar, tColores):
    # kpGeneralTemp   NO es vacío
    tAlgor = []
    tPatr = []
    tMeto = []
    for tNombreFil, tDatosFil in pGeneralTmp.iterrows():
        tValor = pGeneralTmp.loc[tNombreFil, "Algoritmo"]
        if not tValor in tAlgor:
            tAlgor.append(tValor)
        tValor = pGeneralTmp.loc[tNombreFil, "Patron"]
        if not tValor in tPatr:
            tPatr.append(tValor)
        tValor = pGeneralTmp.loc[tNombreFil, "Metodo"]
        if not tValor in tMeto:
            tMeto.append(tValor)      
    tTipos = {"Algoritmo": tAlgor, "Patron":tPatr , "Metodo": tMeto}
    tGrafi = [['RMSE','R2']]
    #tparams = {'legend.fontsize': 7, 'lines.linewidth':1, 'linestyle': "."}
    tparams = {'legend.fontsize': 7}
    pl.rcParams.update(tparams)
    for tGra in tGrafi:    
        for tTip in tTipos:
            tNombre = "Figuras\Scatter - " + gProyecto +" - "+pLugar+" - "+tTip.replace("*","-")+" - RELACION - "+tGra[0]+ " - "+tGra[1]
            if (tGra[0]=="CV-RMSE") or (tGra[1]=="CV-RMSE") or (tGra[0]=="RMSE") or (tGra[1]=="RMSE") :
                tLocG = 1
            else:
                tLocG = 2   
            if (tGra ==  [ "CV-RMSE", "Accuracy"]) and (pLugar == "Todos") and (tTip=="Algoritmo"):
                tQPareto = True
            else:
                tQPareto = False
            genereUnGrafico(pGeneralTmp, tColores, tTipos[tTip], tTip, 'scatter',tGra[0], tGra[1], tLocG, tNombre, tQPareto)            
                  
#########################################################################
def muestreResultadosFinales(pProyecto, pListaLugares):
    global gGeneral
    xrmse = 3
    xrmse_std = 1
    xrmse_cv = 0.20
    xr2 = 0.3
    xr2_std = 0.20
    xr2_cv = 0.20
    tCuantosHead = 80
    tColores = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", 
                    "DarkBlue", "DarkGreen", "DarkRed"]     
    # Ordena por precision                       
    gGeneral = gGeneral.sort(["Accuracy", "CV-RMSE"], ascending=[False, True]) 
    #######################################
    # Guarda la posición en que quedó inicialmente
    tNCol = len(gGeneral.columns)
    tvaloresFil = range(len(gGeneral.index))
    gGeneral.insert(tNCol, "Posicion",  tvaloresFil)      
    gGeneralTmp = gGeneral.head(tCuantosHead)     
    if len(gGeneralTmp) > 0:
        genereGraficos(pProyecto, gGeneralTmp, "Todos", tColores)
        if pListaLugares > 1:
            for pLug in pListaLugares:
                tParte = gGeneral[(gGeneral["Lugar"] == pLug )]
                if len(tParte) > 0:
                    if gMuestreAvance:
                        print("Preparando: ",pLug)
                    tParte.to_excel("Figuras\Datos - "+pLug+"  .xlsx", sheet_name=pLug, engine="openpyxl")
                    genereGraficos(pProyecto, tParte.head(tCuantosHead), pLug, tColores)
            xParaGraf = gGeneral.sort(["Patron", "Algoritmo","Metodo" ], ascending=True)
            tFilas = xParaGraf.index
            tFig = pd.DataFrame({}, columns= xParaGraf.columns)
            # este algortimo parte de que el dataframe esta ordenado por "Patron", "Algoritmo","Metodo" 
            tCont1 = 0
            while tCont1 < len(tFilas):
                tCuantasEsta = 1
                tListaIncluir = [tCont1]
                tCont2 = tCont1+1
                while tCont2 < len(tFilas):
                    if (xParaGraf.loc[tFilas[tCont1], "Patron"] == xParaGraf.loc[tFilas[tCont2], "Patron"]):
                        if (xParaGraf.loc[tFilas[tCont1], "Algoritmo"] == xParaGraf.loc[tFilas[tCont2], "Algoritmo"]) :
                            if (xParaGraf.loc[tFilas[tCont1], "Metodo"] == xParaGraf.loc[tFilas[tCont2], "Metodo"]) :
                                # Atencion, se parte que estos tres son unicos por lugar, si se hciera el mismo
                                # algortimo pero con diferentes parametros habria que revisarlo
                                tCuantasEsta += 1
                                tListaIncluir.append(tCont2)
                                if tCuantasEsta == len(pListaLugares):
                                    break
                            else:
                                break
                        else: break
                    else:
                        break
                    tCont2 += 1
                if  tCuantasEsta == len(pListaLugares):
                    for tInc in tListaIncluir:
                        tFig = tFig.append(xParaGraf.loc[tFilas[tInc] ])  
                    tCont1 = tCont2
                tCont1 += 1
        if gMuestreAvance:
            print("Elementos que están en los dos: ", len(tFig))
        #######################################
        #  Ordena los resultados por RMSE de menor a mayor considerando todos los lugares
        # Parte de que en tFig ya vienen los lugares juntos para ser promediados
        tNCol = len(tFig.columns)
        tvaloresFil = np.zeros(len(tFig.index))
        tFig.insert(tNCol, "Promedio RMSE lugares",  tvaloresFil)
        tCantLugares = len(pListaLugares)
        tCont3 = 0
        tFilasFig = tFig.index
        while tCont3 < len(tFilasFig):
            tTot1 = 0
            for tCont4 in range(tCantLugares):
                tTot1 += tFig.loc[tFilasFig[tCont3+tCont4], "RMSE"]
            for tCont4 in range(tCantLugares):
                tFig.loc[tFilasFig[tCont3+tCont4], "Promedio RMSE lugares"] = tTot1/tCantLugares
            tCont3 = tCont3 + tCantLugares
        tFig = tFig.sort(["Promedio RMSE lugares", "Patron", "Algoritmo","Metodo", "RMSE"], ascending=True)
        tNombre = "Figuras\Scatter - " +gProyecto +" - Todos los lugares - RELACION - RMSE - Accuracy"
        genereUnGrafico(tFig.head(tCuantosHead), tColores, pListaLugares, "Lugar", 'scatter',
                        "RMSE", "Accuracy", 1, tNombre, False)       
        tFig.to_excel("Figuras\Resultado - "+pProyecto+"  .xlsx", sheet_name="Resultado", engine="openpyxl")     
    else:
        print("Archivo vacio")
                    
#########################################################################
def cargueGeneral(tArchivoEntrada):
    global gGeneral
    gGeneral = pd.io.parsers.read_csv(tArchivoEntrada, sep=gSeparadorLista, header = 0)  # En fila 0 títulos
    tNFilas = len(gGeneral.index)
    tNColumnas = len(gGeneral.columns)
    gGeneral.index = list(range(0,tNFilas))    


######################################################################### 
#########################################################################
#   Programa principal
#########################################################################
#########################################################################

if __name__ == "__main__":

    ###############################################################
    #   Revisar si el programa se está corriendo con argumentos
    #   Un tipo de Algoritmo.  Debe venir de primero el tipo de algoritmo
    gSeparadorLista = ";"    
    gCuantosParam = len(sys.argv)
    gListaParam = sys.argv
    if gCuantosParam == 2:
        tAgregueAlNombre = " - " + gListaParam[1]  
    else: 
        tAgregueAlNombre = ""       
    
    
    gT ={ "Discretizacion":[], "EscalaX":[], "EscalaY":[], "MetodoEscalar":[], "Lugar":[],"Metodo": [], "Patron": [], 
         "Algoritmo":[], "Detalles": [], "Train": [], "Test": [], "Num_Columnas": [], "RMSE":[], "R2": [], "std-RMSE": [], 
            "CV-RMSE":[], "std-R2": [], "Accuracy":[], "CV-R2":[], "Precision":[], "Recall":[]}
    gGeneral = pd.DataFrame(gT)
    gEscalaParaVarAPredecir = np.array([])
    
    ###############################################################
    #  Inicializar la semilla para random
    np.random.seed(42)  # Con el fin de lograr repetibilidad    
        
    ###############################################################
    ###############################################################
    #   V A R I A B L E S    D E    C O N F I G U R A C I O N 
    ###############################################################
    ###############################################################
    
    gMinMaxScaler = 0  # Variable para el objeto Scaler
    gCantidadRangosVarAPredecir = 0
    gEscalaTres = ["bajo","medio","alto"]
    gEscalaCinco = ["muy bajo", "bajo","medio","alto", "muy alto"]                    
    gCompareGrupos = False
    gMuestreResultados = False
    gCreeArchivoGeneral = True
    gMuestreAvance = True
    ##############################################################################
    # Indica que el valor a predecir es incluido en los patrones mayores a 
    # una semana antes
    gIncluyaPredichoEnPatron = True
    ##############################################################################

    ##############################################################################
    # Indica si en lugar de predecir un valor absoluto, se desea predecir un  
    # porcentaje de cambio con respecto a la base
    gTrabajarComoTasaCambio = False
    ##############################################################################    
    
    ##############################################################################
    # Indica si antes de hacer las N corridas con un patrón, primero optimiza 
    # parámetros para esas N iteraciones
    gOptimiceParametros = True
    ##############################################################################       
    
    gNumIteraciones = 10 # Normalmente 10 folds
    gCantidadVarAPredecir = 1
    gMaxPeriodsAntes = 8  # 8 Recomendado
    gMaxSemenasPrediccion = 1  #  8 Recomendado
    
    gSVR_C =10   # se recomienda 10
    gSVR_gamma = 0.5   # o gamma de 0.5
    g_degree = 1
    gLeakingRate = 0.07  # leaking rate 0.05
    gInitLenPorcentaje = 0.05  # Recomendado
    gNumFigura = 1
    
    
    gGenereArchivoDiscretizado = False
    
    gGenereArchivoConPatronDefinido = False

    # Esta opción genera todos los archivos con el patron deseado
    # los coloca en el subdirectorio llamado Patrones
    gGenereArchivosDePatrones = False
    
    # Puede ser True o False.  Solo toma sentido si gTipoProceso="Siguiente"
    gHagaValidacion = False
    # Si gHagaValidacion=True,  indica cuantos periodos validar, 
    #   contando del ultima hacia atras (fecha mas reciente hacia mas antigua)
    gCuantosPeriodosValidar = 105
    #gCuantosPeriodosValidar = 105 o 55
    
    tTitulosRV ={ "Discretizacion":[], "EscalaX":[], "EscalaY":[], "MetodoEscalar":[], "Lugar":[],"Metodo": [], "APatron": [], 
         "Algoritmo":[], "Detalles": [], "Num_Train": [],  "Num_Validacion": [],"Valor1Real":[], "Valor2Predicho":[]}
         
         
    ######################################################################################
    tLugarValidacion = "LaRita-Lineales  R2-Stats Discr_S LINEALES opt"
    tAlgoritmosArch = "EnPat=T TasaC=F Discretice=FyT"
    gArchResultValidacion =   "2016-07-09-Sigatoka - "+tLugarValidacion+tAlgoritmosArch+ tAgregueAlNombre+" - "+str(gCuantosPeriodosValidar)
    gArchDelProyectoInicial = "2016-07-09-Sigatoka - "+tLugarValidacion+tAlgoritmosArch + tAgregueAlNombre
    gNombreArchivoConPatronDefinido = "2016-07-09-Sigatoka - "+tLugarValidacion+tAlgoritmosArch + tAgregueAlNombre
    ######################################################################################
    
    
    # tipos de proceso   "Pruebas"    "Siguiente"
    gTipoProceso = "Pruebas"
    # El patrón se da don dos digítos en el antes y dos en el despues

    gListaTipoPatron =  [  " 4S-Period-P 1",  " 5S-Period-P 1", " 7S-Period-P 2",  "12S-Period-P 2"]

    gListaTipoPatron =  [  " 4S-Period-P 1",  " 5S-Period-P 1"]

    gListaTipoPatron =  [  " 1S-Period-P 1",  " 2S-Period-P 1", " 2S-Period-P 2",  " 3S-Period-P 2" ]     


    gListaTipoPatron =  [  " 1S-Period-P 1",  " 2S-Period-P 1"]

    gListaTipoPatron =  [  " 1S-Period-P 1",  " 2S-Period-P 1", " 2S-Period-P 2",  " 3S-Period-P 2",
                           " 8S-Period-P 3",  "10S-Period-P 3", "12S-Period-P 3"]    
                          

    gListaTipoPatron =  [  " 6S-Period-P 1",  "10S-Period-P 1", " 8S-Period-P 2",  "11S-Period-P 2","12S-Period-P 2",
                           " 8S-Period-P 3",  "10S-Period-P 3", "12S-Period-P 3"]     
 
    gListaTipoPatron =  [    " 1S-Period-P 1",  " 2S-Period-P 1", " 3S-Period-P 1",  " 4S-Period-P 1",
                             " 6S-Period-P 1", " 7S-Period-P 1", " 8S-Period-P 1", " 9S-Period-P 1",
                             "10S-Period-P 1", "11S-Period-P 1","12S-Period-P 1", " 5S-Period-P 1",
                             " 1S-Period-P 2", " 2S-Period-P 2", " 3S-Period-P 2", " 4S-Period-P 2",
                             " 6S-Period-P 2", " 7S-Period-P 2", " 8S-Period-P 2", " 9S-Period-P 2",
                             "10S-Period-P 2", "11S-Period-P 2", "12S-Period-P 2", " 5S-Period-P 2",
                             " 1S-Period-P 3", " 2S-Period-P 3", " 3S-Period-P 3", " 4S-Period-P 3",
                             " 6S-Period-P 3", " 7S-Period-P 3", " 8S-Period-P 3", " 9S-Period-P 3",
                             "10S-Period-P 3", "11S-Period-P 3", "12S-Period-P 3", " 5S-Period-P 3"
                        ]     
    gListaTipoPatron =  [  " 4S-Period-P 1"]  
    #   Ejemplos:  "SVR model con Kernel rbf",  "SVR model con Kernel linear",  "SVR model con Kernel sigmoid" 
    #               "ESN",  "LinearRegression",  "linear_model.Lasso",  "Ridge",  "LDA" , "newMethod2" , ElasticNet
    gListaTipoAlgoritmo = [ "SVR model con Kernel linear" ]  
    gListaTipoAlgoritmo = [ "LinearRegression", "Ridge", "ElasticNet" , "ESN"]   
    gListaTipoAlgoritmo = [ "LinearRegression" ]  
    
    #   Ejemplos:  ['Todas', 'NecesariasConfig', 'Entropia', 'RoughSet', 'Unica*TAireMax', 'Unica*TAireMin', 
    #   'Unica*MinHumed', 'Unica*MaxHumed', 'Unica*Rasolwm', 'Unica*VelViMax', 'Solo*TempeAire', 
    #   'Solo*Humedad', 'Solo*Precipita', 'Solo*VelViento', 'Par*TempeAire*Humedad',
    #    'Par*TempeAire*Precipita', 'Par*TempeAire*VelViento', 'Par*Humedad*Precipita', 
    #   'Par*Humedad*VelViento', 'Par*Precipita*VelViento']

  
    gListaTipoMetodo = ["Todas"]

    
    gListaTipoMetodo = ["Todas", "NecesariasConfig", "Par*1*3", "Par*1*4"]
    
    gListaTipoMetodo = ["Unica*1", "Todas", "NecesariasConfig", "Par*1*3", "Par*1*4"] 
    gListaTipoMetodo = [ "Par*1*4"]     
    
    if gTipoProceso == "Pruebas":    # Para que lo ejecute solo una vez con Pruebas
        gListaTipoPatron = ["NA"]
        gListaTipoAlgoritmo = ["NA"]
        gListaTipoMetodo = ["NA"]

    #  Si gDiscreticeEspecial es True, no importa lo que diga el archivo de configuacion sino que 
    # calcula el coeficiente de variacion del atributo y con el determina los rangos
    gListaDiscreticeEspecial = [ True, False]
         
    gListagEscaleDatosEntreAyB = [True, False]
    gListagEscaleY = [False]
    gListagMetodoParaEscalar = [5]
    #gListagLimiteNM = [ 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 0.90 ]
    gListagLimiteNM = [ 0.30]
    
    gHagaEntropia = False  # Según archivo de configuración, Columna: entropia
    gHagaRoughSet = False  # Según archivo de configuración, Columna: roughset
    gHagaReducida = True   # Según archivo de configuración, Columna: reducida
                            # si no se conisedera: "n".   Y luego con 1,2,3,...  se agrupan las variables a usar   
    gHagaNecesarias = True  # Según archivo de configuración, Columna: necesaria
    gHagaPares = True   # Según archivo de configuración. Hace pares de todos los que estén en reducida
    gHagaCombinatoria = True # Produce las combinaciones entre todas las 
            # variables en Reducida a partir de tres en tres y sin incluir
            # todas
    
    
    if gMuestreAvance:
        print(time.strftime("%a, %d %b %Y %H:%M:%S"))
    
    tPrimeraVez = True    
    gResultadoValidacion = pd.DataFrame(tTitulosRV)
    
    for gDiscreticeEspecial in gListaDiscreticeEspecial:
        for gTipoPatron in gListaTipoPatron :
            for gTipoAlgoritmo in gListaTipoAlgoritmo :
                for gTipoMetodo in gListaTipoMetodo : 
                                  
                    gResultadosSiguiente = []
                    gListaValoresReales = []
                    
                    for contEscalaAyB in gListagEscaleDatosEntreAyB:
                        for contEscaleYin in gListagEscaleY:
                            for contMetodoEscalar in gListagMetodoParaEscalar:
                                for contLimiteNM in gListagLimiteNM:
                                    
                                    if gDiscreticeEspecial == True:
                                        gTipoDiscretizacion = "Discretice-Esp"
                                    else:
                                        gTipoDiscretizacion = "Segun-Config"
                                    gEscaleDatosEntreAyB = contEscalaAyB
                                    gEscaleY = contEscaleYin
                                    gMetodoParaEscalar = contMetodoEscalar
                                    gLimiteNM = contLimiteNM
                    
                    ##################################################################################################################
                    #####################
                                    #########################################################################################################
                                    ##################################################################################################################                
                                    
                                    ####################################################################################
                                    #                                   C I C A F E - Barba - Mensual - Roya
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    #gProyecto = "GENERAL - CICAFE-Roya - Barba - IncidenciaRoya"
                                    #gListaLugares = ["CICAFE"]
                                    #gListaDirectorios = ["C:/AAA-Doctorado/CORRIDAS DEL PROGRAMA/CICafe/2015-11"]
                                    #gListaArchivosEntrada = ["2015-11-11-Roya-Barba-Mensual-03-CSV.csv"]
                                    #gListaArchivosConfig = ["2015-11-11- Roya- Configurar - Mensual.csv"]
                                    #########################################################################################################
                                    
                                    #########################################################################################################
                                    #                                   C I C A F E - Poas - Mensual - Roya
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    #gProyecto = "GENERAL - CICAFE-Roya - Poas - IncidenciaRoya"
                                    #gListaLugares = ["CICAFE"]
                                    #gListaDirectorios = ["D:/Cambios/Doctorado/CORRIDAS DEL PROGRAMA/CICafe/2015-11"]
                                    #gListaArchivosEntrada = ["2015-11-11-Roya-Poas-Mensual-03-CSV.csv"]
                                    #gListaArchivosConfig = ["2015-11-11- Roya- Configurar - Mensual.csv"]
                                    #########################################################################################################
                              
                                    #########################################################################################################
                                    #                                   C I C A F E - San Vito - Mensual - Roya
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    #gProyecto = "GENERAL - CICAFE-Roya - San Vito - IncidenciaRoya"
                                    #gListaLugares = ["CICAFE"]
                                    #gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\CICafe\Mensual\San Vito"]
                                    #gListaArchivosEntrada = ["2015-05-05- CICAFE - ROYA - 03 - Listo procesar - San Vito.csv"]
                                    #gListaArchivosConfig = ["CICAFE- Roya- Configurar - Mensual.csv"]
                                    #########################################################################################################
                    
                    
                    #########################################################################################################
                    
                    
                                    #########################################################################################################
                                    #                                   C I C A F E - Barba - Semanal - Roya
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    #gProyecto = "GENERAL - CICAFE-Roya - Barba SEMANAL - IncidenciaRoya"
                                    #gListaLugares = ["CICAFE"]
                                    #gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\CICafe\Semanal\Barba"]
                                    #gListaArchivosEntrada = ["2015-11-01-Entrada clima barba semanal - ajustado CSV.csv"]
                                    #gListaArchivosConfig = ["CICAFE- Roya- Configurar - Semanal.csv"]
                                    #########################################################################################################
                    
                    
                    ##################################################################################################################
                    ##################################################################################################################                
                                                                 
                                    #########################################################################################################
                                    #########################################################################################################
                                    #                                   C O R B A N A - peso-neto-kg - Semanal
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    #gProyecto = "GENERAL - Corbana-Produccion - peso-neto-kg  - Semanal"
                                    #gListaLugares = ["28 Millas - Produccion - peso-neto-kg - semanal"]
                                    #gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\Corbana\Floracion\Semanal\peso-neto-kg"]
                                    #gListaArchivosEntrada = ["2015-01-21- 28 Millas - peso-neto-kg 03 - CSV.csv"]
                                    #gListaArchivosConfig = ["28 Millas - peso-neto-kg - Configurar.csv"]
                                    #########################################################################################################
                    
                    
                                    #########################################################################################################
                                    #                                   C O R B A N A - Embolse-ha - Semanal
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    #gProyecto = "GENERAL - Corbana-Produccion - Embolse-ha  - Semanal"
                                    #gListaLugares = ["28 Millas - Produccion - Embolse-ha - semanal"]
                                    #gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\Corbana\Floracion\Semanal\embolse-ha"]
                                    #gListaArchivosEntrada = ["2015-01-21- 28 Millas - embolse-ha - 03 - CSV.csv"]
                                    #gListaArchivosConfig = ["28 Millas - embolse-ha - Configurar.csv"]
                                    #########################################################################################################
                    
                                    #########################################################################################################
                                    #                                   C O R B A N A - racimos-cortados - Semanal
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    #gProyecto = "GENERAL - Corbana-Produccion - racimos-cortados  - Semanal"
                                    #gListaLugares = ["28 Millas - Produccion - racimos-cortados - semanal"]
                                    #gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\Corbana\Floracion\Semanal\\racimos-cortados"]
                                    #gListaArchivosEntrada = ["2015-01-21- 28 Millas - racimos-cortados - 03 - CSV.csv"]
                                    #gListaArchivosConfig = ["28 Millas - racimos-cortados - Configurar.csv"]
                                    #########################################################################################################
                    
                    
                    
                    ##################################################################################################################
                    ##################################################################################################################                
                    
                                    #########################################################################################################
                                    #                                   C O R B A N A - Sigatoka - semanal
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    gListaLugares = ["La Rita - Sigatoka - Semanal"]
                                    gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\Corbana\Sigatoka - Semanal - EstadoEvolucion"]
                                    
                                    ###  Revisar                                    
                                    gListaArchivosEntrada = ["2015-12-01- La Rita - Enfermedad - Ultima Semana  es 46 CSV - SIN SEVERIDAD.csv"]  
                                    #gListaArchivosEntrada = ["2015-12-01- La Rita - Enfermedad - Ultima Semana  es 46 CSV.csv"]  
                                    #gListaArchivosEntrada = ["2015-12-01- La Rita - Enfermedad - Ultima Semana  es 46 CSV SECUENCIA.csv"]
                                    #gListaArchivosEntrada = ["2015-12-01- AMBAS - Sigatoka CSV.csv"]      
                                    ######
                                                                        
                                    gListaArchivosConfig = [ "2015-12-01-Sigatoka - Configurar.csv" ]
                                    
                                    #gListaLugares = ["28 Millas - Sigatoka - Semanal"]
                                    #gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\Corbana\Sigatoka - Semanal - EstadoEvolucion"]
                                    #gListaArchivosEntrada = ["2015-12-01- 28 Millas - Enfermedad - Ultima Semana  es 46 CSV.csv"]
                                    #gListaArchivosConfig = [ "2015-12-01-Sigatoka - Configurar.csv" ]
                                    
                                    ###############################################################
                    
                    
                                    #########################################################################################################
                                    #                                   La Rita ---- C O R B A N A - Sigatoka - semanal - Validacion
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    #gListaLugares = ["La Rita - Sigatoka - Semanal"]
                                    #gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\Corbana\Validación\La Rita"]
                                    #gListaArchivosEntrada = ["2015-05-02- La Rita - Enfermedad - Validacion - CSV - Ultima Semana  es 17.csv" ]
                                    #gListaArchivosConfig = [ "La Rita - Sigatoka - Configurar - Semanal.csv" ]
                                    ###############################################################
                    
                                    #########################################################################################################
                                    #                                   28 Millas ---- C O R B A N A - Sigatoka - semanal - Validacion
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    #gListaLugares = ["28 Millas - Sigatoka - Semanal"]
                                    #gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\Corbana\Validación\\28 Millas"]
                                    #gListaArchivosEntrada = ["2015-05-02- 28 Millas- Enfermedad - Validacion - CSV - Ultima Semana  es 17.csv" ]
                                    #gListaArchivosConfig = [ "28 Millas - Sigatoka - Configurar - Semanal.csv" ]
                                    ###############################################################
                    
                    
                                    #########################################################################################################
                                    #                                   La Rita ---- C O R B A N A - Sigatoka - diario - Validacion
                                    #########################################################################################################
                                    #gListaLugares = [ "La Rita - Sigatoka - Diario"]
                                    #gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\Corbana\Sigatoka - Diario - EstadoEvolucion"]
                                    #gListaArchivosEntrada = ["Diarios - La Rita- Datos Climatológicos de toda la semana MAS Enfermedad FINAL.csv"]
                                    #gListaArchivosConfig = ["La Rita - Sigatoka - Configurar - Diario.csv" ]
                                    #########################################################################################################
                    
                    
                                    #########################################################################################################
                                    #                                   C O R B A N A - Sigatoka - IND - Semanal
                                    #########################################################################################################
                                    # Esta es la configuración que habría que cambiar al probar con otras bases de datos
                                    #gProyecto = "GENERAL - Corbana- Sigatoka - IND  - Semanal"
                                    #gListaLugares = [  "28 Millas - IND - semanal1", "28 Millas - IND - semanal2",
                                    #                   "28 Millas - IND - semanal3", "28 Millas - IND - semanal4",
                                    #                   "28 Millas - IND - semanal5", "28 Millas - IND - semanal6",
                                    #                   "28 Millas - IND - semanal7", "28 Millas - IND - semanal8",
                                    #                   "28 Millas - IND - semanal9", "28 Millas - IND - semanal10" ]
                                    #gListaDirectorios = ["D:\Cambios\Doctorado\CORRIDAS DEL PROGRAMA\Corbana\Experimentos-IND"] * 10
                                    #gListaArchivosEntrada = ["Floracion-Planta-1.csv", "Floracion-Planta-2.csv","Floracion-Planta-3.csv",
                                    #                           "Floracion-Planta-4.csv","Floracion-Planta-5.csv","Floracion-Planta-6.csv",
                                    #                           "Floracion-Planta-7.csv","Floracion-Planta-8.csv","Floracion-Planta-9.csv",
                                    #                           "Floracion-Planta-10.csv"]
                                    #gListaArchivosConfig = ["28 Millas - IND - Configurar.csv"] * 10
                                    #########################################################################################################
                    
                    
                                  
                    ###############################################################################################################                
                    ###############################################################################################################                
                                    tNombreT = gArchDelProyectoInicial  + " - " + gTipoDiscretizacion
                                    tNombreT = tNombreT.replace("[","")
                                    tNombreT = tNombreT.replace("]","")
                                    tNombreT = tNombreT.replace("'","")                
                                    gProyecto = tNombreT                                
                                    #########################################################################################################
                                    if gCreeArchivoGeneral:
                                        for gCor in range(len(gListaLugares)):
                                            EjecuteSVR_ESN(gListaDirectorios[gCor], gListaLugares[gCor], gListaArchivosEntrada[gCor], 
                                                           gListaArchivosConfig[gCor], gTipoProceso, gTipoPatron,  gTipoAlgoritmo, gTipoMetodo )    
                                        gGeneralFile = open( gProyecto + ".csv", "a") # Para Todos
                                        if tPrimeraVez:
                                            gGeneralFile.write("Discretizacion;EscalaX;EscalaY;MetodoEscalar;Lugar;Metodo;Patron;Algoritmo;Detalles;Train;Test;Num_Columnas;RMSE;R2;std-RMSE;CV-RMSE;std-R2;Accuracy;CV-R2;Precision;Recall"+"\n")
                                            tPrimeraVez = False
                                        for gNombreFil, gDatosFil in gGeneral.iterrows():
                                            gSalida = gGeneral.loc[gNombreFil, "Discretizacion" ]
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "EscalaX" ])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "EscalaY" ])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "MetodoEscalar" ])
                                            gSalida += ";"+ gGeneral.loc[gNombreFil, "Lugar" ]
                                            gSalida += ";"+ gGeneral.loc[gNombreFil, "Metodo"]
                                            gSalida += ";"+ gGeneral.loc[gNombreFil, "Patron"]
                                            gSalida += ";"+ gGeneral.loc[gNombreFil, "Algoritmo"]
                                            gSalida += ";"+ gGeneral.loc[gNombreFil, "Detalles"]
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "Train"])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "Test"])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "Num_Columnas"])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "RMSE"])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "R2"])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "std-RMSE"])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "CV-RMSE"])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "std-R2"])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "Accuracy"])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "CV-R2"])
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "Precision"]) 
                                            gSalida += ";"+ str(gGeneral.loc[gNombreFil, "Recall"]) + "\n"  
                                            gGeneralFile.write(gSalida)
                                        gGeneralFile.close()  
                                        #######################################
                                    else: # Tiene que cargarlos
                                        cargueGeneral(gProyecto + ".csv")
                                        
                                    if gMuestreResultados and (gTipoProceso != "Siguiente"):
                                        # Agrega dos columnas a partir de Patron, cuantas semanas antes y cuantas después
                                        if gMuestreAvance:
                                            print("Incluyendo columnas: Periods antes y Periods despues")
                                        tNCol = len(gGeneral.columns)
                                        tvaloresFil = gGeneral["Patron"]
                                        gGeneral.insert(tNCol, "Periodos antes",  tvaloresFil) 
                                        tNCol = len(gGeneral.columns)
                                        tvaloresFil = gGeneral["Patron"]
                                        gGeneral.insert(tNCol, "Periodos despues",  tvaloresFil)       
                                        tFilas = gGeneral.index
                                        tCont = 0
                                        while tCont < len(tFilas):
                                            gGeneral.loc[tFilas[tCont], "Periodos antes"] = gGeneral.loc[tFilas[tCont], "Periodos antes"][0:2]
                                            gGeneral.loc[tFilas[tCont], "Periodos despues"] = gGeneral.loc[tFilas[tCont], "Periodos despues"][12:]        
                                            tCont += 1
                                        muestreResultadosFinales(gProyecto, gListaLugares)
                    if gMuestreAvance:   
                        if gHagaValidacion or (gTipoProceso == "Siguiente") :
                            print("Predichos:")
                            print(gResultadosSiguiente)
                            print("Reales: ")
                            print(gListaValoresReales)
                            print("================")
                            #print(gResultadoValidacion)
    
    
    if gHagaValidacion or (gTipoProceso == "Siguiente") :
        #gResultadoValidacion = gResultadoValidacion.sort([ "Num_Validacion"], ascending=True)
        gResultadoValidacion.to_excel(gArchResultValidacion + ".xlsx",sheet_name="Resultado", engine="openpyxl")    
                    
    
    print(time.strftime("%a, %d %b %Y %H:%M:%S"))
    
    #########################################################################
    #   Fin del programa
    ##########################################################################
